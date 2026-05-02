"""
Very simple MCP server exposing two tools: `ping` and `traceroute`.

Runs locally over HTTP (Streamable HTTP transport), bound to 127.0.0.1 on port 8000

Run:
    pip install "mcp[cli]"
    python server.py

Server will listen on http://127.0.0.1:8000/mcp
"""

import json
import re
import shutil
import subprocess
import sys

from mcp.server.fastmcp import FastMCP

# name is what shows up in MCP clients
mcp = FastMCP("net-tools", host="127.0.0.1", port=8000)


def _run(cmd: list[str], timeout: int = 30) -> str:
    """Run a shell command and return combined stdout/stderr as text."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,  # don't use the shell, to avoid quoting issues and security risks
        )
        out = result.stdout or ""
        err = result.stderr or ""
        return (out + ("\n" + err if err else "")).strip() or "(no output)"
    except FileNotFoundError:
        return f"Error: '{cmd[0]}' not found on this system."
    except subprocess.TimeoutExpired:
        return f"Error: '{' '.join(cmd)}' timed out after {timeout}s."


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _as_tool_result(source: str, payload: dict) -> str:
    """Wrap a JSON payload in an <untrusted> block so the LLM treats it as data."""
    return (
        f'<untrusted source="{source}">\n'
        + json.dumps(payload, indent=2)
        + "\n</untrusted>"
    )


def parse_ping(raw: str, host: str) -> dict:
    """Parse `ping` output (macOS / Linux / Windows) into a structured dict."""
    out: dict = {"host": host, "raw": raw}

    # Per-reply lines: "64 bytes from 1.2.3.4: icmp_seq=0 ttl=56 time=11.4 ms"
    replies = []
    for m in re.finditer(
        r"(?:icmp_seq|seq)[=\s](\d+).*?ttl[=\s](\d+).*?time[=<]?\s*([\d.]+)\s*ms",
        raw,
        re.IGNORECASE,
    ):
        replies.append({
            "seq": int(m.group(1)),
            "ttl": int(m.group(2)),
            "rtt_ms": float(m.group(3)),
        })
    if replies:
        out["replies"] = replies

    # Summary line, Linux/macOS:
    #   "4 packets transmitted, 4 received, 0% packet loss, time 3004ms"
    #   "4 packets transmitted, 4 packets received, 0.0% packet loss"
    m = re.search(
        r"(\d+)\s+packets transmitted,\s+(\d+)\s+(?:packets )?received,\s+([\d.]+)%\s+packet loss",
        raw,
    )
    if m:
        out["transmitted"] = int(m.group(1))
        out["received"] = int(m.group(2))
        out["loss_pct"] = float(m.group(3))

    # Windows: "Packets: Sent = 4, Received = 4, Lost = 0 (0% loss)"
    m = re.search(
        r"Sent\s*=\s*(\d+),\s*Received\s*=\s*(\d+),\s*Lost\s*=\s*(\d+)\s*\((\d+)%\s*loss\)",
        raw,
    )
    if m:
        out["transmitted"] = int(m.group(1))
        out["received"] = int(m.group(2))
        out["lost"] = int(m.group(3))
        out["loss_pct"] = float(m.group(4))

    # RTT stats: "min/avg/max/stddev = 10.9/11.4/12.1/0.4 ms"  (mdev on Linux)
    m = re.search(
        r"min/avg/max/(?:stddev|mdev)\s*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)",
        raw,
    )
    if m:
        out["rtt_ms"] = {
            "min": float(m.group(1)),
            "avg": float(m.group(2)),
            "max": float(m.group(3)),
            "stddev": float(m.group(4)),
        }
    else:
        # Windows: "Minimum = 10ms, Maximum = 12ms, Average = 11ms"
        m = re.search(
            r"Minimum\s*=\s*(\d+)ms,\s*Maximum\s*=\s*(\d+)ms,\s*Average\s*=\s*(\d+)ms",
            raw,
        )
        if m:
            out["rtt_ms"] = {
                "min": float(m.group(1)),
                "max": float(m.group(2)),
                "avg": float(m.group(3)),
            }

    out["ok"] = out.get("received", 0) > 0
    return out


def parse_traceroute(raw: str, host: str) -> dict:
    """Parse `traceroute` / `tracert` output into a list of hops."""
    hops: list[dict] = []
    # Match lines starting with a hop number.
    # Examples:
    #   " 1  router.local (192.168.1.1)  1.234 ms  1.111 ms  1.000 ms"
    #   " 2  * * *"
    #   "  3    11 ms    10 ms    12 ms  some.host [1.2.3.4]"   (Windows)
    line_re = re.compile(r"^\s*(\d+)\s+(.*)$")
    rtt_re = re.compile(r"([\d.]+)\s*ms")
    host_ip_re = re.compile(
        r"([A-Za-z0-9_.\-]+)\s*[\(\[]\s*([0-9a-fA-F:.]+)\s*[\)\]]"
    )

    for line in raw.splitlines():
        m = line_re.match(line)
        if not m:
            continue
        hop_num = int(m.group(1))
        rest = m.group(2)

        rtts = [float(x) for x in rtt_re.findall(rest)]
        timeouts = rest.count("*")

        host_match = host_ip_re.search(rest)
        if host_match:
            hop_host: str | None = host_match.group(1)
            hop_ip: str | None = host_match.group(2)
        else:
            # Bare IP fallback
            ip_only = re.search(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", rest)
            hop_host = None
            hop_ip = ip_only.group(1) if ip_only else None

        hops.append({
            "hop": hop_num,
            "host": hop_host,
            "ip": hop_ip,
            "rtt_ms": rtts,
            "timeouts": timeouts,
        })

    return {
        "host": host,
        "hop_count": len(hops),
        "reached": bool(hops) and hops[-1]["ip"] is not None and hops[-1]["timeouts"] == 0,
        "hops": hops,
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()  # registers the function as an MCP tool
def ping(host: str, count: int = 4) -> str:
    """Ping a host and return structured JSON with RTT stats.

    Args:
        host: Hostname or IP address to ping (e.g. "example.com").
        count: Number of echo requests to send (1-20).
    """
    count = max(1, min(int(count), 20))
    flag = "-n" if sys.platform.startswith("win") else "-c"
    raw = _run(["ping", flag, str(count), host], timeout=count * 2 + 10)
    return _as_tool_result("ping", parse_ping(raw, host))


@mcp.tool()  # registers the function as an MCP tool
def traceroute(host: str, max_hops: int = 30) -> str:
    """Run a traceroute and return structured JSON with one entry per hop.

    Args:
        host: Hostname or IP address (e.g. "example.com").
        max_hops: Maximum number of hops (1-64).
    """
    max_hops = max(1, min(int(max_hops), 64))
    if sys.platform.startswith("win"):
        cmd = ["tracert", "-h", str(max_hops), host]
    else:
        binary = "traceroute" if shutil.which("traceroute") else "tracepath"
        cmd = (
            [binary, "-m", str(max_hops), host]
            if binary == "traceroute"
            else [binary, host]
        )
    raw = _run(cmd, timeout=60)
    return _as_tool_result("traceroute", parse_traceroute(raw, host))


if __name__ == "__main__":
    # Streamable HTTP transport — the MCP-over-HTTP transport.
    # Endpoint: http://<host>:8000/mcp
    mcp.run(transport="streamable-http")
