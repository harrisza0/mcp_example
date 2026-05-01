"""
Interactive Chat client for our MCP server and Ollama API.

Prereqs:
    pip install mcp ollama
    ollama pull gemma4:e2b   # any tool-capable model (qwen2.5, mistral-nemo, ...)
    python server.py       # in another terminal

Run:
    python ollama_client.py

Type a message and press Enter. Type 'exit' or Ctrl+C to quit.
The conversation history is preserved across turns.
"""

import asyncio
import json

import ollama
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MODEL = "gemma4:e2b"
MCP_URL = "http://localhost:8000/mcp"


def mcp_tools_to_ollama(mcp_tools):
    """Convert MCP tool defs to Ollama's OpenAI-style tool schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]   


async def run_turn(session, ollama_tools, messages):
    """Send one user turn through the model, handling tool calls until it answers."""
    while True:
        resp = ollama.chat(model=MODEL, messages=messages, tools=ollama_tools) #using ollama chat API to send the conversation history and tool definitions to the model.
        msg = resp["message"]
        messages.append(msg) 
        #we append all messages to the history and send it back to the model on each turn
        #so it has the full conversation context and tool call history

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            print(f"\nAssistant: {msg.get('content', '')}\n")
            return

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            print(f"  >> tool: {name}({args})")
            result = await session.call_tool(name, args)
            text = "".join(getattr(b, "text", "") for b in result.content)
            messages.append({"role": "tool", "name": name, "content": text})


async def main():
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            ollama_tools = mcp_tools_to_ollama(tools)

            tool_names = ", ".join(t.name for t in tools) or "(none)"
            #print(tools)
            print(f"Connected. Model: {MODEL}. Tools: {tool_names}")
            print("Type 'exit' or Ctrl+C to quit.\n")

            messages = [] #reset messages at the start of the conversation
            while True:
                try:
                    user = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user:
                    continue
                if user.lower() in {"exit", "quit"}:
                    break

                messages.append({"role": "user", "content": user})
                await run_turn(session, ollama_tools, messages)


if __name__ == "__main__":
    asyncio.run(main())
