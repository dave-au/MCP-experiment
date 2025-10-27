#
# Simple chatbot in langchain
#
import os
import logging
import asyncio
import argparse
import json
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from llm_and_tools_log import LLMAndToolLogger
from typing import Any, Optional, Dict
from langchain_core.tools import BaseTool
from contextlib import AsyncExitStack


# Want debugging?
# logging.getLogger("langchain_mcp_adapters").setLevel(logging.DEBUG)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.WARNING)
# logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp").setLevel(logging.WARNING)

# OPENAI_API_KEY needs to be set in env var
load_dotenv()

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You have tools. Prefer calling them when the user asks for something.\n"
        "When callings tools, it is most reliable if you use the exact parameter names from the tool schema. "
        "Do not invent names. Do not use positional arrays. Send a JSON object with the schema keys."
    )

)


def load_mcp_config_yaml(path: str = "mcp_servers.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    servers = cfg.get("servers", {})
    if not isinstance(servers, dict) or not servers:
        raise ValueError("mcp_servers.yaml has no 'servers' section.")
    return servers


def build_client_map(servers_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    client_map: Dict[str, Dict[str, Any]] = {}
    for name, spec in servers_cfg.items():
        transport = str(spec.get("transport", "")).lower().replace("_", "-")
        if transport == "stdio":
            cmd = spec.get("command")
            args = spec.get("args", [])
            if not cmd:
                raise ValueError(f"[{name}] stdio requires 'command'")
            client_map[name] = {"transport": "stdio", "command": cmd, "args": list(args)}
        elif transport == "streamable_http":
            url = spec.get("url")
            if not url:
                raise ValueError(f"[{name}] streamable_http requires 'url'")
            client_map[name] = {"transport": "streamable_http", "url": url}
        else:
            raise ValueError(f"[{name}] unknown transport: {transport!r}")
    return client_map


class ConfirmTool(BaseTool):
    """A proxy tool that asks the human to confirm before calling the real tool."""
    name: str
    description: str
    _inner: Any  # the real tool

    def __init__(self, inner_tool: Any):
        super().__init__(name=getattr(inner_tool, "name", "tool"),
                         description=getattr(inner_tool, "description", ""))
        self._inner = inner_tool

    # ---- sync path (rarely used in your setup) ----
    def _run(self, tool_input: Any, run_manager: Optional[Any] = None) -> Any:
        pretty = _pretty(tool_input)
        prompt = f"\n[HITL] Call tool '{self.name}' with args:\n{pretty}\nProceed? [y/N]: "
        resp = input(prompt).strip().lower()
        if resp not in ("y", "yes"):
            raise RuntimeError(f"User declined calling tool '{self.name}'.")
        return self._inner.invoke(tool_input)

    # ---- async path (what MCP tools use) ----
    async def _arun(self, tool_input: Any, run_manager: Optional[Any] = None) -> Any:
        pretty = _pretty(tool_input)
        prompt = f"\n[HITL] Call tool '{self.name}' with args:\n{pretty}\nProceed? [y/N]: "
        # use a thread so we don't block the event loop
        resp = (await asyncio.to_thread(input, prompt)).strip().lower()
        if resp not in ("y", "yes"):
            raise RuntimeError(f"User declined calling tool '{self.name}'.")
        # delegate to the real tool (prefer async)
        if hasattr(self._inner, "ainvoke"):
            return await self._inner.ainvoke(tool_input)
        return self._inner.invoke(tool_input)


def _pretty(obj: Any) -> str:
    try:
        # If obj is a JSON string, parse it; otherwise dump dict/list directly.
        if isinstance(obj, str):
            try:
                return json.dumps(json.loads(obj), ensure_ascii=False, indent=2)
            except Exception:
                return obj
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False, indent=2)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


async def main():
    # Parse CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-tools", action="store_true", help="Require human confirmation before each tool call.")
    args = ap.parse_args()

    # 1) Load YAML config → build client map
    servers_cfg = load_mcp_config_yaml("mcp_servers.yaml")
    client_map = build_client_map(servers_cfg)

    # 2) Create MCP client from config
    mcp_client = MultiServerMCPClient(client_map)

    # 3) Open all sessions dynamically and load tools
    tools = []
    async with AsyncExitStack() as stack:
        sessions = {}
        for name in client_map.keys():
            sess = await stack.enter_async_context(mcp_client.session(name))
            sessions[name] = sess

        for name, sess in sessions.items():
            session_tools = await load_mcp_tools(sess)
            # print(f"[{name}] tools: {[t.name for t in session_tools]}")  # optional
            tools.extend(session_tools)

        # (Debug) List the tools that we found
        #print("Loaded tools:", [t.name for t in tools])

        # (Debug) Test to tool directly to make sure it works
        #for t in tools:
        #    if t.name.endswith("multiply"):
        #        res = await t.ainvoke({"a": 2, "b": 3})
        #        print("Direct tool test multiply(2,3) -> ", res)
        #    if t.name.endswith("add"):
        #        res = await t.ainvoke({"a": 1, "b": 2})
        #        print("Direct tool test add(1,2) -> ", res)

        if args.confirm_tools:
            tools = [ConfirmTool(t) for t in tools]
            print("[HITL] Human confirmation is ENABLED for tool calls.")

        model = ChatOpenAI(model="gpt-4o")

        # Create an agent that uses the discovered tools
        app = create_react_agent(model, tools)

        # Keep running history so the agent has context. Start with the system prompt
        messages = [SYSTEM_PROMPT]

        # Create the transcript logger (file only; no console spam)
        logger = LLMAndToolLogger(
            logfile="llm_and_tools_wire.log",
            also_print=False,
            redact=[os.environ.get("OPENAI_API_KEY")],
            pretty_json=True,
            max_schema_chars=4000,
            max_tool_result_chars=4000,
        )

        # Interactive loop for the chatbot
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "gtfo"]:
                break

            messages.append(HumanMessage(content=user_input))

            result = await app.ainvoke(
                {"messages": messages},
                config={"callbacks": [logger]},
            )

            messages = result["messages"]
            bot_reply = messages[-1].content
            print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

