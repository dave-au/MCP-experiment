#
# Simple chatbot in langchain
#
import os
import logging
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# Want debugging?
# logging.getLogger("langchain_mcp_adapters").setLevel(logging.DEBUG)
logging.getLogger("langchain_mcp_adapters").setLevel(logging.WARNING)
# logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp").setLevel(logging.WARNING)

# OPENAI_API_KEY needs to be set in env var
load_dotenv()

SYSTEM_PROMPT = SystemMessage(
    content="You have tools. Prefer calling them when the user asks for something."
)

async def main():

    #
    # Connect to MCP server/s
    #
    mcp_client =  MultiServerMCPClient({
        # Stdio connector, no tap
        #"math": {"transport": "stdio", "command": "python3", "args": ["mcp_server_math.py"]},

        # Stdio connector with tap (for traffic inspection)
        # "math": {"transport": "stdio", "command": "python3", "args": ["mcp_stdio_tap.py", "--", "python3", "mcp_server_math.py"]},
        "math": {"transport": "stdio", "command": "python3", "args": ["mcp_stdio_tap.py", "--logfile", "mcpmath.log", "--pretty", "--", "python3", "mcp_server_math.py"]},

        # Stdio connection, no tap
        "shell": {"transport": "stdio", "command": "python3", "args": ["mcp_server_shell.py"]},
    })

    # Keep the session open while the agent uses the tools
    async with mcp_client.session("math") as math_session, mcp_client.session("shell") as shell_session:

        # Load MCP tools into LangChain
        tools = []
        tools += await load_mcp_tools(math_session)
        tools += await load_mcp_tools(shell_session)

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

        model = ChatOpenAI(model="gpt-4o")

        # Create an agent that uses the discovered tools
        app = create_react_agent(model, tools)

        # Keep running history so the agent has context. Start with the system prompt
        messages = [SYSTEM_PROMPT]

        # Interactive loop for the chatbot
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "gtfo"]:
                break

            messages.append(HumanMessage(content=user_input))

            result = await app.ainvoke({"messages": messages})

            messages = result["messages"]
            bot_reply = messages[-1].content
            print(f"Bot: {bot_reply}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

