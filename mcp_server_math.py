#
# Simple MCP server for maths
#
# - Uses stdio
#
from mcp.server.fastmcp import FastMCP
import sys
import logging

#logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

mcp = FastMCP("Math")
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")


