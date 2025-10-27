#
# Simple math MCP server
#
# Uses streamable HTTP
#

from mcp.server.fastmcp import FastMCP
import sys, logging

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
    # Streamable HTTP server (plain HTTP on a port)
    # Default path is /mcp
    mcp.run(
        transport="streamable_http",
        #host="0.0.0.0",
        #port=8000,
    )

