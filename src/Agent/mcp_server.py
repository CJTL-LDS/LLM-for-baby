from mcp.server.fastmcp import FastMCP
from bilibili_api import search as search_m, sync


mcp = FastMCP("Bilibili MCP Server")

@mcp.tool()
def search(key: str) -> dict:
    """
    Search Bilibili API with the given key

    Args:
        key: Search term to look for on Bilibili

    Returns:
        Dictionary contained the search results from Bilibili
    """
    return sync(search_m.search(key))


if __name__ == "__main__":
    mcp.run(transport="stdio")