from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "sequentialthinking": {
            "transport": "stdio",  # Local subprocess communication
            "command": "npx",
            # Absolute path to your math_server.py file
            "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"],
        },
        "memory": {
            "transport": "stdio",
            "command": "docker",
            "args": [
                "run",
                "-i",
                "-v",
                "claude-memory:/app/dist",
                "--rm",
                "mcp/memory",
            ],
        },
    }
)

# tools = await client.get_tools()
