from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
from langchain.tools import tool
from dotenv import load_dotenv
import os
import asyncio
from sqlmapOutputParser import sqlmapOutputParser

try:
    from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
except Exception:
    from mcp_server import KaliToolsClient, setup_mcp_server

load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.137:5000")
client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)
savedPayload = {}

testEndpoint = os.getenv(key="TEST_ENDPOINT", default="http://192.168.157.136/")
testEndpointData = os.getenv(key="TEST_ENDPOINT_DATA", default="")


# -------------------------------------------------------------------------------#
#                             SQLmap tool implementation                         #
# -------------------------------------------------------------------------------#


# pydantic schema
class sqlmapInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = Field(..., description="URL address of target.")
    data: str = Field("", description="Data string to be sent through POST.")
    additional_args: str = Field("", description="Any additional SQLmap arguments.")


@tool(
    args_schema=sqlmapInput,
    description="Perform SQL injection testing.",
    response_format="content",
)
async def sqlmap_scan(url: str, data: str, additional_args: str = "") -> Dict[str, Any]:
    payload = {
        "url": url,
        "data": data,
        "additional_args": additional_args,
    }

    await returnSqlmapToolCall(mode="write", payload=payload)
    result = await mcp.call_tool(name="sqlmap_scan", arguments=payload)
    return result


async def returnSqlmapToolCall(mode: str, payload=None):
    global savedPayload

    if mode == "write":
        savedPayload = payload
    elif mode == "read":
        return savedPayload


# -------------------------------------------------------------------------------#
#                                       Tool test                                #
# -------------------------------------------------------------------------------#


# function for test tool call
async def sqlmapTest():
    print("\n" + "-" * 20)
    print("SQLmap tool test\n")

    result = await sqlmap_scan.ainvoke(
        {
            "url": testEndpoint,
            "data": testEndpointData,
            "additional_args": "--batch --level=1 --risk=1",
        }
    )

    print("> Raw tool output from MCP:\n")
    print(result)

    print("\n> Parsed tool output from MCP:\n")
    print(sqlmapOutputParser(result))


if __name__ == "__main__":
    asyncio.run(sqlmapTest())
