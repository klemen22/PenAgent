from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
from langchain.tools import tool
from dotenv import load_dotenv
import os
import asyncio
import json

# from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
from mcp_server import KaliToolsClient, setup_mcp_server

load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")
client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)
savedPayload = {}

testSqlmapAddr = os.getenv(key="TEST_TARGET", default="http://192.168.157.133")

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
            "url": testSqlmapAddr,
            "data": "",
            "additional_args": "--batch --forms --crawl=2",
        }
    )

    print("> Raw tool output from MCP:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(sqlmapTest())
