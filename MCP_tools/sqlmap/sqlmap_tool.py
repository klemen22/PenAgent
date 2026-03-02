from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
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

VALID_TAMPER = {
    "between",
    "space2comment",
    "charunicodeencode",
    "randomcase",
}


# -------------------------------------------------------------------------------#
#                             SQLmap tool implementation                         #
# -------------------------------------------------------------------------------#


class sqlmapConfig(BaseModel):
    level: int = Field(default=1, ge=1, le=5)
    risk: int = Field(default=1, ge=1, le=3)
    batch: bool = True
    random_agent: bool = False
    current_db: bool = False
    enumerate_tables: bool = False
    tamper: Optional[List[str]] = Field(default=None)


# pydantic schema
class sqlmapInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = Field(..., description="URL address of target.")
    data: str = Field("", description="Data string to be sent through POST.")
    config: sqlmapConfig


# @tool was removed because of the tool wrapper conflicts
async def sqlmap_scan(
    url: str,
    data: str,
    config: sqlmapConfig,
) -> Dict[str, Any]:

    additional_args = buildAdditionalArgs(config=config)

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
#                                      Helper function                           #
# -------------------------------------------------------------------------------#


def buildAdditionalArgs(config: sqlmapConfig) -> str:
    args = []

    args.append(f"--level={config.level}")
    args.append(f"--risk={config.risk}")

    if config.batch:
        args.append("--batch")

    if config.random_agent:
        args.append("--random-agent")

    if config.current_db:
        args.append("--current-db")

    if config.enumerate_tables:
        args.append("--tables")

    if config.tamper:
        valid = [t for t in config.tamper if t in VALID_TAMPER]

        if valid:
            args.append(f"--tamper={','.join(valid)}")

    return " ".join(args)


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
            "config": {"level": 1, "risk": 1, "batch": True},
        }
    )

    print("> Raw tool output from MCP:\n")
    print(result)

    print("\n> Parsed tool output from MCP:\n")
    print(sqlmapOutputParser(result))


if __name__ == "__main__":
    asyncio.run(sqlmapTest())
