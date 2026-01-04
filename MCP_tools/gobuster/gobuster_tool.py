from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
from langchain.tools import tool
from dotenv import load_dotenv
import os
import asyncio
from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server

load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")
client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)

savedPayload = {}

testGobusterAddr = os.getenv(key="TEST_TARGET", default="http://192.168.157.133")

# ------------------------------------------------------------------------------- #
#                         Gobuster tool implementation                            #
# ------------------------------------------------------------------------------- #


# pydantic stuff
class gobusterInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: str = Field(..., description="URL address of the target.")
    mode: str = Field(default="dir", description="Gobuster scan mode.")
    additional_args: str = Field(
        default="", description="Additional gobuster arguments."
    )


@tool(
    args_schema=gobusterInput,
    description="Perform gobuster scan.",
    response_format="content",
)
async def gobuster_scan(
    url: str, mode: str, additional_args: str = ""
) -> Dict[str, Any]:

    payload = {
        "url": url,
        "mode": mode,
        "wordlist": "/usr/share/wordlists/dirb/common.txt",
        "additional_args": additional_args,
    }
    await returnGobusterToolCall(mode="write", payload=payload)

    result = await mcp.call_tool(name="gobuster_scan", arguments=payload)
    return result


async def returnGobusterToolCall(mode: str, payload=None):
    global savedPayload

    if mode == "write":
        savedPayload = payload
    elif mode == "read":
        return savedPayload


# ------------------------------------------------------------------------------- #
#                                       Tool test                                 #
# ------------------------------------------------------------------------------- #


async def gobusterTest():
    print("\n" + "-" * 20)
    print("Gobuster tool test\n")

    result = await gobuster_scan.ainvoke(
        {"url": testGobusterAddr, "mode": "dir", "additional_args": ""}
    )

    print("> Raw tool output from MCP:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(gobusterTest())
