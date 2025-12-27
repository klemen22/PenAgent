from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any
from langchain.tools import tool
from dotenv import load_dotenv
import os
from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server

load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")
client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)
savedPayload = {}

# -------------------------------------------------------------------------------#
#                              NMAP tool implementation                          #
# -------------------------------------------------------------------------------#


# pydantic schema
class nmapInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: str = Field(
        ..., description="IP address, hostname or CIDR (ex. 192.168.157.0/24)"
    )
    scan_type: str = Field("-sV", description="Nmap scan type (ex. -sS -sV)")
    ports: str = Field(
        "", description="Comma-separated ports or ranges (ex. '22,80,443')"
    )
    additional_args: str = Field("", description="Additional nmap args")


@tool(
    args_schema=nmapInput,
    description="Performs network or host scan in given network.",
    response_format="content",
)
async def nmap_scan(
    target: str,
    scan_type: str = "-sV",
    ports: str = "",
    additional_args: str = "",
) -> Dict[str, Any]:
    payload = {
        "target": target,
        "scan_type": scan_type,
        "ports": ports,
        "additional_args": additional_args,
    }
    await returnToolCall(mode="write", payload=payload)
    result = await mcp.call_tool(name="nmap_scan", arguments=payload)
    return result


async def returnToolCall(mode: str, payload=None):  # very useful stuff lmao
    global savedPayload
    if mode == "write":
        savedPayload = payload
    elif mode == "read":
        return savedPayload
