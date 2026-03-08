from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os
import asyncio

try:
    from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
except Exception:
    from mcp_server import KaliToolsClient, setup_mcp_server


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
    ports: Optional[str] = Field(
        default=None, description="Comma-separated ports or ranges (ex. '22,80,443')"
    )
    additional_args: Optional[str] = Field(
        default=None, description="Additional nmap args"
    )


async def nmap_scan(input: nmapInput) -> Dict[str, Any]:
    payload = {
        "target": input.target,
        "scan_type": input.scan_type,
        "ports": input.ports,
        "additional_args": input.additional_args,
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


# -------------------------------------------------------------------------------#
#                                       Tool test                                #
# -------------------------------------------------------------------------------#


async def nmapTest():
    print("\n" + "-" * 20)
    print("Nmap tool test\n")

    result = await nmap_scan(
        nmapInput(
            target="192.168.157.0/24",
            scan_type="-sn",
            ports="",
            additional_args="",
        )
    )

    print("> Raw tool output from MCP:\n")
    print(result)


if __name__ == "__main__":
    asyncio.run(nmapTest())
