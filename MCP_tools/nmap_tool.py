from pydantic import BaseModel, Field
from typing import Dict, Any
from langchain.tools import tool
import asyncio
from dotenv import load_dotenv
from typing import Literal, Union
import os
from .mcp_server import KaliToolsClient, setup_mcp_server

load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")

# -------------------------------------------------------------------------------#
#                              NMAP tool implementation                          #
# -------------------------------------------------------------------------------#


# pydantic schema
class nmapInput(BaseModel):
    target: str = Field(
        ..., description="IP address, hostname or CIDR (npr. 192.168.157.0/24)"
    )
    scan_type: str = Field("-sV", description="Nmap scan type (npr. -sS -sV)")
    ports: str = Field(
        "", description="Comma-separated ports or ranges (npr. '22,80,443')"
    )
    additional_args: str = Field("", description="Additional nmap args")


# tool implementation
def nmap_implement(
    target: str, scan_type: str = "-sV", ports: str = "", additional_args: str = ""
) -> Dict[str, Any]:
    payload = {
        "target": target,
        "scan_type": scan_type,
        "ports": ports,
        "additional_args": additional_args,
    }

    return asyncio.run(toolCall(payload=payload))


# async tool call
async def toolCall(payload):
    client = KaliToolsClient(server_url=KALI_API)
    mcp = setup_mcp_server(kali_client=client)
    return await mcp.call_tool(name="nmap_scan", arguments=payload)


# langchain exposed tool
@tool(
    args_schema=nmapInput,
    description="Performs network or host scan in given network.",
    response_format="content",
)
def nmap_scan(
    target: str, scan_type: str = "-sV", ports: str = "", additional_args: str = ""
) -> Dict[str, Any]:
    return nmap_implement(
        target=target, scan_type=scan_type, ports=ports, additional_args=additional_args
    )


# -------------------------------------------------------------------------------#
#                                  Output schemas                                #
# -------------------------------------------------------------------------------#


class networkDiscovery(BaseModel):
    discovered_hosts: list[str] = Field(description="List of all active IPs.")
    initial_ports: dict[str, list[str]] = Field(
        description="Dictionary of active IPs and their matching ports."
    )


class individualHost(BaseModel):
    host: str = Field(description="Host IP address.")
    scanned_ports: list[str] = Field(
        description="List of all scanned ports."
    )  # keep an eye on this one // maybe its better to save port ranges instead of a whole port list
    open_ports: list[str] = Field(description="List of discovered open ports.")
    services: dict[str, str] = Field(description="Discovered active services")


class finalReport(BaseModel):
    summary: str = Field(description="Concise report of discovered hosts and services")
    selected_hosts: list[str] = Field(
        description="List of proposed hosts for further scanning."
    )
    next_steps: str = Field(
        description="Recommended next steps based on the acquired knowledge."
    )


# parent schema that will be passed to an agent
class stepSchema(BaseModel):
    step_type: Literal["network_discovery", "indivvidual_host_scan", "final_report"] = (
        Field(description="List of all available step types.")
    )
    reason: str = Field("Explanation why this step was taken.")
    payload: Union[networkDiscovery, individualHost, finalReport]
    next_action: str = Field("Next determined action based on a current knowledge.")


# -------------------------------------------------------------------------------#
#                                    Debugging                                   #
# -------------------------------------------------------------------------------#


if __name__ == "__main__":
    # debug: run directly
    test = {
        "target": "127.0.0.1",
        "scan_type": "-sS -sV",
        "ports": "22,80,443",
        "additional_args": "",
    }

    print("Calling tool implementation directly")
    out = nmap_implement(**test)
    print(out)
