# TODO: implement dynamic context!

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from langchain.tools import tool, ToolRuntime
import asyncio
from dotenv import load_dotenv
import os
from .mcp_server import KaliToolsClient, setup_mcp_server
import re

from langgraph.types import Command
import json

load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")


# -------------------------------------------------------------------------------#
#                              NMAP tool implementation                          #
# -------------------------------------------------------------------------------#


# pydantic schema
class nmapInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: ToolRuntime
    target: str = Field(
        ..., description="IP address, hostname or CIDR (npr. 192.168.157.0/24)"
    )
    scan_type: str = Field("-sV", description="Nmap scan type (npr. -sS -sV)")
    ports: str = Field(
        "", description="Comma-separated ports or ranges (npr. '22,80,443')"
    )
    additional_args: str = Field("", description="Additional nmap args")


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
    runtime: ToolRuntime,
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

    result = asyncio.run(toolCall(payload=payload))
    updateAgentState(runtime=runtime, payload=payload, result=formatRawOutput(result))

    return result


# -------------------------------------------------------------------------------#
#                                 Custom agent state                             #
# -------------------------------------------------------------------------------#


class customAgentState(BaseModel):
    discovered_hosts: List[str] = Field(
        description="A list of discovered hosts that are in queue to be scanned.",
        default_factory=list,
    )
    scanned_hosts: List[str] = Field(
        description="A list of hosts already scanned in detail.", default_factory=list
    )
    pending_hosts: Dict[str, Any] = Field(
        description="Dictionary with listed hosts for each iterration.",
        default_factory=dict,
    )
    memory: Dict[str, Any] = Field(
        description="Dictionary with individual hosts and their coresponding facts.",
        default_factory=dict,
    )


# -------------------------------------------------------------------------------#
#                                   Memory update                                #
# -------------------------------------------------------------------------------#

SCAN_STEPS = [
    "basic_port_sweep",
    "service_version_detection",
    "script_analysis",
    "aggressive_profiling",
    "focused_rescan",
]


def updateAgentState(
    runtime: ToolRuntime, payload: Dict[str, Any], result: Dict[str, Any]
):
    # initialize fields
    if "discovered_hosts" not in runtime.state:
        runtime.state["discovered_hosts"] = []
    if "scanned_hosts" not in runtime.state:
        runtime.state["scanned_hosts"] = []
    if "pending_hosts" not in runtime.state:
        runtime.state["pending_hosts"] = {}
    if "memory" not in runtime.state:
        runtime.state["memory"] = {}

    if result["success"]:

        # I. update discovered hosts
        outputBlock = result["stdout"].split("Nmap scan report for ")[1:]

        for block in outputBlock:
            addr = re.match(r"([0-9]{1,3}(?:\.[0-9]{1,3}){3})", block.strip())
            if addr:
                ip = addr.group(1)
                if ip not in runtime.state["discovered_hosts"]:
                    runtime.state["discovered_hosts"].append(ip)

        # II. update facts in memory
        targetIP = payload["target"]
        currentMemory = runtime.state["memory"]

        # create field in agent's memory if not present
        currentMemory.setdefault(targetIP, {"facts": []})
        currentMemory[targetIP]["facts"].append(
            {
                "notes": f"With the scan of {payload['target']} we've learnt following facts:\n\n{result['stdout']}",
            }
        )

        # III. update pending hosts and already scanned hosts
        if targetIP not in runtime.state["pending_hosts"]:
            runtime.state["pending_hosts"][targetIP] = {
                "next_step": 0,
                "next_step_name": SCAN_STEPS[0],
                "total_steps": len(SCAN_STEPS),
            }

        runtime.state["pending_hosts"][targetIP]["next_step"] += 1

        currentStep = runtime.state["pending_hosts"][targetIP]["next_step"]
        allSteps = runtime.state["pending_hosts"][targetIP]["total_steps"]

        if currentStep >= allSteps:
            if targetIP not in runtime.state["scanned_hosts"]:
                runtime.state["scanned_hosts"].append(targetIP)
                runtime.state["pending_hosts"].pop(targetIP)
        else:
            runtime.state["pending_hosts"][targetIP]["next_stape_name"] = SCAN_STEPS[
                currentStep
            ]

    # debug for checking agent state in each iterration
    print(json.dumps(stateDebug(runtime.state), indent=2))
    return


def formatRawOutput(rawInput: Any) -> Dict:
    # handle tuple, dict and unknown cases

    if isinstance(rawInput, tuple):
        for item in rawInput:
            if isinstance(item, dict):
                if "result" in item and isinstance(item["result"], dict):
                    return item["result"]
                return item

    if isinstance(rawInput, dict):
        return rawInput

    return {"success": False, "error": str(rawInput)}


def stateDebug(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "discovered_hosts": state.get("discovered_hosts", []),
        "scanned_hosts": state.get("scanned_hosts", []),
        "pending_hosts": state.get("pending_hosts", {}),
        "memory": state.get("memory", {}),
    }
