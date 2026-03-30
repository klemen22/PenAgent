from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any, Literal
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import logging
from pathlib import Path
import re
import ipaddress
import shlex
import json

load_dotenv()

from MCP_tools.nmap.nmap_toolV2 import nmap_scan, nmapInput
from Orchestrator.memory.agent_output import AgentOutput, HostMemory, portInfo
from metadata.metadata_logger import setupMetadataLogger, logMetadata, logTotalTokens
from reasoning.reasoning_logger import setupReasoningLogger, logReasoning

# ------------------------------------------------------------------------------- #
#                                  LLM setup                                      #
# ------------------------------------------------------------------------------- #

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")
TOKEN_WINDOW_SIZE = os.getenv(key="TOKEN_WINDOW_SIZE", default=4096)
AGENT_NAME = "nmap_agent"
LLM_MODEL = os.getenv(key="LLM_MODEL", default="huihui_ai/qwen3-abliterated:8b")

llm = ChatOllama(
    name=AGENT_NAME,
    model=LLM_MODEL,
    base_url=LM_API,
    temperature=0.1,
    num_ctx=TOKEN_WINDOW_SIZE,
    format=None,
    system="You are a specialized cybersecurity assistant. You MUST always respond in English. Do not use any other languages under any circumstances.",
)

finalAgent = llm.bind_tools([nmap_scan])

logDir = Path("MCP_tools/nmap/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1


with open("MCP_tools/nmap/nmap_allowed_arguments.json") as f:
    ALLOWED_ARGS = json.load(f)


setupMetadataLogger(AGENT_NAME)
setupReasoningLogger(AGENT_NAME)

logReasoning(agentName=AGENT_NAME, reasoning=f" {20 * "="} NMAP REASONING {20 * "="} ")

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class nmapPlanStep(BaseModel):
    description: str = Field(default="")
    target: str = Field(default="")
    # scan_type: str = Field(default="")

    model_config = ConfigDict(extra="forbid")


class nmapOutputPlan(BaseModel):
    reasoning: str = Field(default="")
    steps: List[nmapPlanStep] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class agentFeedback(BaseModel):
    reasoning: str = Field(
        ..., description="Agent's analysis of the results and current state."
    )
    decision: Optional[Literal["continue", "replan", "finish_host"]] = Field(
        default=None
    )
    confidence: float = Field(default=0.0)


class nmapToolCall(BaseModel):
    reasoning: str
    target: str
    scan_type: str
    ports: Optional[List[int]] = Field(default_factory=list)
    additional_args: Optional[str] = Field(default="")


"""
class portInfo(BaseModel):
    port: Optional[int] = Field(default=None)
    service: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    state: str = Field(default="")
"""


class hostDiscovery(BaseModel):
    currentToolCall: Optional[nmapToolCall] = Field(default=None)
    last_tool_output: Optional[Any] = Field(default=None)

    replan_reason: Optional[str] = Field(default=None)
    replan_count: int = Field(default=0)
    max_replans: int = 5

    replan_flag: bool = Field(default=True)
    done: bool = Field(default=False)


class hostMemory(BaseModel):
    ip: str = Field(default="")
    status: str = Field(default="unknown")

    open_ports: List[portInfo] = Field(default_factory=list)
    services_found: List[str] = Field(default_factory=list)
    os_guess: Optional[str] = Field(default=None)
    scans_performed: List[Dict[str, Any]] = Field(
        default_factory=list, description="History of tool calls for a specific host."
    )
    currentToolCall: Optional[nmapToolCall] = Field(default=None)
    last_tool_output: Optional[Dict[str, Any]] = Field(default=None)

    plan: Optional[List[nmapPlanStep]] = Field(default=None)
    step_index: int = Field(default=0)

    feedback: Optional[agentFeedback] = Field(default=None)

    replan_reason: Optional[str] = Field(default=None)
    replan_count: int = Field(default=0)
    max_replans: int = 5

    replan_flag: bool = Field(default=False)
    done: bool = Field(default=False)


class nmapAgentState(BaseModel):
    objective: str = Field(
        default="", description="Main objective given by the orchestartor."
    )
    target: List[str] = Field(default_factory=list)

    host_discovery: hostDiscovery = Field(default_factory=hostDiscovery)

    discovered_hosts: List[str] = Field(
        default_factory=list, description="List of discovered hosts."
    )

    host_index: int = Field(default=0)
    host_memory: Dict[str, hostMemory] = Field(default_factory=dict)

    decision: Optional[str] = Field(default=None)

    iteration: int = Field(default=0)
    max_iteration: int = Field(default=50)

    summary: Optional[str] = Field(default=None)

    agent_output: AgentOutput = Field(default_factory=AgentOutput)

    fail_reason: Optional[str] = Field(default="")
    fail: bool = Field(default=False)

    done: bool = Field(default=False)


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #


async def initNode(state: nmapAgentState):
    logData(message="[INIT NODE] -> enter node")

    if state.target:
        targets = list(set(state.target))
        logData(f"[INIT NODE] -> targets provided by orchestrator: {targets}")
    else:
        targets = list(set(extractTargets(state.objective)))
        logData(f"[INIT NODE] -> targets extracted from objective: {targets}")

    if not targets:
        logData("[INIT NODE] -> no targets detected -> host discovery required")
        return {
            "decision": "plan",
        }

    networkTargets = [t for t in targets if isNetworkTarget(t)]
    ipTargets = [t for t in targets if not isNetworkTarget(t)]

    if networkTargets:
        logData(
            f"[INIT NODE] -> network detected {networkTargets} -> discovery required"
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[INIT] Network detected {networkTargets} -> discovery required.",
        )

        return {
            "host_discovery": hostDiscovery(
                done=False,
                currentToolCall=None,
            ),
            "decision": "plan",
        }

    for ip in ipTargets:

        if ip not in state.host_memory:

            state.host_memory[ip] = hostMemory(
                ip=ip,
                status="alive",
            )

            state.discovered_hosts.append(ip)

    state.host_discovery.done = True

    logData(f"[INIT NODE] -> skipping host discovery, using direct hosts: {ipTargets}")
    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f"[INIT] Skipping host discovery, using direct hosts: {ipTargets}.",
    )

    return {
        "host_memory": state.host_memory,
        "discovered_hosts": state.discovered_hosts,
        "host_discovery": state.host_discovery,
        "decision": "plan",
    }


async def planningNode(state: nmapAgentState):
    logData(message="[PLANNING NODE] -> enter node")

    if not state.host_discovery.done:
        logData(message="[PLANNING NODE] -> exit node: moving to host discovery.")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning="[PLANNING] Moving to host discovery.",
        )
        return {"decision": "continue"}

    # all hosts were scanned
    if state.host_discovery.done and state.host_index >= len(state.discovered_hosts):
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLANNING] All hosts were scanned moving to summary.",
        )
        return {"decision": "stop", "done": True}

    currentMemory = getCurrentHost(state=state)

    if currentMemory.plan and state.decision != "replan":
        logData(message="[PLANNING NODE] -> exit node: returning to tool call.")
        return {"decision": "continue"}

    if currentMemory.replan_flag:
        additionalPrompt = f"""
        [WARNING]
        
        You must replan your actions for host {currentMemory.ip}
        
        Replan reason: {currentMemory.replan_reason if currentMemory.replan_reason else ""}
        Last tool output:
        {currentMemory.last_tool_output if currentMemory.last_tool_output else ""}
        
        Feedback:
        {currentMemory.feedback if currentMemory.feedback else ""}
        
        [ORIGINAL TASK]\n\n
        """

    prompt = f"""
    You are an autonomous nmap agent.
    
    Objective:
    {state.objective}
    
    Initial proposed target/s from orchestrator:
    {state.target}
    
    Target host - known facts:
    IP: {currentMemory.ip}
    Status: {currentMemory.status}
    Open ports: {currentMemory.open_ports if currentMemory.open_ports else ""}
    Services: {currentMemory.services_found if currentMemory.services_found else ""}
    
    SCANS ALREADY PERFORMED:
    {currentMemory.scans_performed if currentMemory.scans_performed else "None"}
    
    Create a scan plan.

    IMPORTANT:
    You are working on ONLY this IP address.
    Do NOT consider any other addresses.
    DO NOT perform any vulnerability scans
    
    Return a valid JSON:
    - reasoning
    - steps list
    
    Inside setps list for each step return:
    - description
    - target
    """

    retries = 3

    finalPrompt = additionalPrompt + prompt if currentMemory.replan_flag else prompt

    if currentMemory.replan_flag:
        currentMemory.replan_flag = False
        currentMemory.replan_reason = ""

    for attempt in range(retries):

        try:
            outputPlanFull = await llm.with_structured_output(
                nmapOutputPlan, include_raw=True
            ).ainvoke(finalPrompt)

            outputPlan = outputPlanFull["parsed"]
            outPutPlanRaw = outputPlanFull["raw"]

            logMetadata(agent_name=AGENT_NAME, metadata=outPutPlanRaw.response_metadata)
            logData(message=f"[PLANNING NODE] -> created new plan: {outputPlan}")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning=f"[PLANNING] {outputPlan.reasoning}",
            )

            currentMemory.plan = outputPlan.steps
            currentMemory.step_index = 0
            logData(message=f"[PLANNING NODE -> exit node")
            return {
                "host_memory": state.host_memory,
                "decision": "continue",
            }

        except Exception as e:
            logData(f"Planning JSON parse failed attempt {attempt+1}")
            finalPrompt += "\n\nWARNING: Your previous output was not valid JSON. Return ONLY raw JSON. No explanations."

    logData(
        message=f"[WARNING] -> Planning failed after {retries} retries -> moving to output node"
    )
    currentMemory.plan = []
    currentMemory.step_index = 0

    state.fail = True
    state.fail_reason = f"""
    Planning failed after {retries} retries
    
    Final task before failure:
    {finalPrompt}
    
    Last tool output:
    {currentMemory.last_tool_output if currentMemory.last_tool_output else ""}
    """

    return {
        "fail_reason": state.fail_reason,
        "fail": state.fail,
        "host_memory": state.host_memory,
        "decision": "stop",
    }


async def selectToolCall(state: nmapAgentState):
    logData(message="[TOOL CALL NODE] -> enter node")
    # check host discovery
    hostDiscovery = state.host_discovery
    if not hostDiscovery.done:

        prompt = f"""
        You are an autonomous nmap agent.
        
        Your current objective:
        {state.objective}
            
        It was decided that host discovery is needed.
        
        Available nmap options for host discovery:
        {ALLOWED_ARGS}
        
        Decide:
            - which options are appropriate
            - explain reasoning
            
        You MUST follow these rules EXACTLY.

        IMPORTANT RULE:
        If performing host discovery:
        - You MUST only use scan_type
        - You MUST NOT include ports
        - You MUST NOT include service detection or OS detection
        
        IMPORTANT RULES FOR NMAP ARGUMENTS:
        1. 'additional_args' must be a STRING, not a list.
        2. Syntax for top ports: Use '--top-ports 100', NEVER just '100' or '--top-ports' alone.
        3. Order matters: Flags that require a value MUST be followed immediately by that value.
        4. Do not repeat the target IP in additional_args.

        BAD EXAMPLE: "additional_args": '100 --open --top-ports'
        GOOD EXAMPLE: "additional_args": '--top-ports 100 --open'
        
        Return ONLY JSON in the following format:
        {{
            "reasoning": <Your reasoning for this tool call>,
            "target": <IP address, hostname or CIDR>,
            "scan_type": <List of nmap scan arguments (ex. -sS -sV)>,
            "ports": <A JSON list of integers (e.g., [80, 443]), or an empty list [] if no specific ports are targeted>,
            "additional_args": <A string of space seperated additional nmap arguments (e.g., '-T4 --open'), or empty list '' if non are needed>,
        }}
            
        CORRECT EXAMPLES:
        (DO NOT take target address from examples! Use given addresses.)
        
        {{
            "reasoning": "Performing host discovery to find live hosts in the subnet.",
            "target": "10.10.10.0/24",
            "scan_type": "-sn",
            "ports": [],
            "additional_args": "", 
        }}

        You MUST return valid JSON only.
        """

        # Return valid json in the following form:
        # - target <IP address, hostname or CIDR>
        # - scan_type <List of nmap scan arguments (ex. -sS -sV)>
        # - ports (!FOR HOST DISCOVERY LEAVE EMPTY!)
        # - additional_args (!FOR HOST DISCOVERY LEAVE EMPTY!)

        finalPrompt = prompt

        if hostDiscovery.replan_flag:
            additionalPrompt = f"""
            [WARNING]
            
            You must replan your tool call.
            
            Replan reason: {hostDiscovery.replan_reason if hostDiscovery.replan_reason else ""}
            Last tool output:
            {hostDiscovery.last_tool_output if hostDiscovery.last_tool_output else ""}
            
            [ORIGINAL TASK]\n\n
            """
            finalPrompt = additionalPrompt + prompt
            hostDiscovery.replan_flag = False
            hostDiscovery.replan_reason = ""

        toolCallFull = await llm.with_structured_output(
            nmapToolCall, include_raw=True
        ).ainvoke(finalPrompt)

        toolCall = toolCallFull["parsed"]
        toolCallRaw = toolCallFull["raw"]

        toolCall = normalizeToolCall(toolCall=toolCall)
        toolCall = normalizeAdditionalArgs(toolCall=toolCall)

        logMetadata(agent_name=AGENT_NAME, metadata=toolCallRaw.response_metadata)
        logData(f"[SELECT TOOL CALL] -> raw produced tool call: {toolCall}")
        logData(
            message=f"[SELECT TOOL CALL] -> reasoning for current tool call: {toolCall.reasoning}"
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[SELECT TOOL CALL] {toolCall.reasoning}",
        )
        logData(message=f"[SELECT TOOL CALL] -> exit node")

        hostDiscovery.currentToolCall = toolCall

        return {
            "decision": "continue",
            "host_discovery": state.host_discovery,
        }

    # create tool call for current host
    currentHostMemory = getCurrentHost(state=state)
    performed_summary = ""
    for scan in currentHostMemory.scans_performed:
        performed_summary += f"- Command: {scan} (COMPLETED)\n"

    if currentHostMemory.step_index >= len(currentHostMemory.plan):
        logData(message=f"[SELECT TOOL CALL] -> exit node: replan needed!")
        currentHostMemory.replan_count += 1
        currentHostMemory.replan_flag = True
        currentHostMemory.replan_reason = "Maximum number of plan steps reached!"
        return {
            "decision": "replan",
            "host_memory": state.host_memory,
        }

    planStep = currentHostMemory.plan[currentHostMemory.step_index]

    prompt = f"""
    Current host - known facts:
    IP: {currentHostMemory.ip}
    Status: {currentHostMemory.status}
    Open ports: {currentHostMemory.open_ports if currentHostMemory.open_ports else "None."}
    Services discovered: {currentHostMemory.services_found if currentHostMemory.services_found else "None."}
    OS guess: {currentHostMemory.os_guess if currentHostMemory.os_guess else "None."}
    
    Current plan step:
    {planStep}

    [PROGRESS TRACKER]
    You have already attempted these specific actions:
    {performed_summary}

    CRITICAL: Do NOT repeat any command that is already marked as COMPLETED. 
    If a scan returned no ports, do NOT scan the same ports again with the same arguments.
    
    Last tool output:
    {currentHostMemory.last_tool_output}
    
    [AVAILABLE NMAP ARGUMENTS]
    {ALLOWED_ARGS}
    
    Last feedback:
    {currentHostMemory.feedback if currentHostMemory.feedback else ""}
    
    Decide:
    - which tool call and options are appropriate
    - prefer using DIFFERENT TOOL CALLS each iteration
    - try NOT to perform the same tool call as listed in "Already performed tool calls" section
    - explain with reasoning
    

    You MUST follow these rules EXACTLY.

    [COMMAND CONSTRUCTION RULES]
    1. SCAN_TYPE: Use ONLY flags like -sS, -sV, -sU, -sC, -sn. (NEVER put -p here!)
    2. PORTS: This MUST be a JSON list of integers. Example: [80, 443].
    3. ADDITIONAL_ARGS: Use ONLY flags from the [AVAILABLE NMAP ARGUMENTS] list.
    
    CRITICAL RULES:
    - NEVER use '-p-' unless specifically requested in the objective. 
    - Always prefer scanning top ports (e.g., using '--top-ports 100') or specific common ports first.
    - If you suspect a host is up but 'standard' ports are closed, use '--top-ports 1000' instead of '-p-'.
    
    BAD EXAMPLE: "additional_args": '100 --open --top-ports'
    GOOD EXAMPLE: "additional_args": '--top-ports 100 --open'

    Return ONLY JSON in the following format:
    {{
        "reasoning": <Your reasoning for this tool call>,
        "target": <IP address, hostname or CIDR>,
        "scan_type": <List of nmap scan arguments (ex. -sS -sV)>,
        "ports": <A JSON list of integers (e.g., [80, 443]), or an empty list [] if no specific ports are targeted>,
            "additional_args": <A string of space seperated additional nmap arguments (e.g., '-T4 --open'), or empty list '' if non are needed>,
    }}
        
    CORRECT EXAMPLES:
    (DO NOT take target address from examples! Use given addresses.)
    
    {{
        "reasoning": "Performing host discovery to find live hosts in the subnet.",
        "target": "10.10.10.0/24",
        "scan_type": "-sn",
        "ports": [],
        "additional_args": "",
        
    }}
    
    Example of a correct service scan:
    {{
        "reasoning": "Scanning common services",
        "target": "10.10.10.2",
        "scan_type": "-sV",
        "ports": [80, 443, 8080],
        "additional_args": "-T4 --open",
        
    }}

    You MUST return valid JSON only.
    """

    retries = 3

    for retry in range(retries):

        toolCallFull = await llm.with_structured_output(
            nmapToolCall, include_raw=True
        ).ainvoke(prompt)

        toolCall = toolCallFull["parsed"]
        toolCallRaw = toolCallFull["raw"]

        toolCall = normalizeToolCall(toolCall=toolCall)
        toolCall = normalizeAdditionalArgs(toolCall=toolCall)

        # currentHostMemory.currentToolCall = toolCall

        logMetadata(agent_name=AGENT_NAME, metadata=toolCallRaw.response_metadata)
        logData(f"[SELECT TOOL CALL] -> raw produced tool call: {toolCall}")
        logData(
            message=f"[SELECT TOOL CALL] -> reasoning for current tool call: {toolCall.reasoning}"
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[SELECT TOOL CALL] {toolCall.reasoning}",
        )

        if validateToolCall(toolCall=toolCall):
            logData(f"[SELECT TOOL CALL] -> tool call was validated!")
            currentHostMemory.currentToolCall = toolCall
            break

    if retry >= 2:
        logData(
            message=f"[SELECT TOOL CALL] -> agent couldn't create a valid tool call in {retry} tries"
        )

        logData(message=f"[SELECT TOOL CALL] -> replan is needed to fix agent failure!")

        currentHostMemory.replan_count += 1
        currentHostMemory.replan_flag = True
        currentHostMemory.replan_reason = f"Agent didn't create a valid tool call in {retry} tries. Last tool call created:\n\n{toolCall}\nTry to resolve the problem."

        return {
            "decision": "replan",
            "host_memory": state.host_memory,
        }

    logData(message=f"[SELECT TOOL CALL] -> exit node")

    return {
        "decision": "continue",
        "host_memory": state.host_memory,
    }


async def toolExecuteNode(state: nmapAgentState):
    logData(message="[EXECUTE TOOL] -> enter node")

    hostDiscovery = state.host_discovery

    if not hostDiscovery.done and hostDiscovery:

        if not hostDiscovery.currentToolCall:
            logData(
                message="[EXECUTE TOOL] -> error: hostDiscovery.currentToolCall is None!"
            )
            return {
                "decision": "plan",
            }

        tool = hostDiscovery.currentToolCall

        try:

            ports = (
                ",".join(map(str, tool.ports))
                if isinstance(tool.ports, list)
                else (tool.ports or "")
            )
            rawOutput = await nmap_scan(
                nmapInput(
                    target=tool.target,
                    scan_type=tool.scan_type,
                    ports=ports,
                    additional_args=tool.additional_args,
                )
            )
        except Exception as e:
            rawOutput = {"stdout": "", "stderr": str(e), "success": False}

        if isinstance(rawOutput, tuple):
            rawOutput = rawOutput[1]["result"]

        hostDiscovery.last_tool_output = rawOutput
        logData(message="[EXECUTE TOOL] -> exit node - discovery done")
        return {
            "decision": "continue",
            "host_discovery": state.host_discovery,
        }

    currentMemory = getCurrentHost(state=state)

    if not currentMemory or not currentMemory.currentToolCall:
        logData("[EXECUTE TOOL] -> error currentToolCall is None for current host!")
        return {
            "decision": "plan",
        }

    currentToolCall = currentMemory.currentToolCall

    try:

        ports = (
            ",".join(map(str, currentToolCall.ports))
            if isinstance(currentToolCall.ports, list)
            else (currentToolCall.ports or "")
        )

        rawOutput = await nmap_scan(
            nmapInput(
                target=currentToolCall.target,
                scan_type=currentToolCall.scan_type,
                ports=ports,
                additional_args=currentToolCall.additional_args,
            )
        )
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }

        logData(message=f"[TOOL EXECUTE] -> exit node - exception occured: {str(e)}")
        return {
            "decision": "evaluate",
            "host_memory": state.host_memory,
        }

    if isinstance(rawOutput, tuple):
        rawOutput = rawOutput[1]["result"]

    currentMemory.last_tool_output = rawOutput
    currentMemory.scans_performed.append(
        {
            "target": currentToolCall.target,
            "scan_type": currentToolCall.scan_type,
            "ports": currentToolCall.ports,
            "additional_args": currentToolCall.additional_args,
        }
    )
    logData(
        message=f"[TOOL EXECUTE] -> exit node - tool call was successful: {rawOutput.get('success','')}"
    )

    return {
        "decision": "continue",
        "host_memory": state.host_memory,
    }


async def parseOutputNode(state: nmapAgentState):
    logData("[PARSE OUTPUT] -> enter node")

    hostDiscovery = state.host_discovery
    if not hostDiscovery.done:
        if hostDiscovery.last_tool_output.get("success"):

            output = hostDiscovery.last_tool_output
            stdout = output.get("stdout", "")
            discovered = []

            stdout = output.get("stdout", "")

            discovered = re.findall(
                r"Nmap scan report for (\d+\.\d+\.\d+\.\d+)", stdout
            )

            state.discovered_hosts = discovered
            for ip in discovered:
                if ip not in state.host_memory:
                    # state.host_memory[ip] = hostMemory(
                    #    ip=ip,
                    #    status="alive",
                    # )

                    state.host_memory.setdefault(ip, hostMemory(ip=ip, status="alive"))

            hostDiscovery.done = True

            logData(f"[PARSE OUTPUT] -> discovered hosts: {discovered}")
            logData(f"[PARSE OUTPUT] -> exit node")

            return {
                "host_discovery": state.host_discovery,
                "host_memory": state.host_memory,
                "discovered_hosts": state.discovered_hosts,
                "decision": "plan",
            }
        else:
            hostDiscovery.replan_count += 1
            hostDiscovery.replan_flag = True
            hostDiscovery.replan_reason = "Tool call encountered error."

            if hostDiscovery.replan_count >= hostDiscovery.max_replans:
                state.fail = True
                state.fail_reason = f"""
                [FAILED]
                
                Reason: {hostDiscovery.replan_reason}
                
                Last tool output:
                {hostDiscovery.last_tool_output}
                """

                return {
                    "host_discovery": state.host_discovery,
                    "host_memory": state.host_memory,
                    "fail": state.fail,
                    "fail_reason": state.fail_reason,
                    "decision": "stop",
                }

            return {
                "host_discovery": state.host_discovery,
                "host_memory": state.host_memory,
                "decision": "replan",
            }

    # normal host scan parsing
    currentHost = getCurrentHost(state)

    if not currentHost:
        logData("[PARSE OUTPUT] -> no current host (check discovery)")
        return {"decision": "replan"}

    output = currentHost.last_tool_output

    if not output:
        logData("[PARSE OUTPUT] -> empty tool output")
        currentHost.replan_count += 1
        currentHost.replan_flag = True
        currentHost.replan_reason = "Empty tool output!"
        return {
            "host_discovery": state.host_discovery,
            "decision": "replan",
            "host_memory": state.host_memory,
        }

    stdout = output.get("stdout", "")

    parsedPorts, os_guess = parsePorts(stdout)
    currentHost.os_guess = os_guess

    for p in parsedPorts:

        exists = any(x.port == p.port for x in currentHost.open_ports)

        if not exists and p.state == "open":

            currentHost.open_ports.append(p)

            if p.service and p.service not in currentHost.services_found:
                currentHost.services_found.append(p.service)

    logData(
        f"[PARSE OUTPUT] -> host {currentHost.ip} open ports: {len(currentHost.open_ports)}"
    )

    logData(f"[PARSE OUTPUT] -> exit node")

    return {
        "host_discovery": state.host_discovery,
        "host_memory": state.host_memory,
        "decision": "continue",
    }


async def evaluateNode(state: nmapAgentState):
    logData(message=f"[EVALUATE NODE] -> enter node")

    currentMemory = getCurrentHost(state=state)

    if not currentMemory.plan:
        return {"decision": "plan"}

    planStep = currentMemory.plan[currentMemory.step_index]
    lastScan = (
        currentMemory.scans_performed[-1] if currentMemory.scans_performed else None
    )

    prompt = f"""
    You are a professional security analyst evaluating nmap results.
    
    Current plan step: {planStep.description}
    Last command: {lastScan}
    Output: {currentMemory.last_tool_output}
    
    Decide:
        1. 'continue': If the scan was successful and you want to move to the NEXT step in the plan.
        2. 'replan': If the scan failed or returned unexpected results and you need a NEW approach for THIS host.
        3. 'finish_host': If you have gathered enough information or the host is unresponsive/empty.
      
    Return valid json in the following form:
    - reasoning
    - decision <one of [continue, replan, finish_host]>
    - confidence <a number between 0.0 and 1.0>
    """

    feedbackFull = await llm.with_structured_output(
        agentFeedback, include_raw=True
    ).ainvoke(prompt)

    feedback = feedbackFull["parsed"]
    feedbackRaw = feedbackFull["raw"]

    currentMemory.feedback = feedback

    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f"[EVALUATE] {feedback.reasoning}",
    )
    logMetadata(agent_name=AGENT_NAME, metadata=feedbackRaw.response_metadata)

    state.iteration += 1
    if state.iteration >= state.max_iteration:
        state.fail_reason = "Maximum number of allowed iterations reached."

        return {
            "decision": "stop",
            "fail": True,
            "host_memory": state.host_memory,
            "fail_reason": state.fail_reason,
        }

    decision = feedback.decision.lower()

    if decision == "finish_host":
        logData(f"[EVALUATE] -> agent decided to finish host {currentMemory.ip}")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[EVALUATE] Agent decided to finish host {currentMemory.ip}.",
        )
        state.host_index += 1
        return {
            "decision": "plan",
            "host_memory": state.host_memory,
            "host_index": state.host_index,
        }

    if decision == "replan":
        currentMemory.replan_count += 1
        if currentMemory.replan_count >= currentMemory.max_replans:
            logData("[EVALUATE] -> max replans reached, moving to next host.")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning="[EVALUATE] Max replans reached, moving to next host.",
            )
            state.host_index += 1
            return {
                "decision": "plan",
                "host_memory": state.host_memory,
                "host_index": state.host_index,
            }

        currentMemory.replan_flag = True
        currentMemory.replan_reason = feedback.reasoning
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[EVALUATE] {feedback.reasoning}",
        )
        return {"decision": "replan", "host_memory": state.host_memory}

    currentMemory.step_index += 1
    if currentMemory.step_index >= len(currentMemory.plan):
        state.host_index += 1
        return {
            "decision": "plan",
            "host_memory": state.host_memory,
            "host_index": state.host_index,
        }

    return {"decision": "continue", "host_memory": state.host_memory}


async def outputNode(state: nmapAgentState):
    logData(message="[OUTPUT NODE] -> enter node")

    if state.fail:
        logData(message="[OUTPUT NODE] -> general task failed")
        prompt = f"""
        You failed at performing your task.
        Create a concise summary why that happened based on the bollow listed facts.
        
        Fail reason:
        {state.fail_reason}
        
        Agent state before failure:
        {state.model_dump_json()}
        
        NO markdown, NO emojis.
        """
    else:
        prompt = f"""
        Create a concise summary based on the given initial objective and gathered infromation.
        
        Objective:
        {state.objective}
        
        Initial proposed target/s from orchestrator:
        {state.target}
        
        Host infromation:
        {state.host_memory}
        
        NO markdown, NO emojis.
        """

    logData(message="[OUTPUT NODE] -> generating summary")
    response = await llm.ainvoke(prompt)
    logMetadata(agent_name=AGENT_NAME, metadata=response.response_metadata)
    # end of metadata saving
    logTotalTokens(agent_name=AGENT_NAME)
    state.summary = response.content
    logData(message="[OUTPUT NODE] -> exit node - summary done")
    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f" {20 * "="} NMAP SUMMARY {20 * "="} ",
    )
    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f"[SUMMARY] {response.content}",
    )

    if state.fail:
        return {
            "agent_output": AgentOutput(
                agent_name="nmap",
                success=False,
                fail=True,
                fail_reason=state.fail_reason,
                summary=state.summary,
            )
        }
    else:
        hosts = {
            key: HostMemory(
                ip=value.ip,
                status=value.status,
                open_ports=value.open_ports,
                os_guess=value.os_guess,
            )
            for key, value in state.host_memory.items()
        }

        return {
            "agent_output": AgentOutput(
                agent_name="nmap",
                success=True,
                discovered_hosts=state.discovered_hosts,
                host_memory=hosts,
                summary=state.summary,
            )
        }


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #


def isNetworkTarget(target: str) -> bool:
    try:
        if "/" in target:
            return True

        ip = ipaddress.ip_address(target)

        if target.endswith(".0") or target.endswith(".255"):
            return True

        return False
    except:
        return False


def extractTargets(prompt: str):

    cidr = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b", prompt)
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", prompt)

    return cidr + ips


def normalizeToolCall(toolCall: nmapToolCall):
    if "-p" in toolCall.scan_type:
        extracted_ports = re.findall(r"-p\s*([\d,]+)", toolCall.scan_type)
        if extracted_ports:
            new_ports = []
            for part in extracted_ports[0].split(","):
                if part.strip().isdigit():
                    new_ports.append(int(part.strip()))

            existing = toolCall.ports if toolCall.ports else []
            toolCall.ports = list(set(existing + new_ports))

        toolCall.scan_type = re.sub(r"-p\s*[\d,]+", "", toolCall.scan_type).strip()
        toolCall.scan_type = toolCall.scan_type.replace("-p-", "").strip()

    return toolCall


def normalizeAdditionalArgs(toolCall: nmapToolCall) -> nmapToolCall:
    if not toolCall.additional_args:
        toolCall.additional_args = ""
        return toolCall

    try:
        parts = shlex.split(toolCall.additional_args)
        cleaned_parts = [p.strip() for p in parts if p.strip()]
        toolCall.additional_args = " ".join(cleaned_parts)
    except Exception as e:
        logData(f"[NORMALIZE] -> error parsing additional_args: {e}")
        toolCall.additional_args = " ".join(toolCall.additional_args.split())

    return toolCall


import re


def validateToolCall(toolCall: nmapToolCall):
    if not toolCall.target or not toolCall.scan_type:
        return False

    full_command_string = f"{toolCall.scan_type} {toolCall.additional_args}"
    forbidden_chars = {";", "&&", "||", "|", "`", "$", "(", ")", ">", "<"}
    if any(char in full_command_string for char in forbidden_chars):
        logData("[VALIDATION] -> rejected: forbidden characters detected!")
        return False

    p_flag_pattern = r"\b-p\b"
    if re.search(p_flag_pattern, toolCall.scan_type.lower()) or re.search(
        p_flag_pattern, toolCall.additional_args.lower()
    ):
        logData(
            "[VALIDATION] -> rejected: -p found in arguments (use ports field instead)"
        )
        return False

    is_discovery = "-sn" in toolCall.scan_type
    has_ports = len(toolCall.ports) > 0 if toolCall.ports else False
    if is_discovery:
        forbidden_discovery = ["-sS", "-sT", "-sV", "-O", "-sC", "--top-ports"]
        if has_ports:
            logData("[VALIDATION] -> rejected: host discovery (-sn) cannot have ports")
            return False
        for f_flag in forbidden_discovery:
            if re.search(rf"\b{re.escape(f_flag)}\b", full_command_string):
                logData(f"[VALIDATION] -> rejected: host discovery cannot use {f_flag}")
                return False

    allowed_flags = set()
    for cat in ALLOWED_ARGS.values():
        items = []
        if isinstance(cat, list):
            items = cat
        elif isinstance(cat, dict):
            items = (
                cat.get("safe", [])
                + cat.get("aggressive", [])
                + cat.get("high_risk", [])
                + cat.get("stealth_evasion", [])
            )

        for item in items:
            allowed_flags.add(item["name"].split()[0])

    try:
        current_args_list = shlex.split(
            f"{toolCall.scan_type} {toolCall.additional_args}"
        )
        for arg in current_args_list:
            if arg.startswith("-"):
                base_flag = arg.split("=")[0]

                is_dynamic_flag = any(
                    base_flag.startswith(f) for f in ["-PS", "-PA", "-PU", "-D"]
                )

                if base_flag not in allowed_flags and not is_dynamic_flag:
                    logData(f"[VALIDATION] -> rejected unknown flag: {base_flag}")
                    return False
    except Exception as e:
        logData(f"[VALIDATION] -> shlex error: {e}")
        return False

    if toolCall.ports:
        if not all(isinstance(p, int) for p in toolCall.ports):
            logData("[VALIDATION] -> rejected: ports must be a list of integers")
            return False

    return True


def setupLogger():
    logFile = logDir / f"nmap_agent_log{logCount}.log"
    logger = logging.getLogger("nmap_agent_log")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def logData(message: str):
    logging.getLogger("nmap_agent_log").info(message)


def getCurrentHost(state: nmapAgentState) -> hostMemory:
    if state.host_index >= len(state.discovered_hosts):
        return None

    host = state.discovered_hosts[state.host_index]

    return state.host_memory.get(host)


def parsePorts(stdout: str):
    ports = []
    os_guess = None

    for line in stdout.splitlines():
        # port / service detection
        match = re.match(r"(\d+)/tcp\s+(\w+)\s+([\w\-\.]+)?\s*(.*)", line)
        if match:
            port = int(match.group(1))
            state = match.group(2)
            service = match.group(3)
            version = match.group(4).strip()
            ports.append(
                portInfo(port=port, state=state, service=service, version=version)
            )

        # os detection
        os_match = re.match(r"(OS details|Running):\s*(.*)", line)
        if os_match:
            os_guess = os_match.group(2).strip()

    return ports, os_guess


def retrieveCurrentDecision(state: nmapAgentState):
    currentState = state.decision

    if currentState in ["plan", "replan"]:
        return "plan"
    else:
        return currentState


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #
def nmapBuilder():
    setupLogger()

    workflow = StateGraph(nmapAgentState)

    # -------------------------------
    # graph nodes
    # -------------------------------

    workflow.add_node("init_node", initNode)
    workflow.add_node("planning_node", planningNode)
    workflow.add_node("tool_call_node", selectToolCall)
    workflow.add_node("execute_tool_node", toolExecuteNode)
    workflow.add_node("parse_output_node", parseOutputNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------
    workflow.add_edge(START, "init_node")
    workflow.add_edge("init_node", "planning_node")
    workflow.add_conditional_edges(
        "planning_node",
        retrieveCurrentDecision,
        {
            "continue": "tool_call_node",
            "stop": "output_node",
        },
    )
    workflow.add_conditional_edges(
        "tool_call_node",
        retrieveCurrentDecision,
        {
            "continue": "execute_tool_node",
            "plan": "planning_node",
        },
    )
    workflow.add_conditional_edges(
        "execute_tool_node",
        retrieveCurrentDecision,
        {
            "continue": "parse_output_node",
            "evaluate": "evaluate_node",
        },
    )
    workflow.add_conditional_edges(
        "parse_output_node",
        retrieveCurrentDecision,
        {
            "continue": "evaluate_node",
            "plan": "planning_node",
            "stop": "output_node",
        },
    )
    workflow.add_conditional_edges(
        "evaluate_node",
        retrieveCurrentDecision,
        {
            "continue": "tool_call_node",
            "plan": "planning_node",
            "stop": "output_node",
        },
    )
    workflow.add_edge("output_node", END)
    graph = workflow.compile(checkpointer=False)

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "MCP_tools/nmap/nmap_agent_graph.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    return graph


async def agentRunner(prompt):

    graph = nmapBuilder()

    state = nmapAgentState()
    state.objective = prompt

    result = await graph.ainvoke(state.model_dump(), {"recursion_limit": 1000})

    print(f"[FINAL RESULT]:\n\n{result}")


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "Nmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    testPrompt = "Position yourself in the network 192.168.157.0/24 and discover all relevant hosts."
    result = asyncio.run(agentRunner(prompt=testPrompt))
