from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import logging
from pathlib import Path
import re

load_dotenv()

from MCP_tools.nmap.nmap_toolV2 import nmap_scan, nmapInput

# ------------------------------------------------------------------------------- #
#                                  LLM setup                                      #
# ------------------------------------------------------------------------------- #

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")

llm = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b",
    base_url=LM_API,
    temperature=0.2,
    format=None,
)

finalAgent = llm.bind_tools([nmap_scan])

logDir = Path("MCP_tools/nmap/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class nmapPlanStep(BaseModel):
    description: str = Field(default="")
    target: str = Field(default="")
    scan_type: str = Field(default="")

    model_config = ConfigDict(extra="forbid")


class nmapOutputPlan(BaseModel):
    reasoning: str = Field(default="")
    steps: List[nmapPlanStep] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class agentFeedback(BaseModel):
    confidence: float = Field(default=0.0)
    reasoning: str = Field(default="")


class nmapToolCall(BaseModel):
    target: str = Field(default="")
    scan_type: str = Field(
        default="", description="This field is meant for scan parameters."
    )
    ports: Optional[str] = Field(default="")
    additional_args: Optional[str] = Field(default="")

    reasoning: str = Field(..., description="Agent's reasoning for specific tool call.")


class portInfo(BaseModel):
    port: Optional[int] = Field(default=None)
    service: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    state: str = Field(default="")


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

    fail_reason: Optional[str] = Field(default="")
    fail: bool = Field(default=False)

    done: bool = Field(default=False)


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #
async def planningNode(state: nmapAgentState):
    logData(message="[PLANNING NODE] -> enter node")

    if not state.host_discovery.done:
        logData(message="[PLANNING NODE] -> exit node: moving to host discovery.")
        return {"decision": "continue"}

    # all hosts were scanned
    if state.host_discovery.done and state.host_index >= len(state.discovered_hosts):
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
    
    Target host - known facts:
    IP: {currentMemory.ip}
    Status: {currentMemory.status}
    
    Create a scan plan.

    Steps must include:
    1. port scan
    2. service detection
    
    IMPORTANT:
    You are working on ONLY this IP address.
    Do NOT consider any other addresses.
    
    For each step return JSON:
    - description
    - target
    - scan_type
    """

    retries = 2

    finalPrompt = additionalPrompt + prompt if currentMemory.replan_flag else prompt

    if currentMemory.replan_flag:
        currentMemory.replan_flag = False
        currentMemory.replan_reason = ""

    for attempt in range(retries):

        try:
            outputPlan = await llm.with_structured_output(nmapOutputPlan).ainvoke(
                finalPrompt
            )
            logData(message=f"[PLANNING NODE] -> created new plan: {outputPlan}")
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

    # TODO: add allowed arguments
    with open("MCP_tools/nmap/nmap_allowed_arguments.json") as f:
        allowedArguments = f.read()

    # check host discovery
    hostDiscovery = state.host_discovery
    if not hostDiscovery.done:

        prompt = f"""
        You are an autonomous nmap agent.
        
        Your current objective:
        {state.objective}
        
        First you must perform a host discovery phase.
        
        Available nmap options for host discovery:
        {allowedArguments}
        
        Decide:
            - which options are appropriate
            - explain reasoning
            
        Return ONLY JSON in the following format:
        {{
            "target": "<IP address, hostname or CIDR>",
            "scan_type": "<List of nmap scan arguments (ex. -sS -sV)>",
            "ports": "<Comma-separated ports or ranges, empty if host discovery>",
            "additional_args": "<Any additional nmap arguments if needed>",
            "reasoning": "<Your reasoning for this tool call>"
        }}
        
        Example output:
        {{
            "target": "192.168.157.0/24",
            "scan_type": "-sn",
            "ports": "",
            "additional_args": "",
            "reasoning": "Performing host discovery to find live hosts in the subnet."
        }}
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

        toolCall = await llm.with_structured_output(nmapToolCall).ainvoke(finalPrompt)
        logData(
            message=f"[SELECT TOOL CALL] -> reasoning for current tool call: {toolCall.reasoning}"
        )
        logData(message=f"[SELECT TOOL CALL] -> exit node")

        hostDiscovery.currentToolCall = toolCall

        return {
            "decision": "continue",
            "host_discovery": state.host_discovery,
        }

    # create tool call for current host
    currentHostMemory = getCurrentHost(state=state)

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
    Description: {planStep.description}
    Proposed scan type: {planStep.scan_type}
    
    Already performed tool calls:
    {currentHostMemory.scans_performed}
    
    Last tool output:
    {currentHostMemory.last_tool_output}
    
    Available nmap options:
    {allowedArguments}
    
    Last feedback:
    {currentHostMemory.feedback if currentHostMemory.feedback else ""}
    
    Decide:
    - which tool call and options are appropriate
    - explain with reasoning
    

    Return ONLY JSON in the following format:
    {{
        "target": "<IP address, hostname or CIDR>",
        "scan_type": "<List of nmap scan arguments (ex. -sS -sV)>",
        "ports": "<Comma-separated ports or ranges, empty if host discovery>",
        "additional_args": "<Any additional nmap arguments if needed>",
        "reasoning": "<Your reasoning for this tool call>"
    }}
        
    Example output:
    {{
        "target": "192.168.157.0/24",
        "scan_type": "-sn",
        "ports": "",
        "additional_args": "",
        "reasoning": "Performing host discovery to find live hosts in the subnet."
    }}
    """

    toolCall = await llm.with_structured_output(nmapToolCall).ainvoke(prompt)
    currentHostMemory.currentToolCall = toolCall

    logData(
        message=f"[SELECT TOOL CALL] -> reasoning for current tool call: {toolCall.reasoning}"
    )
    logData(message=f"[SELECT TOOL CALL] -> exit node")

    return {
        "decision": "continue",
        "host_memory": state.host_memory,
    }


async def toolExecuteNode(state: nmapAgentState):
    logData(message="[EXECUTE TOOL] -> enter node")

    hostDiscovery = state.host_discovery

    if not hostDiscovery.done:
        hostDiscoveryToolCall = hostDiscovery.currentToolCall
        try:
            rawOutput = await nmap_scan(
                nmapInput(
                    target=hostDiscoveryToolCall.target,
                    scan_type=hostDiscoveryToolCall.scan_type,
                    ports=hostDiscoveryToolCall.ports,
                    additional_args=hostDiscoveryToolCall.additional_args,
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
    currentToolCall = currentMemory.currentToolCall

    try:
        rawOutput = await nmap_scan(
            nmapInput(
                target=currentToolCall.target,
                scan_type=currentToolCall.scan_type,
                ports=currentToolCall.ports,
                additional_args=currentToolCall.additional_args,
            )
        )
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }

        logData(message=f"[TOOL EXECUTE] -> exit node - exception occured!")
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
        "host_memory": state.host_memory,
        "decision": "continue",
    }


async def evaluateNode(state: nmapAgentState):
    logData(message=f"[EVALUATE NODE] -> enter node")

    currentMemory = getCurrentHost(state=state)

    if not currentMemory.plan:
        return {"decision": "plan"}

    planStep = currentMemory.plan[currentMemory.step_index]

    prompt = f"""
    You are a professional evaluater agent.
    Your task is to analyze bellow listed results and give feedback.
    
    Current plan:
    {planStep}
    
    Last tool call:
    {currentMemory.scans_performed[-1]}
    
    Las tool output:
    {currentMemory.last_tool_output}
    
    Decide confidence value based on a tool call performed and coresponding result.
      
    Return valid json in the following form:
    - confidence <a number between 0.0 and 1.0>
    - reasoning
    """

    feedback = await llm.with_structured_output(agentFeedback).ainvoke(prompt)
    currentMemory.feedback = feedback

    state.iteration += 1
    if state.iteration >= state.max_iteration:
        state.fail_reason = "Maximum number of allowed iterations reached."

        return {
            "decision": "stop",
            "fail": True,
            "host_memory": state.host_memory,
            "fail_reason": state.fail_reason,
        }

    # too low confidence
    if currentMemory.feedback.confidence < 0.3:
        currentMemory.replan_count += 1

        # max number of replans reached -> moving to next host
        if currentMemory.replan_count >= currentMemory.max_replans:
            state.host_index += 1
            logData(
                message="[EVALUATE NODE] -> exit node (replan cap reached) -> continue with next node"
            )
            return {
                "decision": "plan",
                "host_memory": state.host_memory,
                "host_index": state.host_index,
            }
        else:
            # replan
            currentMemory.replan_reason = "Confidence too low."
            logData(
                message="[EVALUATE NODE] -> exit node (too low confidence, replan is needed)"
            )

            currentMemory.replan_flag = True
            currentMemory.replan_reason = "Confidence is too low!"
            return {
                "decision": "replan",
                "host_memory": state.host_memory,
            }

    currentMemory.step_index += 1

    # all planned steps executed
    if currentMemory.step_index >= len(currentMemory.plan):
        logData(message="[EVALUATE NODE] -> exit node - moving to next host")
        state.host_index += 1
        return {
            "decision": "plan",
            "host_memory": state.host_memory,
            "host_index": state.host_index,
        }

    logData(message="[EVALUATE NODE] -> exit node - moving to next step")
    return {
        "host_memory": state.host_memory,
        "decision": "continue",
    }


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

    prompt = f"""
    Create a concise summary based on the given initial objective and gathered infromation.
    
    Objective:
    {state.objective}
    
    Host infromation:
    {state.host_memory}
    
    NO markdown, NO emojis.
    """

    logData(message="[OUTPUT NODE] -> generating summary")
    state.summary = await llm.ainvoke(prompt)

    logData(message="[OUTPUT NODE] -> exit node - summary done")
    return {
        "summary": state.summary,
        "host_memory": state.host_memory,
    }


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #


def setupLogger():
    logFile = logDir / f"nmap_agent_log{logCount}.log"
    logger = logging.getLogger("nmap_agent")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def logData(message: str):
    logging.getLogger("nmap_agent").info(message)


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
async def agentRunner(prompt):
    agentState = nmapAgentState()
    setupLogger()

    workflow = StateGraph(nmapAgentState)
    SESSIN_ID = "default_session"

    # test prompt
    # agentState.objective = "Position yourself in the network 192.168.157.0 and discover all relevant hosts."
    agentState.objective = prompt

    # -------------------------------
    # graph nodes
    # -------------------------------

    workflow.add_node("planning_node", planningNode)
    workflow.add_node("tool_call_node", selectToolCall)
    workflow.add_node("execute_tool_node", toolExecuteNode)
    workflow.add_node("parse_output_node", parseOutputNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------
    workflow.add_edge(START, "planning_node")
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

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "MCP_tools/nmap/nmap_agent_graph.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    result = await graph.ainvoke(
        agentState.model_dump(),
        config={"thread_id": SESSIN_ID, "recursion_limit": 1000},
    )

    # print(
    #    f"[FINAL RESULT]:\n\nSummary:\n{result.get("summary")}\n\nMemory:{result.get("host_memory")}"
    # )

    return {
        "summary": result.get("summary").content if result.get("summary") else "",
        "hosts": {
            ip: host.model_dump() for ip, host in result.get("host_memory", {}).items()
        },
    }


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "Nmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    testPrompt = "Position yourself in the network 192.168.157.0 and discover all relevant hosts."
    result = asyncio.run(agentRunner(prompt=testPrompt))

    print(f"[FINAL RESULT]:\n\n{result}")
