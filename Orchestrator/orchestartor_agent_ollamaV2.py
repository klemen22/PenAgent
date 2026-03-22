from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Literal
from langchain_core.runnables import RunnableConfig
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import json
import logging
from pathlib import Path

# tool agents for subgraphs
from MCP_tools.nmap.nmap_agent_ollamaV2 import nmapBuilder
from MCP_tools.gobuster.gobuster_agent_ollamaV2 import gobusterBuilder
from MCP_tools.sqlmap.sqlmap_agent_ollamaV3 import sqlmapBuilder
from metadata.metadata_logger import setupMetadataLogger, logMetadata
from memory.agent_output import portInfo, HostMemory, attackVector, vulnerability

# output schema
from Orchestrator.memory.agent_output import AgentOutput

"""
# database
from memory.sqlite_manager import (
    initializeDB,
    storeAttackVector,
    storeHosts,
    storePorts,
    storeVulnerability,
    getAttackVector,
    getHosts,
    getPorts,
    getVulnerability,
)
initializeDB()
"""

load_dotenv()

# get compiled graphs
NMAP_GRAPH = nmapBuilder()
GOBUSTER_GRAPH = gobusterBuilder()
SQLMAP_GRAPH = sqlmapBuilder()

AGENT_REGISTRY = {
    "nmap": {
        "capabilities": ["host_discovery", "port_scan"],
    },
    "gobuster": {
        "capabilities": ["directory_enum", "web_enum"],
    },
    "sqlmap": {
        "capabilities": ["sql_injection"],
    },
}

AGENT_REQUIREMENTS = {"nmap": [], "crawler": [], "sqlmap": ["attack_vectors"]}
AGENT_NAME = "orchestrator_agent"

setupMetadataLogger(agent_name=AGENT_NAME)

# ------------------------------------------------------------------------------- #
#                                  LLM setup                                      #
# ------------------------------------------------------------------------------- #

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")
TOKEN_WINDOW_SIZE = os.getenv(key="TOKEN_WINDOW_SIZE", default=4096)
LLM_MODEL = os.getenv(key="LLM_MODEL", default="huihui_ai/qwen3-abliterated:8b")

llm = ChatOllama(
    name=AGENT_NAME,
    model=LLM_MODEL,
    base_url=LM_API,
    temperature=0.2,
    format=None,
    num_ctx=TOKEN_WINDOW_SIZE,
)

logDir = Path("Orchestrator\logs")

if not os.path.exists(logDir):
    os.makedirs(logDir)


logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #

"""
# ---------------------
# nmap info - basic host info
# ---------------------
class portInfo(BaseModel):
    port: Optional[int] = Field(default=None)
    service: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    state: str = Field(default="unknown")


class HostMemory(BaseModel):
    ip: str = Field(default="")
    status: str = Field(default="unknown")
    open_ports: List[portInfo] = Field(default=[])
    os_guess: Optional[str] = Field(default=None)


# ---------------------
# crawler info - attack vector
# ---------------------


class attackVector(BaseModel):
    endpoint: str = Field(default="")
    method: str = Field(default="")
    parameters: List[str] = Field(default_factory=list)
    vector_type: str = Field(default="")
    confidence: int = Field(default=0)
    cookies: Optional[Dict[str, Any]] = Field(default=dict)
    origins: List[str] = Field(default_factory=list)


# ---------------------
# vulnerabilities
# ---------------------


class vulnerability(BaseModel):
    source_agent: str = Field(default="")
    host: str = Field(default="")
    url: str = Field(default="")
    parameters: List[str] = Field(default_factory=list)
    vulner_type: str = Field(default="")
    severity: str = Field(default="")
    evidence: str = Field(default="")
"""

# ---------------------
# task queue
# ---------------------


class task(BaseModel):
    id: int = Field(default=0)
    agent: Optional[str] = Field(default=None)
    description: str = Field(default="")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=5)
    done: bool = Field(default=False)


class taskQueue(BaseModel):
    queue: List[task] = Field(
        default_factory=list, description="List of tasks made by planning node."
    )


class PlannerOutput(BaseModel):
    queue: List[task]


class expansionOutput(BaseModel):
    new_tasks: List[task]


# ---------------------
# main state
# ---------------------


class executorOutput(BaseModel):
    agent: Optional[Literal["nmap", "gobuster", "sqlmap"]] = Field(default=None)
    agent_prompt: str = Field(default="")


class agentInput(BaseModel):
    prompt: Optional[str] = Field(default=None)
    additional_data: Optional[Any] = Field(default=None)


class agentCall(BaseModel):
    agent: Optional[Literal["nmap", "gobuster", "sqlmap"]] = Field(default=None)
    task_ID: Optional[int] = Field(default=None)
    agent_input: agentInput = Field(default_factory=agentInput)

    execution_status: Literal["pending", "executed", "skipped", "failed", "unknown"] = (
        Field(default="pending")
    )
    message: Optional[str] = Field(default=None)


class reasoningOutput(BaseModel):
    task_ID: Optional[int] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


class reasoningNodeOutput(BaseModel):
    route: Optional[Literal["new", "retry", "expand", "replan", "stop"]] = Field(
        default=None
    )

    selected_task_ID: Optional[int] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


class evaluateNodeOutput(BaseModel):
    decision: Optional[Literal["success", "retry", "expand", "replan"]] = Field(
        default=None
    )
    confidence: float = Field(default=0.0)
    reasoning: Optional[str] = Field(default=None)


class orchestratorState(BaseModel):

    # mission objective
    objective: str = Field(default="", description="Objective given by the user.")

    # task queue
    task_queue: Optional[taskQueue] = Field(default=None)
    new_tasks: Optional[List[task]] = Field(default=None)
    completed_tasks: Optional[List[task]] = Field(default=None)

    # cannonical knowledge
    discovered_hosts: List[str] = Field(default_factory=list)
    host_memory: Dict[str, HostMemory] = Field(default_factory=dict)
    attack_vectors: List[attackVector] = Field(default_factory=list)
    vulnerabilities: List[vulnerability] = Field(default_factory=list)

    # memory
    # retrieved_memory: Dict[Any] = Field(default_factory=dict)

    # raw agent output history
    agent_outputs_history: Dict[int, AgentOutput] = Field(
        default_factory=dict, description="[agent_ID_number, agentOutput]"
    )

    # current agent output
    agent_output: Optional[AgentOutput] = Field(default=None)

    # agent execution
    agent_call: agentCall = Field(default_factory=agentCall)

    # reasoning
    reasoning: reasoningNodeOutput = Field(default_factory=reasoningNodeOutput)

    # evaluate
    evaluate: evaluateNodeOutput = Field(default_factory=evaluateNodeOutput)

    # control
    iteration: int = Field(default=0)
    max_iterations: int = Field(default=100)

    # output
    output_summary: str = Field(default="No summary was given.")

    done: bool = Field(default=False)
    fail: bool = Field(default=False)
    fail_reason: Optional[str] = Field(default=None)


# ------------------------------------------------------------------------------- #
#                              Orchestrator nodes                                 #
# ------------------------------------------------------------------------------- #


async def planningNode(state: orchestratorState):
    logData(message="[PLANNING NODE] -> enter node")

    if state.task_queue is not None and state.reasoning.route != "replan":
        return state

    reason = state.reasoning

    if reason.route == "replan":
        logData(message="[PLANNING NODE] -> replanning...")
        prompt = f"""
        [REPLANNING]
        
        You are a cybersecurity planning agent.
        The current task queue must be replaced with a new plan.

        Objective:
        {state.objective}

        Discovered hosts:
        {state.discovered_hosts}

        Host memory:
        {state.host_memory}

        Attack vectors:
        {state.attack_vectors}

        Vulnerabilities:
        {state.vulnerabilities}

        Feedback from evaluation:
        {state.evaluate.reasoning}
        
        [FEEDBACK FROM REASONING]
        
        {reason.reasoning}
    
        Rules:
        - produce between 3 and 10 tasks
        - tasks must be ordered
        - tasks must map to agents (nmap, gobuster, sqlmap)

        Return JSON:
        - id (choose unique identifiers for tasks in ascending order)
        - agent (suggest agent and explain your decision in "description field")
        - description 
        """

    prompt = f"""
    You are a cybersecurity planning agent.
    
    Your job is to break down the objective into a list of high level tasks required to perform a penetration test.
    
    Objective:
    {state.objective}
    
    Rules:
    - produce between 3 and 10 tasks
    - tasks must be ordered
    - tasks must be actionable
    - tasks must be mapped to a security tool (nmap, gobuster, sqlmap)
    
    Currently available security tools:
    - nmap -> host discovery, host and port scanning, service discovery,...
    - gobuster -> combined with crawler enables detailed host scan for discovering any active endpoints
    - sqlmap -> scans target for any potential injection exploits and perform them 
    
    
    Creat a plan.
    
    Plan must be formulated as a queue with actionable tasks:
    - queue
    
    For each task return JSON:
    - id (choose unique identifiers for tasks in ascending order)
    - agent (suggest agent and explain your decision in "description field")
    - descriptiongetPendingTasks
    """
    retries = 2

    logData(message="[PLANNING NODE] -> execute planning")

    for attempt in range(retries):
        try:
            outputPlanFull = await llm.with_structured_output(
                PlannerOutput, include_raw=True
            ).ainvoke(prompt)
            logData(message=f"[PLANNING NODE] -> created a new plan")

            outputPlan = outputPlanFull["parsed"]
            outputPlanRaw = outputPlanFull["raw"]

            logData(message=f"[PLANNING NODE] -> plan: {outputPlan}")
            state.task_queue = taskQueue(queue=outputPlan.queue)
            logData(message=f"[PLANNING NODE] -> exit node")
            logMetadata(agent_name=AGENT_NAME, metadata=outputPlanRaw.response_metadata)

            return {
                "task_queue": state.task_queue,
            }
        except Exception as e:
            # TODO: add fallback
            logData(message=f"[PLANNING NODE] -> planning failed: {str(e)}")
            attempt += 1


"""
def memoryRetrieveNode(state: orchestratorState, query: Optional[str] = None):
    logData(message="[MEMORY RETRIEVE NODE] -> enter node")

    retrieved = {
        "hosts": [],
        "ports": [],
        "attack_vectors": [],
        "vulnerabilities": [],
    }

    # TODO: change this
    try:
        if query:
            if "host" in query.lower():
                retrieved["hosts"] = getHosts()

            if "port" in query.lower():
                retrieved["hosts"] = getHosts()

            if "attack" in query.lower():
                retrieved["attack_vectors"] = getAttackVector()

            if "vuln" in query.lower():
                retrieved["vulnerabilities"] = getVulnerability()

        else:
            retrieved["hosts"] = getHosts()
            retrieved["hosts"] = getHosts()
            retrieved["attack_vectors"] = getAttackVector()
            retrieved["vulnerabilities"] = getVulnerability()

    except Exception as e:
        logData(message=f"[MEMORY RETRIEVE NODE] -> memory retrieval failed:\n{str(e)}")

    logData(
        message=f"[MEMORY RETRIEVE NODE] -> retrieved: { {k: len(v) for k,v in retrieved.items()} }"
    )

    state.retrieved_memory = retrieved

    logData(message="[MEMORY RETRIEVE NODE] -> exit node")

    return {
        "retrieved_memory": state.retrieved_memory,
    }


def memoryWriteNode(state: orchestratorState):
    logData("[MEMORY WRITE NODE] -> enter node")

    try:
        for host in state.host_memory.values():
            storeHosts(host_memory=HostMemory)
            for port in host.open_ports:
                storePorts(port)

        for vector in state.attack_vectors.values():
            storeAttackVector(attack_vector=vector)

        for vuln in state.vulnerabilities.values():
            storeVulnerability(vulner=vuln)

        logData("[MEMORY WRITE NODE] -> write complete")

    except Exception as e:
        logData(f"[MEMORY WRITE NODE] -> write failed:\n{str(e)}")

    return {}
"""


async def reasoningNode(state: orchestratorState):
    logData(message="[REASONING NODE] -> enter node")

    # iteration check
    state.iteration += 1

    reason = state.reasoning
    evaluateOutput = state.evaluate
    pending_task = getPendingTasks(state.task_queue)

    if not pending_task and state.reasoning.route != "expand":
        logData(message="[REASONING NODE] -> exit node (no more tasks left)")
        state.done = True
        reason.route = "stop"
        reason.reasoning = "All tasks are completed."

        return {
            "done": state.done,
            "iteration": state.iteration,
            "reasoning": state.reasoning,
        }

    if state.iteration >= state.max_iterations:
        logData(message="[REASONING NODE] -> exit node (max iteartions reached)")
        state.fail = True
        reason.route = "stop"
        reason.reasoning = "Maximum number of iterations reached."

        return {
            "fail": state.fail,
            "iteration": state.iteration,
            "reasoning": state.reasoning,
        }

    # -----------------------------
    # deterministic action selector
    # -----------------------------

    if evaluateOutput.decision:
        # evaluate node output
        if evaluateOutput.decision == "retry":
            logData("[REASONING NODE] -> retry same task")
            reason.route = "retry"

        elif evaluateOutput.decision == "expand":
            logData("[REASONING NODE] -> plan needs expansion")
            reason.route = "expand"

        elif evaluateOutput.decision == "replan":
            logData("[REASONING NODE] -> replan is needed")
            reason.route = "replan"

        elif evaluateOutput.decision == "success":
            logData("[REASONING NODE] -> task completed moving to next task")
            reason.route = "new"

    elif reason.route in ["expand", "replan"]:
        logData(f"[REASONING NODE] -> creating new plan after {reason.route}")
        reason.route = "new"

    else:
        reason.route = "new"
        logData("[REASONING NODE] -> creating new plan")

    # -----------------------------
    # reasoning
    # -----------------------------
    if state.reasoning.selected_task_ID:
        taskID = state.reasoning.selected_task_ID
        currentTask = getTaskById(task_id=taskID, queue=state.task_queue)

    if reason.route == "new":
        prompt = f"""
        [TASK SELECTION]
        
        You are a cybersecurity orchestration agent.
        
        Objective:
        {state.objective}
        
        Available tasks:
        {pending_task}
        
        Completed tasks:
        {state.completed_tasks}
        
        Discovered hosts:
        {state.discovered_hosts}
        
        Host memory:
        {state.host_memory}
        
        Attack vectors:
        {state.attack_vectors}
        
        Vulnerabilities:
        {state.vulnerabilities}
        
        Choose the best next task to be executed.
        
        Return valid JSON:
        - task_ID (a valid task ID number)
        - reasoning (reasoning behind picking a certain task)
        """
    elif reason.route == "expand":
        prompt = f"""
        [PLAN EXPANSION]
        
        You are a cybersecurity orchestration agent.
        It was decided that task expansion is needed.
        
        [FEEDBACK FROM EVALUATION]
        
        Confidence: {evaluateOutput.confidence}
        
        Reasoning: {evaluateOutput.reasoning}
        
        Based on the information listed bellow give your additional feedback for expanding the number of tasks.
        Objective:
        {state.objective}
        
        Available tasks:
        {pending_task}
        
        Completed tasks:
        {state.completed_tasks}
        
        Discovered hosts:
        {state.discovered_hosts}
        
        Host memory:
        {state.host_memory}
        
        Attack vectors:
        {state.attack_vectors}
        
        Vulnerabilities:
        {state.vulnerabilities}
        
        
        Return valid JSON:
        - task_ID (LEAVE EMPTY)
        - reasoning (your additional guidance for task expansion)
        """

    elif reason.route == "replan":

        prompt = f"""
        [REPLANNING]
        
        You are a cybersecurity orchestration agent.
        It was decided that current task queue needs replanning.
        
        [FEEDBACK FROM EVALUATION]
        
        Confidence: {evaluateOutput.confidence}
        
        Reasoning: {evaluateOutput.reasoning}
        
        Based on the information listed bellow give your additional feedback for replanning task.
        Objective:
        {state.objective}
        
        Available tasks:
        {pending_task}
        
        Completed tasks:
        {state.completed_tasks}
        
        Discovered hosts:
        {state.discovered_hosts}
        
        Host memory:
        {state.host_memory}
        
        Attack vectors:
        {state.attack_vectors}
        
        Vulnerabilities:
        {state.vulnerabilities}
        
        
        Return valid JSON:
        - task_ID (LEAVE EMPTY)
        - reasoning (your additional guidance for replanning)
        """

    elif reason.route == "retry":

        prompt = f"""
        [RETRY TASK]
        
        You are a cybersecurity orchestration agent.
        It was decided that task needs to be performed again.
        
        [FEEDBACK FROM EVALUATION]
        
        Confidence: {evaluateOutput.confidence}
        
        Reasoning: {evaluateOutput.reasoning}
        
        Based on the information listed bellow give your additional feedback for retrying current task.
        Objective:
        {state.objective}
        
        Current task in progress:
        {currentTask}
        
        Discovered hosts:
        {state.discovered_hosts}
        
        Host memory:
        {state.host_memory}
        
        Attack vectors:
        {state.attack_vectors}
        
        Vulnerabilities:
        {state.vulnerabilities}
        
        
        Return valid JSON:
        - task_ID (return current task ID: {taskID})
        - reasoning (your additional guidance for retrying the task)
        """

    try:
        outputFull = await llm.with_structured_output(
            reasoningOutput, include_raw=True
        ).ainvoke(prompt)
        output = outputFull["parsed"]
        outputRaw = outputFull["raw"]

        logMetadata(agent_name=AGENT_NAME, metadata=outputRaw.response_metadata)
        reason.reasoning = output.reasoning

        if isValidTaskID(output.task_ID, state.task_queue):
            reason.selected_task_ID = output.task_ID
        else:
            reason.selected_task_ID = None

        if reason.route == "retry" and reason.selected_task_ID != taskID:
            reason.selected_task_ID = taskID

    except Exception as e:
        # TODO: add error handling node fallback
        logData(f"[REASONING NODE] -> LLM failure: {str(e)}")

    # reset evaluate node output
    evaluateOutput.decision = None
    evaluateOutput.confidence = 0.0
    evaluateOutput.reasoning = None

    # reset selected task ID if replan or expand was chosen
    if state.reasoning.route in ["replan", "expand"]:
        state.reasoning.selected_task_ID = None

    return {
        "reasoning": state.reasoning,
        "evaluate": state.evaluate,
    }


async def planExpansionNode(state: orchestratorState):
    logData(message="[PLAN EXPANSION NODE] -> enter node")

    pending_tasks = getPendingTasks(state.task_queue)
    reason = state.reasoning

    prompt = f"""
    
    [PLAN EXPANSION]

    You are a cybersecurity orchestration agent.
    The current task plan needs to be expanded.
    
    Objective:
    {state.objective}

    Current pending tasks:
    {pending_tasks}
    
    Completed tasks:
    {state.completed_tasks}

    Discovered hosts:
    {state.discovered_hosts}
    
    Attack vectors:
    {state.attack_vectors}

    Vulnerabilities:
    {state.vulnerabilities}

    [FEEDBACK FROM REASONING]
    
    Reasoning: {reason.reasoning}

    Add new tasks that should be executed.
    
    Rules:
    - do NOT repeat existing tasks
    - produce between 1 and 5 new tasks
    - tasks must map to agents (nmap, gobuster, crawler, sqlmap)

    Return JSON:
    - new_tasks
    """

    try:
        outputFull = await llm.with_structured_output(
            expansionOutput, include_raw=True
        ).ainvoke(prompt)

        output = outputFull["parsed"]
        outputRaw = outputFull["raw"]

        logMetadata(agent_name=AGENT_NAME, metadata=outputRaw.response_metadata)

        nextID = getNextTaskID(state.task_queue)

        for task in output.new_tasks:
            task.id = nextID
            nextID += 1
            state.task_queue.queue.append(task)

        logData(f"[PLAN EXPANSION NODE] -> added {len(output.new_tasks)} tasks")

    except Exception as e:
        logData(f"[PLAN EXPANSION NODE] -> failed: {str(e)}")

    logData(f"[PLAN EXPANSION NODE] -> exit node")
    return {
        "task_queue": state.task_queue,
    }


async def agentExecutionNode(state: orchestratorState):
    logData("[AGENT EXECUTION NODE] -> enter node")

    # get task
    taskID = state.reasoning.selected_task_ID
    task = getTaskById(task_id=taskID, queue=state.task_queue)

    if task is None:
        logData(message="[AGENT EXECUTION NODE] -> no valid task selected")
        state.fail = True
        state.fail_reason = "No valid task selected by reasoning Node"
        return {
            "fail": state.fail,
            "fail_reason": state.fail_reason,
        }

    logData(message=f"[AGENT EXECUTION NODE] -> selected task: {task.id}")

    reason = state.reasoning

    prompt = f"""
    You are a cybersecurity tool agent.
    Your job is to decide which of the given tools should be used and how based on the current task.
    
    [SELECTED TASK]
    
    Task ID: {taskID}
    Task: {task}
    
    Reasoning behind selection fot he current task: {reason.reasoning}
    
    [ADDITIONAL INFORMATION]
    
    Known hosts:
    {state.discovered_hosts}

    Host memory:
    {state.host_memory}

    Attack vectors:
    {state.attack_vectors}

    Vulnerabilities:
    {state.vulnerabilities}
    
    You must pick exactly one of the following tool agents:
    - nmap
    - sqlmap
    - gobuster
    
    Additionaly create a simple 1 sentence instruction or prompt for the chosen agent (WITHOUT any specific tool calls).
    
    Return valid JSON:
    - agent (exactly on of the listed tool agents)
    - agent_prompt
    """

    outputFull = await llm.with_structured_output(
        executorOutput, include_raw=True
    ).ainvoke(prompt)

    output = outputFull["parsed"]
    outputRaw = outputFull["raw"]

    logMetadata(agent_name=AGENT_NAME, metadata=outputRaw.response_metadata)

    logData(message=f"[AGENT EXECTUION NODE] -> created an agent call: {output}")

    state.agent_call.agent = output.agent
    state.agent_call.task_ID = taskID

    requirements = AGENT_REQUIREMENTS.get(output.agent, [])

    logData(
        message=f"[AGENT EXECUTION NODE] -> checking requirements for {state.agent_call.agent}: {requirements}"
    )

    if not requirements == []:
        for req in requirements:
            logData(message="[AGENT EXECUTION NODE] -> checking requirements")

            if getattr(state, req) is not None and output.agent_prompt:
                logData(
                    message=f"[AGENT EXECUTION NODE] -> requirements for {output.agent} agent satisfied"
                )
                state.agent_call.agent_input.prompt = output.agent_prompt
                state.agent_call.execution_status = "pending"

                if output.agent == "sqlmap":
                    vectors = getattr(state, req)
                    untestedVectors = []

                    for vector in vectors:
                        if not vector.tested:
                            untestedVectors.append(vector)

                    state.agent_call.agent_input.additional_data = untestedVectors
                else:
                    state.agent_call.agent_input.additional_data = getattr(state, req)

                break

            else:
                state.agent_call.agent_input.prompt = output.agent_prompt
                state.agent_call.execution_status = "skipped"
                logData(
                    f"[AGENT EXECUTION NODE] -> {output.agent} agent call skipped due to missing arguments"
                )

    else:
        logData(
            message="[AGENT EXECUTION NODE] -> Agent doesn't have any specifc requirements, saving prompt..."
        )
        state.agent_call.agent_input.prompt = output.agent_prompt
        state.agent_call.execution_status = "pending"

    logData(message="[AGENT EXECUTION NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
    }


async def nmapAgentNode(state: orchestratorState):
    # TODO: reset previous agent call output in reasoning node!
    logData("[NMAP NODE] -> enter node")

    agentCall = state.agent_call
    agent_ID = agentCall.task_ID

    for retries in range(2):
        try:
            logData(
                f"[NMAP NODE] -> calling nmap agent with following prompt: {agentCall.agent_input.prompt}"
            )
            logData(
                "================================ NMAP AGENT start ================================"
            )
            result = await NMAP_GRAPH.ainvoke(
                {"objective": agentCall.agent_input.prompt},
                {"recursion_limit": 1000},
            )
            logData(
                "================================ NMAP AGENT stop ================================"
            )

            # logData(f"[NMAP NODE] -> Raw agent output:\n\n{result}")

            agent_output = result.get("agent_output")
            logData(f"[NMAP NODE] -> Raw full output:\n\n{result}")
            logData(f"[NMAP NODE] -> Raw agent output:\n\n{agent_output}")

            state.agent_output = agent_output
            state.agent_outputs_history[agent_ID] = agent_output

            if agent_output.success:
                logData("[NMAP NODE] -> agent call successful")
                state.discovered_hosts = list(
                    set(state.discovered_hosts + agent_output.discovered_hosts)
                )
                for ip, host in agent_output.host_memory.items():
                    state.host_memory[ip] = host

                agentCall.execution_status = "executed"

                logData("[NMAP NODE] -> exit node")
                return {
                    # agent specific update
                    "agent_output": state.agent_output,
                    "agent_outputs_history": state.agent_outputs_history,
                    "agent_call": state.agent_call,
                    # global knowledge update
                    "discovered_hosts": state.discovered_hosts,
                    "host_memory": state.host_memory,
                }
            else:
                logData("[NMAP NODE] -> agent call failed")
                agentCall.execution_status = "failed"

                logData("[NMAP NODE] -> exit node")
                return {
                    # agent specific update
                    "agent_output": state.agent_output,
                    "agent_outputs_history": state.agent_outputs_history,
                    "agent_call": state.agent_call,
                }

        except Exception as e:
            logData("[NMAP NODE] -> agent call produced an exception")
            agentCall.message = (
                f"Failed to call {agentCall.agent} within {retries} retries. Last call produced the following error:\n\n"
                + str(e)
            )
            agentCall.execution_status = "failed"

    logData("[NMAP NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
    }


async def gobusterAgentNode(state: orchestratorState):
    logData(message="[GOBUSTER NODE] -> enter node")
    agentCall = state.agent_call
    agentID = agentCall.task_ID

    # TODO: raw tool agent output logging
    for retries in range(2):
        try:

            logData(
                "================================ GOBUSTER AGENT start ================================"
            )

            result = await GOBUSTER_GRAPH.ainvoke(
                {"objective": agentCall.agent_input.prompt},
                {"recursion_limit": 1000},
            )

            logData(
                "================================ GOBUSTER AGENT stop ================================"
            )

            agent_output = result.get("agent_output")

            state.agent_output = agent_output
            state.agent_outputs_history[agentID] = agent_output

            if agent_output.success:
                logData(message="[GOBUSTER NODE] -> agent call successful")
                existing = {(v.endpoint, v.method) for v in state.attack_vectors}

                for vec in agent_output.attack_vectors:
                    key = (vec.endpoint, vec.method)
                    if key not in existing:
                        state.attack_vectors.append(vec)

                agentCall.execution_status = "executed"

                logData(message="[GOBUSTER NODE] -> exit node")
                return {
                    # agent specific updates
                    "agent_call": state.agent_call,
                    "agent_output": state.agent_output,
                    "agent_outputs_history": state.agent_outputs_history,
                    # global knowledge update
                    "attack_vectors": state.attack_vectors,
                }
            else:
                logData(message="[GOBUSTER NODE] -> agent call failed")
                agentCall.execution_status = "failed"

                logData(message="[GOBUSTER NODE] -> exit node")
                return {
                    "agent_call": state.agent_call,
                    "agent_output": state.agent_output,
                    "agent_outputs_history": state.agent_outputs_history,
                }

        except Exception as e:

            logData("[GOBUSTER NODE] -> agent call produced an exception")

            agentCall.message = (
                f"Failed to call {agentCall.agent} w ithin {retries} retries. Last call produced the following error:\n\n"
                + str(e)
                if e
                else "No execption message recieved!"
            )
            agentCall.execution_status = "failed"

    logData(message="[GOBUSTER NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
    }


async def sqlmapAgentNode(state: orchestratorState):
    logData("[SQLMAP NODE] -> enter node")

    agentCall = state.agent_call
    agent_ID = agentCall.task_ID

    vectors = agentCall.agent_input.additional_data or []

    for vector in vectors:
        for stateVector in state.attack_vectors:
            if (
                stateVector.endpoint == vector.endpoint
                and stateVector.method == vector.method
                and stateVector.parameters == vector.parameters
            ):
                stateVector.tested = True

    for retries in range(2):
        try:

            logData(
                "================================ SQLMAP AGENT start ================================"
            )

            result = await SQLMAP_GRAPH.ainvoke(
                {
                    "objective": agentCall.agent_input.prompt,
                    "attack_vectors": agentCall.agent_input.additional_data,
                },
                {"recursion_limit": 1000},
            )

            logData(
                "================================ SQLMAP AGENT stop ================================"
            )

            agent_output = result.get("agent_output")

            state.agent_output = agent_output
            state.agent_outputs_history[agent_ID] = agent_output

            if agent_output.success:
                logData(message="[SQLMAP NODE] -> agent call successful")
                existing = {
                    (v.host, v.url, tuple(v.parameters), v.vulner_type)
                    for v in state.vulnerabilities
                }

                for vuln in agent_output.vulnerabilities:
                    key = (
                        vuln.host,
                        vuln.url,
                        tuple(vuln.parameters),
                        vuln.vulner_type,
                    )

                    if key not in existing:
                        state.vulnerabilities.append(vuln)

                agentCall.execution_status = "executed"

                logData(message="[SQLMAP NODE] -> exit node")
                return {
                    # agent specific updates
                    "agent_call": state.agent_call,
                    "agent_output": state.agent_output,
                    "agent_outputs_history": state.agent_outputs_history,
                    # global knowledge update
                    "vulnerabilities": state.vulnerabilities,
                }

            else:
                logData(message="[SQLMAP NODE] -> agent call failed")
                agentCall.execution_status = "failed"

                logData(message="[SQLMAP NODE] -> exit node")
                return {
                    # agent specific updates
                    "agent_call": state.agent_call,
                    "agent_output": state.agent_output,
                    "agent_outputs_history": state.agent_outputs_history,
                }

        except Exception as e:

            logData("[SQLMAP NODE] -> agent call produced an exception")

            agentCall.message = (
                f"Failed to call {agentCall.agent} within {retries} retries. Last call produced the following error:\n\n"
                + str(e)
            )
            agentCall.execution_status = "failed"

    logData(message="[SQLMAP NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
    }


async def evaluateNode(state: orchestratorState):
    logData(message="[EVALUATE NODE] -> enter node")
    # 3 types of prompts based on the execution status: failed, skipped, executed
    agentCall = state.agent_call
    taskID = state.reasoning.selected_task_ID
    task = getTaskById(task_id=taskID, queue=state.task_queue)

    initialPrompt = f"""
        [INFO]
        You are a cybersecurity orchestration evaluation agent.
        
        [TASK]
        Your job is to evaluate the result of the last tool agent execution
        and decide what the orchestrator should do next.
        
        You MUST evaluate the success of the current task based on:
            - the task description
            - the tool execution result
            - the current global knowledge
                
        You do NOT create new tasks or plans.
        You ONLY evaluate the result and return the next decision.
        
        Objective: {state.objective}
        
        Task ID: {taskID}
        
        Task description:
        {task.description}
        
        Retry count: {task.retry_count}/{task.max_retries}
        """

    if agentCall.execution_status == "executed":
        print("placeholder")

        situationalPrompt = f"""
        [TOOL]
        
        Tool execution was successful!
        
        Selected agent: {agentCall.agent}
        
        Execution status: {agentCall.execution_status}
        
        Tool output:
        {state.agent_output}
        """

    elif agentCall.execution_status == "skipped":
        situationalPrompt = f"""
        [WARNING]
        
        Tool agent was skipped!
        
        Selected agent: {agentCall.agent}
        
        Execution status: {agentCall.execution_status}
        
        Warning message: {agentCall.message}
        """
    elif agentCall.execution_status == "failed":
        situationalPrompt = f"""
        [WARNING]
        
        Tool agent failed!
        
        Selected agent: {agentCall.agent}
        
        Execution status: {agentCall.execution_status}
        
        Error message: {agentCall.message if agentCall.message else "None"}
        
        Tool output:
        {state.agent_output if state.agent_output else "None"}
        """
    else:
        situationalPrompt = f"""
        [WARNING]
        
        Agent's position and tool execution progress is not clearly defined!
        
        Selected agent: {agentCall.agent}
        
        Execution status: {agentCall.execution_status}
        
        Tool output:
        {state.agent_output if state.agent_output else "None"}
        """

    additonalPrompt = f"""
    [RULES]
    
    You must select one of the following decisions:

    success
    - The task was completed successfully.

    retry
    - The task failed but should be attempted again.

    expand
    - The task succeeded and discovered new potential attack surface.
    - The planner should generate additional tasks.

    replan
    - The current plan is ineffective or invalid.
    - The planner should generate a new plan.
    
    [SPECIAL CASES]
    If execution_status is "skipped":
    - The tool was not executed due to missing requirements.
    - Usually this should result in "replan" or "expand".

    If execution_status is "failed":
    - Retry if retries are still available.
    - Otherwise select "replan".

    If execution_status is "executed":
    - Evaluate the actual results returned by the agent.
    
    
    Return valid JSON with:

    decision: one of ["success", "retry", "expand", "replan"]
    confidence: number between 0.0 and 1.0
    reasoning: helpful explanation for orchestrator
    """
    finalPrompt = initialPrompt + situationalPrompt + additonalPrompt

    result = None

    try:
        resultFull = await llm.with_structured_output(
            evaluateNodeOutput, include_raw=True
        ).ainvoke(finalPrompt)

        result = resultFull["parsed"]
        resultRaw = resultFull["raw"]

        logMetadata(agent_name=AGENT_NAME, metadata=resultRaw.response_metadata)

        state.evaluate = result

    except Exception as e:
        logData(f"[EVALUATE NODE] -> LLM failure: {str(e)}")

    # TODO: add current agent output clearning, task marking, task iteration etc.

    if result.decision == "success":
        state.task_queue, finishedTask = completeTask(
            queue=state.task_queue, task_id=taskID
        )

        if finishedTask:
            if state.completed_tasks is None:
                state.completed_tasks = []

            state.completed_tasks.append(finishedTask)

    if result.decision == "retry":
        task.retry_count += 1

        if task.retry_count >= task.max_retries:
            state.evaluate.decision = "replan"

    state.agent_call.execution_status = "pending"
    state.agent_call.message = None
    state.agent_output = None

    if result and result.decision != "retry":
        state.reasoning.selected_task_ID = None

    if state.iteration > 10 and len(state.completed_tasks or []) == 0:
        state.evaluate.decision = "replan"

    logData(message="[EVALUATE NODE] -> exit node")
    return {
        "evaluate": state.evaluate,
        "task_queue": state.task_queue,
        "completed_tasks": state.completed_tasks,
        "agent_call": state.agent_call,
        "agent_output": state.agent_output,
    }


async def outputNode(state: orchestratorState):
    # TODO: change invoke prompt, dont dump whole agent state into the prompt!! (unless you want to make a token bomb)
    if state.done:
        prompt = f"""
        You are a cybersecurity reporting agent.
        The autonomous penetration testing workflow has completed successfully.
        
        [TASK]
        Your job is to generate a clear and detailed summary of the execution based on the final state of the orchestrator.
        
        [MEMORY]
        {state.model_dump(mode="json")}
        
        Instructions:
        - Summarize what the system did.
        - Describe the key findings.
        - Highlight any discovered vulnerabilities.
        - Keep the report concise but informative.
        - Write the report as if it will be read by a security engineer.
        """

    else:
        prompt = f"""
        You are a cybersecurity reporting agent.
        
        [WARNING]
        The autonomous penetration testing workflow failed before completing the objective.
        
        Reason: {state.fail_reason}
        
        [TASK]
        Your job is to analyze the failure and generate a report explaining what went wrong.
        
        [MEMORY]
        {state.model_dump(mode="json")}
        
        Instructions:
        - Explain what the system was trying to do.
        - Describe where the failure occurred.
        - Provide a possible reason for the failure.
        - Suggest what could be done to fix or improve the process.
        """

    response = await llm.ainvoke(prompt)

    logMetadata(agent_name=AGENT_NAME, metadata=response.response_metadata)

    state.output_summary = response.content

    return {
        "output_summary": state.output_summary,
    }


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #


def setupLogger():
    logFile = logDir / f"orchestrator_log{logCount}.log"
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)

    file = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file.setFormatter(format)
    logger.addHandler(file)

    return logger


def logData(message: str):
    logging.getLogger("orchestrator").info(message)


def getPendingTasks(queue: taskQueue):

    return [t for t in queue.queue if not t.done]


def completeTask(queue: taskQueue, task_id: int) -> tuple[taskQueue, Optional[task]]:

    for i, t in enumerate(queue.queue):
        if t.id == task_id:
            t.done = True

            # remove from queue
            completed_task = queue.queue.pop(i)

            return queue, completed_task

    return queue, None


def getTaskById(task_id: int, queue: taskQueue):

    for t in queue.queue:
        if t.id == task_id:
            return t

    return None


def isValidTaskID(task_id: int, queue: taskQueue) -> bool:
    if task_id is None:
        return False

    for t in queue.queue:
        if t.id == task_id and not t.done:
            return True

    return False


def getNextTaskID(queue: taskQueue):
    lastTask = queue.queue[-1]
    return lastTask.id + 1


def getReasoiningRouting(state: orchestratorState):
    route = state.reasoning.route

    if route in ["new", "retry"]:
        return "execute"

    else:
        return state.reasoning.route


def getAgentRouting(state: orchestratorState):
    agentCall = state.agent_call

    if agentCall.execution_status == "skipped":
        return "evaluate"
    else:
        return agentCall.agent


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


def orchestratorBuilder():
    setupLogger()
    workflow = StateGraph(orchestratorState)

    # -------------------------------
    # graph nodes
    # -------------------------------

    # TODO: later add memory nodes
    workflow.add_node("planning_node", planningNode)
    workflow.add_node("reasoning_node", reasoningNode)
    workflow.add_node("plan_expansion_node", planExpansionNode)
    workflow.add_node("agent_execution_node", agentExecutionNode)
    workflow.add_node("nmap_agent_node", nmapAgentNode)
    workflow.add_node("gobuster_agent_node", gobusterAgentNode)
    workflow.add_node("sqlmap_agent_node", sqlmapAgentNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------

    workflow.add_edge(START, "planning_node")
    workflow.add_edge("planning_node", "reasoning_node")
    workflow.add_conditional_edges(
        "reasoning_node",
        getReasoiningRouting,
        {
            "execute": "agent_execution_node",
            "expand": "plan_expansion_node",
            "replan": "planning_node",
            "stop": "output_node",
        },
    )
    workflow.add_edge("plan_expansion_node", "reasoning_node")
    workflow.add_conditional_edges(
        "agent_execution_node",
        getAgentRouting,
        {
            "sqlmap": "sqlmap_agent_node",
            "nmap": "nmap_agent_node",
            "gobuster": "gobuster_agent_node",
            "evaluate": "evaluate_node",
        },
    )
    workflow.add_edge("sqlmap_agent_node", "evaluate_node")
    workflow.add_edge("nmap_agent_node", "evaluate_node")
    workflow.add_edge("gobuster_agent_node", "evaluate_node")
    workflow.add_edge("evaluate_node", "reasoning_node")
    workflow.add_edge("output_node", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "Orchestrator\orchestrator_agent.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    return graph


async def orchestratorRunner(prompt: str):
    graph = orchestratorBuilder()

    state = orchestratorState()
    state.objective = prompt

    config: RunnableConfig = {
        "configurable": {"thread_id": "1"},
        "recursion_limit": 1000,
    }

    response = await graph.ainvoke(state, config=config)
    print(f"[FINAL OUTPUT RAW]: {response}")


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("\n#" + "-" * 20 + "Penetrator_petar_test" + 20 * "-" + "#\n")
    testPrompt = """
    Perform a full reconnaissance and vulnerability discovery workflow in network 192.168.157.0/24

    The workflow should:
    1. Discover the target host and open ports.
    2. Identify any running web services.
    3. Enumerate directories and application endpoints.
    4. Identify potential SQL injection attack vectors on discovered endpoints.
    5. Attempt to confirm SQL injection vulnerabilities.

    Summarize all discovered attack vectors and vulnerabilities.
    """

    asyncio.run(orchestratorRunner(prompt=testPrompt))
