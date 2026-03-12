from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any, Literal
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import json
import logging
from pathlib import Path

# tool agents
from MCP_tools.nmap.nmap_agent_ollama import agentRunner as nmapAgent
from MCP_tools.gobuster.gobuster_agent_ollama import agentRunner as gobusterAgent
from MCP_tools.crawler import main as crawlerMain
from MCP_tools.sqlmap.sqlmap_agent_ollamaV3 import agentRunner as sqlmapAgent

# output schema
from Orchestrator.memory.agent_output import AgentOutput

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


load_dotenv()
initializeDB()

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

logDir = Path("MCP_tools/Orchestartor/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


# ---------------------
# nmap info - basic host info
# ---------------------
class portInfo(BaseModel):
    port: Optional[int] = Field(default=None)
    service: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    state: str = Field(default="unknown")


class hostMemory(BaseModel):
    ip: str = Field(default="")
    status: str = Field(default="unknown")
    open_ports: List[portInfo] = Field(default=[])
    os_guess: Optional[str] = Field(default=None)


# ---------------------
# crawler info - attack vector
# ---------------------


class attackVector(BaseModel):
    url: str = Field(default="")
    method: str = Field(default="")
    parameters: List[str] = Field(default_factory=list)
    origins: List[str] = Field(default_factory=list)


# ---------------------
# vulnerabilities
# ---------------------


class vulnerability(BaseModel):
    host: str = Field(default="")
    url: str = Field(default="")
    parameters: List[str] = Field(default_factory=list)
    vulner_type: str = Field(default="")
    severity: str = Field(default="")
    evidence: str = Field(default="")


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


class executorOutut(BaseModel):
    agent: Literal["nmap", "crawler", "sqlmap"]
    agent_prompt: str = Field(default="")


class agentCall(BaseModel):
    agent: Literal["nmap", "crawler", "sqlmap"]
    task_ID: Optional[int] = Field(default=None)
    agent_prompt: str = Field(default="")


class reasoningOutput(BaseModel):
    task_ID: Optional[str] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


class reasoningNodeOutput(BaseModel):
    route: Literal["new", "retry", "expand", "replan", "stop", None]

    selected_task_ID: Optional[int] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


class evaluateNodeOutput(BaseModel):
    decision: Literal["success", "retry", "expand", "replan", None]
    confidence: int = Field(default=0.0)
    reasoning: Optional[str] = Field(default=None)


class orchestratorState(BaseModel):

    # mission objective
    objective: str = Field(default="", description="Objective given by the user.")

    # task queue
    task_queue: Optional[taskQueue] = Field(default=None)
    new_tasks: Optional[List[task]] = Field(default=None)

    # cannonical knowledge
    discovered_hosts: List[str] = Field(default_factory=list)
    host_memory: Dict[str, hostMemory] = Field(default_factory=dict)
    attack_vectors: List[attackVector] = Field(default_factory=list)
    vulnerabilities: List[vulnerability] = Field(default_factory=list)

    # memory
    retrieved_memory: Dict[Any] = Field(default_factory=dict)

    # raw agent output history
    agent_outputs: Dict[str, AgentOutput] = Field(default_factory=dict)

    # agent execution
    agent_call: agentCall = Field(default_factory=agentCall)

    # reasoning
    reasoning: reasoningNodeOutput = Field(default_factory=reasoningNodeOutput)

    # evaluate
    evaluate: evaluateNodeOutput = Field(default_factory=evaluateNodeOutput)

    # control
    iteration: int = Field(default=0)
    max_iterations: int = Field(default=100)

    done: bool = Field(default=False)
    fail: bool = Field(default=False)
    fail_reason: Optional[str] = Field(default=None)


# ------------------------------------------------------------------------------- #
#                              Orchestrator nodes                                 #
# ------------------------------------------------------------------------------- #


def planningNode(state: orchestratorState):
    logData(message="[PLANNING NODE] -> enter node")

    if state.task_queue is not None:
        return state

    # TODO: add supervisor feedback when creating a plan
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
    - description
    """
    retries = 2

    for attempt in range(retries):
        try:
            outputPlan = llm.with_structured_output(PlannerOutput).invoke(prompt)
            logData(message=f"[PLANNING NODE] -> created a new plan")

            state.task_queue = taskQueue(queue=outputPlan.queue)
            logData(message=f"[PLANNING NODE] -> exit node")

            return {
                "task_queue": state.task_queue,
            }
        except Exception as e:
            # TODO: add fallback
            logData(message=f"[PLANNING NODE] -> planning failed: {str(e)}")
            attempt += 1


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
            storeHosts(host_memory=hostMemory)
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


def reasoningNode(state: orchestratorState):
    logData(message="[REASONING NODE] -> enter node")

    # iteration check
    state.iteration += 1

    reason = state.reasoning
    evaluateOutput = state.evaluate
    pending_task = getPendingTasks(state.task_queue)

    if not pending_task:
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
            # TODO: add llm feedback
            reason.route = "retry"

        elif evaluateOutput.decision == "expand":
            logData("[REASONING NODE] -> plan needs expansion")
            # TODO: add llm feedback
            reason.route = "expand"

        elif evaluateOutput.decision == "replan":
            logData("[REASONING NODE] -> replan is needed")
            reason.route = "replan"

        elif evaluateOutput.decision == "success":
            logData("[REASONING NODE] -> tasl completed moving to next task")
            reason.route = "new"

    elif reason.route in ["expand", "replan"]:
        reason.route = "new"

    else:
        reason.route = "retry"

    # -----------------------------
    # reasoning
    # -----------------------------

    if reason.route == "new":
        prompt = f"""
        [TASK SELECTION]
        
        You are a cybersecurity orchestration agent.
        
        Objective:
        {state.objective}
        
        Available tasks:
        {pending_task}
        
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
        
        Based on the information listed bellow give your additional feedback for replanning tasks.
        Objective:
        {state.objective}
        
        Available tasks:
        {pending_task}
        
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
        currentTask = getTaskById(state.agent_call.task_ID, state.task_queue)

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
        - task_ID (LEAVE EMPTY)
        - reasoning (your additional guidance for retrying the task)
        """

    try:
        output = llm.with_structured_output(reasoningOutput).invoke(prompt)
        reason.reasoning = output.reasoning

        if isValidTaskID(output.task_ID, state.task_queue):
            reason.selected_task_ID = output.task_ID
        else:
            reason.selected_task_ID = None

    except Exception as e:
        # TODO: add error handling node fallback
        logData(f"[REASONING NODE] -> LLM failure: {str(e)}")

    # reset evaluate node output
    evaluateOutput.decision = None
    evaluateOutput.confidence = 0.0
    evaluateOutput.reasoning = None

    return {
        "reasoning": state.reasoning,
        "evaluate": state.evaluate,
    }


def planExpansionNode(state: orchestratorState):
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
        output = llm.with_structured_output(expansionOutput).invoke(prompt)

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


def replanNode(state: orchestratorState):
    logData(f"[REPLAN NODE] -> enter node")
    reason = state.reasoning

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

    try:
        output = llm.with_structured_output(PlannerOutput).invoke(prompt)
        state.task_queue = taskQueue(queue=output.queue)

        logData("[REPLAN NODE] -> task queue replaced")

    except Exception as e:
        logData(f"[REPLAN NODE] -> failed: {str(e)}")

    return {
        "task_queue": state.task_queue,
    }


def agentExecutionNode(state: orchestratorState):
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
    
    You must pick on of the bellow listed available tool agents:
    - nmap
    - sqlmap
    - crawler
    
    Additionaly create a simple 1 sentence instruction or prompt for the chosen agent (WITHOUT any specific tool calls).
    
    Return valid JSON:
    - agent (exactly on of the listed tool agents)
    - agent_prompt
    """

    output = llm.with_structured_output(executorOutut).invoke(prompt)

    state.agent_call.agent = output.agent
    state.agent_call.task_ID = taskID
    state.agent_call.agent_prompt = output.agent_prompt

    return {
        "agent_call": state.agent_call,
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


def markTaskDone(queue: taskQueue, task_id: int):

    for t in queue.queue:
        if t.id == task_id:
            t.done = True
            return


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
