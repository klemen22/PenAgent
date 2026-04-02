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
import re

# tool agents for subgraphs
from MCP_tools.nmap.nmap_agent_ollamaV2 import nmapBuilder
from MCP_tools.gobuster.gobuster_agent_ollamaV2 import gobusterBuilder
from MCP_tools.sqlmap.sqlmap_agent_ollamaV3 import sqlmapBuilder
from metadata.metadata_logger import setupMetadataLogger, logMetadata
from reasoning.reasoning_logger import setupReasoningLogger, logReasoning
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
        "requires": [],
        "input_schema": ["targets"],
    },
    "gobuster": {
        "capabilities": ["directory_enum", "web_enum"],
        "requires": ["host_memory"],
        "input_schema": ["targets"],
    },
    "sqlmap": {
        "capabilities": ["sql_injection"],
        "requires": ["attack_vectors"],
        "input_schema": ["attack_vectors"],
    },
}

AGENT_REQUIREMENTS = {"nmap": [], "crawler": [], "sqlmap": ["attack_vectors"]}
AGENT_NAME = "orchestrator_agent"

setupMetadataLogger(agent_name=AGENT_NAME)
setupReasoningLogger(agentName=AGENT_NAME)

logReasoning(
    agentName=AGENT_NAME, reasoning=f" {20 * "="} ORCHESTRATOR REASONING {20 * "="} "
)
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
    temperature=0.1,
    format=None,
    num_ctx=TOKEN_WINDOW_SIZE,
    system="You are a specialized cybersecurity assistant. You MUST always respond in English. Do not use any other languages under any circumstances.",
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
    agent: Optional[Literal["nmap", "gobuster", "sqlmap"]] = Field(default=None)
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
    agent: Literal["nmap", "gobuster", "sqlmap"]
    agent_prompt: str = Field(default="")
    target_host: Optional[List[str]] = Field(
        default=None, description="The specific IP or URL to target"
    )


class agentInput(BaseModel):
    prompt: Optional[str] = Field(default=None)
    targets: List[str] = Field(default=[])
    attack_vectors: List[attackVector] = Field(default=[])
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
    reasoning: Optional[str] = Field(default=None)
    decision: Optional[Literal["success", "retry", "expand", "replan"]] = Field(
        default=None
    )
    confidence: float = Field(default=0.0)


class agentCounter(BaseModel):
    last_agent: Optional[Literal["nmap", "gobuster", "sqlmap"]] = Field(default=None)
    last_agent_iteration: int = Field(default=0)


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
    host_enum: Dict[str, Dict] = Field(
        default_factory=dict
    )  # add gobuster enum output also
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
    agent_counter: agentCounter = Field(default_factory=agentCounter)

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

    completedTasks = []
    if state.task_queue and state.task_queue.queue:
        completedTasks = [task for task in state.task_queue.queue if task.done]

    taskSummary = "\n".join(
        [f"- ID {t.id}: {t.agent} ({t.description}) -> DONE" for t in completedTasks]
    )

    if reason.route == "replan":
        logData(message="[PLANNING NODE] -> replanning...")
        prompt = f"""
        [REPLANNING]
        You are a cybersecurity planning agent. An error or a change in circumstances occurred.
        Your job is to update the plan without repeating work that is already finished.

        [OBJECTIVE]
        {state.objective}
        
        [ALREADY FINISHED TASKS]
        {taskSummary if taskSummary else "No tasks finished yet."}

        [CURRENT CONTEXT]
        > Discovered hosts: {state.discovered_hosts}
        > Host memory: {state.host_memory}
        > Attack vectors: {state.attack_vectors}
        > Vulnerabilities: {state.vulnerabilities}
            
        [WHY ARE WE REPLANNING?]
        Feedback from evaluator: {state.evaluate.reasoning}
        Reasoning: {reason.reasoning}
        
        [TASK]
        Create a NEW set of tasks to complete the objective from where we left off.
        DO NOT include tasks that are already marked as DONE.
        Focus on the remaining steps (e.g., if nmap is done, focus on gobuster/sqlmap).
        
        [AVAILABLE TOOLS]
        - nmap -> host discovery, host and port scanning, service discovery,...
        - gobuster -> combined with crawler enables detailed host scan for discovering any active endpoints
        - sqlmap -> scans target for any potential injection exploits and perform them 
        
        CURRENTLY EXPLOITING IS ONLY DONE WITH SQLMAP
        
        [RULES]
        - produce between 3 and 10 tasks
        - tasks must be ordered
        - tasks must be actionable
        - tasks must map to agents (nmap, gobuster, sqlmap)

        [TASK]
        Creat a plan.
        Plan must be formulated as a queue with new actionable tasks:
        - queue
        
        For each new task return JSON:
        - id (choose unique identifiers for tasks in ascending order)
        - agent (suggest agent and explain your decision in "description field")
        - descriptiongetPendingTasks
        """
    else:
        prompt = f"""
        [INITIAL PLANNING]
        You are a cybersecurity planning agent.
        Your job is to break down the objective into a list of high level tasks required to perform a penetration test.
        
        [OBJECTIVE]
        {state.objective}
        
        [RULES]
        - produce between 3 and 10 tasks
        - tasks must be ordered
        - tasks must be actionable
        - tasks must be mapped to a security tool (nmap, gobuster, sqlmap)
        
        [AVAILABLE TOOLS]
        - nmap -> host discovery, host and port scanning, service discovery,...
        - gobuster -> combined with crawler enables detailed host scan for discovering any active endpoints
        - sqlmap -> scans target for any potential injection exploits and perform them 
        
        CURRENTLY EXPLOITING IS ONLY DONE WITH SQLMAP
        
        [TASK]
        Creat a plan.
        Plan must be formulated as a queue with actionable tasks:
        - queue
        
        For each task return JSON:
        - id (choose unique identifiers for tasks in ascending order)
        - agent (suggest agent and explain your decision in "description field")
        - descriptiongetPendingTasks
        """
    retries = 3

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
            logReasoning(
                agentName=AGENT_NAME,
                reasoning=f"[PLANNING] \n{json.dumps(outputPlan.model_dump(), indent=4)}",
            )
            newTasks = outputPlan.queue
            startID = len(completedTasks) + 1
            for i, task in enumerate(newTasks):
                task.id = startID + i
                task.done = False

            finalQueue = newTasks.extend(completedTasks) if completedTasks else newTasks

            state.task_queue = taskQueue(queue=finalQueue)
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


async def routerNode(state: orchestratorState):
    # iteration check
    state.iteration += 1

    reason = state.reasoning
    evaluateOutput = state.evaluate
    pending_task = getPendingTasks(state.task_queue)

    if not pending_task and state.reasoning.route != "expand":
        logData(message="[ROUTING NODE] -> exit node (no more tasks left)")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning="[ROUTER] No more tasks left, stopping.",
        )
        state.done = True
        reason.route = "stop"
        reason.reasoning = "All tasks are completed."

        return {
            "done": state.done,
            "iteration": state.iteration,
            "reasoning": state.reasoning,
        }

    if state.iteration >= state.max_iterations:
        logData(message="[ROUTING NODE] -> exit node (max iteartions reached)")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning="[ROUTER] Max iterations reached, stopping.",
        )
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
            logData("[ROUTING NODE] -> retry same task")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning="[ROUTER] Retrying same task.",
            )
            reason.route = "retry"

        elif evaluateOutput.decision == "expand":
            logData("[ROUTING NODE] -> plan needs expansion")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning="[ROUTER] Plan needs expansion.",
            )
            reason.route = "expand"

        elif evaluateOutput.decision == "replan":
            logData("[ROUTING NODE] -> replan is needed")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning="[ROUTER] Replanning is needed.",
            )
            reason.route = "replan"

        elif evaluateOutput.decision == "success":
            logData("[ROUTING NODE] -> task completed moving to next task")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning="[ROUTER] Task completed, moving to next task.",
            )
            reason.route = "new"

    elif reason.route in ["expand", "replan"]:
        logData(f"[ROUTING NODE] -> creating new plan after {reason.route}")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[ROUTER] Selecting new task after {reason.route}",
        )
        reason.route = "new"

    else:
        reason.route = "new"
        logData("[ROUTING NODE] -> selecting new task")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning="Selecting new task.",
        )
    logData("[ROUTER NODE] -> exit node")
    return {
        "iteration": state.iteration,
        "reasoning": state.reasoning,
    }


async def reasoningNode(state: orchestratorState):
    logData(message="[REASONING NODE] -> enter node")
    reason = state.reasoning
    evaluateOutput = state.evaluate
    # pending_task = getPendingTasks(state.task_queue)

    shortInfo = {
        ip: [p.port for p in h.open_ports] for ip, h in state.host_memory.items()
    }

    contextData = f"""
    OBJECTIVE: {state.objective}
    PENDING TASKS: {getPendingTasks(state.task_queue)}
    COMPLETED: {state.completed_tasks}
    NETWORK SNAPSHOT: {shortInfo}
    """

    if state.reasoning.selected_task_ID:
        taskID = state.reasoning.selected_task_ID
        currentTask = getTaskById(task_id=taskID, queue=state.task_queue)

    if reason.route == "new":
        prompt = f"""
        [TASK SELECTION MODE]
        
        You are a cybersecurity orchestration agent.
        
        {contextData}
        
        Choose the best next task ID from the PENDING TASKS list and give your explanation for your decision.
        
        Return valid JSON: 
        {{
            "task_ID": <int>, 
            "reasoning": "<why>"
        }}
        """
    elif reason.route == "expand":
        prompt = f"""
        [PLAN EXPANSION MODE]
        
        You are a cybersecurity orchestration agent.
        It was decided that task expansion is needed.
        
        [FEEDBACK FROM EVALUATION]
        
        Confidence: {evaluateOutput.confidence}
        
        Reasoning: {evaluateOutput.reasoning}

        [TASK]
        Based on the information listed bellow and feedback from evaluation give your guidance for expanding the number of tasks.

        {contextData}
        
        Return valid JSON:
            {{
                "task_ID": None,
                "reasoning": "STRATEGIC GUIDANCE: ..."
            }}
        """

    elif reason.route == "replan":

        prompt = f"""
        [REPLANNING MODE]
        
        You are a cybersecurity orchestration agent.
        It was decided that current task queue needs replanning.
        Analyze why we failed and suggest a new tactical direction for replanning process.
        
        [FEEDBACK FROM EVALUATION]
        
        Confidence: {evaluateOutput.confidence}
        
        Reasoning: {evaluateOutput.reasoning}
        
        Based on the information listed bellow give your additional feedback for replanning tasks.

        > Initial objective: {state.objective}
        
        > Other information:
        
        {contextData}
        
        Return JSON:
        {{
            "task_ID": None,
            "reasoning": "NEW TACTICAL DIRECTION: The previous approach failed because [REASON]. Shift focus to [NEW TARGET/METHOD]."
        }}
        """

    elif reason.route == "retry":

        prompt = f"""
        [TASK RETRY MODE]
        
        You are a cybersecurity orchestration agent.
        It was decided that task needs to be performed again.
        
        [FEEDBACK FROM EVALUATION]
        
        Confidence: {evaluateOutput.confidence}
        
        Reasoning: {evaluateOutput.reasoning}
        
        Based on the information listed bellow give your guidance for retrying current task.

        {contextData}
        
        [TOOL]
        
        > Last agent call:
        {state.agent_call}
        
        > Last agent output:
        {state.agent_output}
        
        Return JSON:
        {{
            "task_ID": {taskID},
            "reasoning": "NEW RETRY STRATEGY: The previous approach failed because [REASON]. Try to do this [NEW METHOD]."
        }}
        """

    try:
        outputFull = await llm.with_structured_output(
            reasoningOutput, include_raw=True
        ).ainvoke(prompt)
        output = outputFull["parsed"]
        outputRaw = outputFull["raw"]

        if reason.route in ["replan", "expand"] and not output.task_ID == None:
            output.task_ID = None
            logData(
                f"[REASONING NODE] -> reasoning for '{reason.route}' decision: {output.reasoning}"
            )

        else:
            logData(f"[REASONING NODE] -> selected task: {output.task_ID}")
            logData(
                f"[REASONING NODE] -> reasoning for selected task: {output.reasoning}"
            )
            logReasoning(
                agentName=AGENT_NAME,
                reasoning=f"[REASONING] Task {output.task_ID} was selected.\nReasoning: {output.reasoning}",
            )
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
        state.reasoning.route = "stop"
        return {
            "reasoning": state.reasoning,
        }

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

    existing_descriptions = [t.description for t in state.task_queue.queue]
    completed_desc = [t.description for t in (state.completed_tasks or [])]

    host_summary = [
        f"IP: {ip}, Ports: {[p.port for p in h.open_ports]}"
        for ip, h in state.host_memory.items()
    ]

    prompt = f"""
    [PLAN EXPANSION]
    Objective: {state.objective}
    
    STRATEGIC GUIDANCE FROM REASONER:
    {state.reasoning.reasoning}

    CURRENT NETWORK STATE:
    {host_summary}

    ALREADY PLANNED OR FINISHED (DO NOT REPEAT):
    {existing_descriptions + completed_desc}

    TASK:
    Generate 1 to 5 NEW specific tasks based on the STRATEGIC GUIDANCE and CURRENT NETWORK STATE.
    
    Each task must have:
    - agent: strictly one of [nmap, gobuster, sqlmap]
    - description: a clear, short command-like description for agent call builder.
    
    CURRENTLY EXPLOITING IS ONLY DONE WITH SQLMAP

    Return JSON:
    - id (choose unique identifiers for tasks in ascending order)
    - agent (suggest agent and explain your decision in "description field")
    - description 
    """

    try:
        outputFull = await llm.with_structured_output(
            expansionOutput, include_raw=True
        ).ainvoke(prompt)

        output = outputFull["parsed"]
        outputRaw = outputFull["raw"]

        logMetadata(agent_name=AGENT_NAME, metadata=outputRaw.response_metadata)

        nextID = getNextTaskID(state.task_queue)

        existingDesc = {t.description for t in state.task_queue.queue}

        for task in output.new_tasks:
            if task.description not in existingDesc:
                task.id = nextID
                nextID += 1
                state.task_queue.queue.append(task)
        logData(f"[PLAN EXPANSION NODE] -> new generated tasks: {output.new_tasks}")
        logData(f"[PLAN EXPANSION NODE] -> added {len(output.new_tasks)} tasks")
        logData(
            f"[PLAN EXPANSION NODE] -> final task queue after changes: {state.task_queue.queue}"
        )

        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLAN EXPANSION] New generated tasks:\n{json.dumps(output.model_dump(), indent=4)}",
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLAN EXPANSION] Updated queue after expansion:\n{json.dumps(state.task_queue.queue, indent=4),}",
        )

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
    host_summary = {
        ip: [p.port for p in h.open_ports] for ip, h in state.host_memory.items()
    }
    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f"[AGENT EXECUTION] Preparing agent call for task {task.id}",
    )

    prompt = f"""
    You are a cybersecurity tool agent. Create a specific call for a tool based on the task.
    
    [SELECTED TASK]
    ID: {taskID}
    Description: {task.description}
    Reasoning: {state.reasoning.reasoning}
    
    [KNOWN HOSTS & PORTS]
    {host_summary}

    [INSTRUCTIONS]
    1. Select the correct agent (nmap, gobuster, sqlmap).
    2. Identify the SPECIFIC IP or URL from the known hosts that needs to be targeted.
    3. Write a short prompt for that agent.

    Return valid JSON:
    - agent (EXACTLY one of [nmap, gobuster, sqlmap])
    - agent_prompt (short prompt for agent - instructions what to do / achieve)
    - target host (SPECIFIC IP or URL from the known hosts that needs to be targeted)
    """

    outputFull = await llm.with_structured_output(
        executorOutput, include_raw=True
    ).ainvoke(prompt)

    output = outputFull["parsed"]
    outputRaw = outputFull["raw"]

    logMetadata(agent_name=AGENT_NAME, metadata=outputRaw.response_metadata)

    logData(message=f"[AGENT EXECTUION NODE] -> created an agent call: {output}")
    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f"[AGENT EXECUTION] Created following for {output.agent}: {output.agent_prompt}",
    )

    state.agent_call.agent = output.agent
    state.agent_call.task_ID = taskID

    state.agent_call.agent_input.prompt = output.agent_prompt
    state.agent_call.execution_status = "pending"
    state.agent_call.agent_input.targets = output.target_host

    logData(message="[AGENT EXECUTION NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
    }


async def inputBuilderNode(state: orchestratorState):
    logData(message="[INPUT BUILDER] -> enter node")

    agent = state.agent_call.agent

    if agent == "nmap" and not state.agent_call.agent_input.targets:
        # 1. check if a valid target is mentioned in given instructions

        prompt = state.agent_call.agent_input.prompt
        targets = extractNmapTargets(prompt=prompt)

        if targets:
            logData(
                f"[INPUT BUILDER] -> found a valid target [{targets}] for {agent} inside agent prompt"
            )
            state.agent_call.agent_input.targets = [targets]

            logData("[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }

        logData(
            f"[INPUT BUILDER] -> no valid targets found for {agent} inside prompt, searching targets in agent state..."
        )

        # 2. search for backup targets in agent state
        savedTargets = state.discovered_hosts

        if savedTargets:
            logData(
                f"[INPUT BUILDER] -> found backup targets in agent state -> sending all forward: {savedTargets}"
            )
            state.agent_call.agent_input.additional_data = savedTargets

            logData("[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }
        else:
            logData("[INPUT BUILDER] -> no valid targets found -> skipping agent call")
            state.agent_call.execution_status = "skipped"
            state.agent_call.message = f"No valid hosts available for {agent} agent."
            logData("[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }

    elif agent == "gobuster" and not state.agent_call.agent_input.targets:
        # 1. check if a valid target is mentioned in given instructions
        prompt = state.agent_call.agent_input.prompt
        targets = extractGobusterTargets(prompt=prompt)

        if targets:
            target_list = [targets] if isinstance(targets, str) else targets
            sanitized_targets = []

            for t in target_list:
                if not t.startswith(("http://", "https://")):
                    logData(
                        f"[INPUT BUILDER] -> target {t} lacks protocol, prepending http://"
                    )
                    sanitized_targets.append(f"http://{t}")
                    sanitized_targets.append(f"https://{t}")
                else:
                    sanitized_targets.append(t)

            logData(
                f"[INPUT BUILDER] -> found a valid target {sanitized_targets} for {agent} inside agent prompt"
            )
            state.agent_call.agent_input.targets = list(set(sanitized_targets))

            logData("[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }

        # 2. try to build targets from agent state
        savedTargets = []
        for ip, host in state.host_memory.items():
            for port in host.open_ports:
                if port.port in [80, 443, 8080, 8443, 8843]:
                    proto = "https" if port.port in [443, 8443] else "http"
                    savedTargets.append(f"{proto}://{ip}:{port.port}")

        if savedTargets:
            state.agent_call.agent_input.targets = savedTargets

            logData(
                f"[INPUT BUILDER] -> built backup targets from agent state -> sending all forward: {savedTargets}"
            )

            logData("[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }
        else:
            logData("[INPUT BUILDER] -> no valid targets found -> skipping agent call")
            state.agent_call.execution_status = "skipped"
            state.agent_call.message = (
                f"No valid targets are available for {agent} agent."
            )
            logData("[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }
    elif agent == "sqlmap":

        # 1. lookup for valid untested attack vectors
        vectors = []

        for vector in state.attack_vectors:
            if not vector.tested:
                vectors.append(vector)

        if vectors:
            logData(
                f"[INPUT BUILDER] -> successfuly prepared {len(vectors)} untested vectors"
            )
            logData(f"[INPUT BUILDER] -> input successfuly built for {agent} agent")
            state.agent_call.agent_input.attack_vectors = vectors

            logData(f"[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }
        else:
            logData(
                f"[INPUT BUILDER] -> no valid untested vectors found for {agent} agent -> skipping agent call"
            )
            state.agent_call.execution_status = "skipped"
            state.agent_call.message = f"No valid untested vectors for {agent} agent."

            logData(f"[INPUT BUILDER] -> exit node")
            return {
                "agent_call": state.agent_call,
            }


async def nmapAgentNode(state: orchestratorState):
    logData("[NMAP NODE] -> enter node")

    agentCall = state.agent_call
    agent_ID = agentCall.task_ID

    state.agent_counter.last_agent = "nmap"
    state.agent_counter.last_agent_iteration += 1

    for retries in range(2):

        if agentCall.agent_input.targets and isinstance(
            agentCall.agent_input.targets[0], list
        ):
            agentCall.agent_input.targets = agentCall.agent_input.targets[0]

        try:
            logData(
                f"[NMAP NODE] -> calling nmap agent with following target: {agentCall.agent_input.targets}"
            )
            logData(
                f"[NMAP NODE] -> calling nmap agent with following prompt: {agentCall.agent_input.prompt}"
            )
            logData(
                "================================ NMAP AGENT start ================================"
            )
            logReasoning(agentName=AGENT_NAME, reasoning="NMAP AGENT start")
            result = await NMAP_GRAPH.ainvoke(
                {
                    "objective": agentCall.agent_input.prompt,
                    "target": agentCall.agent_input.targets,
                },
                {"recursion_limit": 1000},
            )
            logData(
                "================================ NMAP AGENT stop ================================"
            )
            logReasoning(agentName=AGENT_NAME, reasoning="NMAP AGENT stop")

            agent_output = result.get("agent_output")
            logData(f"\n[NMAP NODE] -> Raw full output:\n\n{result}")
            logData(f"\n[NMAP NODE] -> Raw agent output:\n\n{agent_output}")

            state.agent_output = agent_output
            state.agent_outputs_history[agent_ID] = agent_output

            if agent_output.success:
                agentCall.execution_status = "executed"
                logData(f"[NMAP NODE] -> agent call successful")

            else:
                agentCall.execution_status = "failed"
                logData(f"[NMAP NODE] -> agent call failed")

            logData("[NMAP AGENT] -> exit node")
            return {
                "agent_output": state.agent_output,
                "agent_outputs_history": state.agent_outputs_history,
                "agent_call": state.agent_call,
                "agent_counter": state.agent_counter,
            }

        except Exception as e:
            logData(
                f"[NMAP NODE] -> agent call produced the following exception: {str(e)}"
            )
            agentCall.message = (
                f"Failed to call {agentCall.agent} within {retries} retries. Last call produced the following error:\n\n"
                + str(e)
            )
            agentCall.execution_status = "failed"

    logData("[NMAP NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
        "agent_counter": state.agent_counter,
    }


async def gobusterAgentNode(state: orchestratorState):
    logData(message="[GOBUSTER NODE] -> enter node")
    agentCall = state.agent_call
    agentID = agentCall.task_ID

    state.agent_counter.last_agent = "gobuster"
    state.agent_counter.last_agent_iteration += 1

    for retries in range(2):

        # flatten targets just in case
        if agentCall.agent_input.targets and isinstance(
            agentCall.agent_input.targets[0], list
        ):
            agentCall.agent_input.targets = agentCall.agent_input.targets[0]

        try:

            logData(
                "================================ GOBUSTER AGENT start ================================"
            )
            logReasoning(agentName=AGENT_NAME, reasoning="GOBUSTER AGENT start")

            logData(f"[GOBUSTER NODE] -> target: {agentCall.agent_input.targets}")

            result = await GOBUSTER_GRAPH.ainvoke(
                {
                    "objective": agentCall.agent_input.prompt,
                    "target": agentCall.agent_input.targets,
                },
                {"recursion_limit": 1000},
            )

            logData(
                "================================ GOBUSTER AGENT stop ================================"
            )
            logReasoning(agentName=AGENT_NAME, reasoning="GOBUSTER AGENT stop")

            agent_output = result.get("agent_output")

            state.agent_output = agent_output
            state.agent_outputs_history[agentID] = agent_output

            if agent_output.success:
                logData("[GOBUSTER NODE] -> agent call successful")
                agentCall.execution_status = "executed"

            else:
                logData("[GOBUSTER NODE] -> agent call failed")
                agentCall.execution_status = "failed"

            logData("[GOBUSTER NODE] -> exit node")
            return {
                "agent_call": state.agent_call,
                "agent_output": state.agent_output,
                "agent_outputs_history": state.agent_outputs_history,
                "agent_counter": state.agent_counter,
            }

        except Exception as e:

            logData(
                f"[GOBUSTER NODE] -> agent call produced the following exception: {str(e)}"
            )

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
        "agent_counter": state.agent_counter,
    }


async def sqlmapAgentNode(state: orchestratorState):
    logData("[SQLMAP NODE] -> enter node")

    agentCall = state.agent_call
    agent_ID = agentCall.task_ID

    state.agent_counter.last_agent = "sqlmap"
    state.agent_counter.last_agent_iteration += 1

    for retries in range(2):
        try:

            logData(
                "================================ SQLMAP AGENT start ================================"
            )
            logReasoning(agentName=AGENT_NAME, reasoning="SQLMAP AGENT start")

            result = await SQLMAP_GRAPH.ainvoke(
                {
                    "objective": agentCall.agent_input.prompt,
                    "attack_vectors": agentCall.agent_input.attack_vectors,
                },
                {"recursion_limit": 1000},
            )

            logData(
                "================================ SQLMAP AGENT stop ================================"
            )
            logReasoning(agentName=AGENT_NAME, reasoning="SQLMAP AGENT stop")

            agent_output = result.get("agent_output")

            state.agent_output = agent_output
            state.agent_outputs_history[agent_ID] = agent_output

            if agent_output.success:
                logData(message="[SQLMAP NODE] -> agent call successful")
                agentCall.execution_status = "executed"
            else:
                logData(message="[SQLMAP NODE] -> agent call failed")
                agentCall.execution_status = "failed"

            logData("[SQLMAP NODE] -> exit node")
            return {
                "agent_call": state.agent_call,
                "agent_output": state.agent_output,
                "agent_outputs_history": state.agent_outputs_history,
                "agent_counter": state.agent_counter,
            }

        except Exception as e:

            logData(
                f"[SQLMAP NODE] -> agent call produced the following exception: {str(e)}"
            )

            agentCall.message = (
                f"Failed to call {agentCall.agent} within {retries} retries. Last call produced the following error:\n\n"
                + str(e)
            )
            agentCall.execution_status = "failed"

    logData(message="[SQLMAP NODE] -> exit node")
    return {
        "agent_call": state.agent_call,
        "agent_counter": state.agent_counter,
    }


async def evaluateNode(state: orchestratorState):
    logData(message="[EVALUATE NODE] -> enter node")
    # 3 types of prompts based on the execution status: failed, skipped, executed
    agentCall = state.agent_call
    taskID = state.reasoning.selected_task_ID
    task = getTaskById(task_id=taskID, queue=state.task_queue)

    currentKnowledge = {
        "hosts": state.host_memory,
        "attack_vectors": state.attack_vectors,
        "vulnerabilities": state.vulnerabilities,
    }

    interpretationRules = """
    [CRITICAL RULES FOR DECISION]
    1. If the tool found NEW information (ports, files, vectors) -> select 'expand'.
    2. If the tool found NOTHING but executed correctly -> select 'replan' (we need a new target or tool).
    3. If the tool failed due to a timeout or network error -> select 'retry'.
    4. If the task is finished and no further action is needed for THIS specific task -> select 'success'.
    """

    if agentCall.execution_status == "executed":
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

    mainPrompt = f"""
        [ROLE]
        You are a cybersecurity evaluation agent.
        
        [YOUR TASK]
        Your job is to evaluate the result of the last tool agent execution
        and decide what the orchestrator should do next.
        
        [OBJECTIVE]
        {state.objective}
        
        [LAST TASK]
        {task.description}
        
        {situationalPrompt}
        
        [GLOBAL KNOWLEDGE]
        
        {currentKnowledge}
        
        {interpretationRules}
        
        Return JSON:
        - reasoning: (Short explanation for this decision)
        - decision: (success, retry, expand, replan)
        - confidence: (0.0 - 1.0)
        """

    result = None

    try:
        resultFull = await llm.with_structured_output(
            evaluateNodeOutput, include_raw=True
        ).ainvoke(mainPrompt)

        result = resultFull["parsed"]
        resultRaw = resultFull["raw"]

        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[EVALUATE] {result.reasoning}",
        )
        logMetadata(agent_name=AGENT_NAME, metadata=resultRaw.response_metadata)

        state.evaluate = result

    except Exception as e:
        logData(f"[EVALUATE NODE] -> LLM failure: {str(e)}")

    if result.decision in ["success", "expand"]:
        state.task_queue, finishedTask = completeTask(
            queue=state.task_queue, task_id=taskID
        )

        if finishedTask:

            if state.completed_tasks is None:
                state.completed_tasks = []

            logData(f"[EVALUATE NODE] -> task {taskID} marked as COMPLETED")
            logReasoning(
                agentName=AGENT_NAME,
                reasoning=f"[EVALUATE] Task {taskID} marked as COMPLETED.",
            )
            state.completed_tasks.append(finishedTask)

    if result.decision == "retry":
        task.retry_count += 1

        if task.retry_count >= task.max_retries:
            state.evaluate.decision = "replan"
            logData(
                f"[EVALUATE NODE] -> max retires reached for task {taskID} -> forcing REPLAN"
            )
            logReasoning(
                agentName=AGENT_NAME,
                reasoning=f"[EVALUATE] Max retires reached for task {taskID}, forcing REPLAN.",
            )

        else:
            logData(f"[EVALUATE NODE] -> task {taskID} retry count: {task.retry_count}")

    state.agent_call.execution_status = "pending"
    state.agent_call.message = None

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


async def saveToolOutputNode(state: orchestratorState):

    agent_output = state.agent_output
    # ---------------------- nmap ---------------------- #
    if agent_output.agent_name == "nmap" and agent_output.success:
        state.discovered_hosts = list(
            set(state.discovered_hosts + agent_output.discovered_hosts)
        )
        for ip, host in agent_output.host_memory.items():
            state.host_memory[ip] = HostMemory(**host.model_dump())

    # -------------------- gobuster -------------------- #
    elif agent_output.agent_name == "gobuster" and agent_output.success:
        existing = {(v.endpoint, v.method) for v in state.attack_vectors}

        for vec in agent_output.attack_vectors:
            vec = attackVector(**vec.model_dump())
            key = (vec.endpoint, vec.method)

            if key not in existing:
                state.attack_vectors.append(vec)

        for host, data in agent_output.host_enum.items():
            if host not in state.host_enum:
                state.host_enum[host] = data
            else:
                existing_endpoints = state.host_enum[host].get("endpoints", [])
                new_endpoints = data.get("endpoints", [])
                seen_combinations = {
                    (ep.get("base_url"), ep.get("path")) for ep in existing_endpoints
                }

                for ep in new_endpoints:
                    combo = (ep.get("base_url"), ep.get("path"))
                    if combo not in seen_combinations:
                        existing_endpoints.append(ep)
                        seen_combinations.add(combo)

                state.host_enum[host]["endpoints"] = existing_endpoints

    # --------------------- sqlmap --------------------- #
    elif agent_output.agent_name == "sqlmap" and agent_output.success:
        vectors = agentCall.agent_input.additional_data or []

        for vector in vectors:
            for stateVector in state.attack_vectors:
                if (
                    stateVector.endpoint == vector.endpoint
                    and stateVector.method == vector.method
                    and stateVector.parameters == vector.parameters
                ):
                    stateVector.tested = True

        existing = {
            (v.host, v.url, tuple(v.parameters), v.vulner_type)
            for v in state.vulnerabilities
        }

        for vuln in agent_output.vulnerabilities:
            vuln = vulnerability(**vuln.model_dump())

            key = (
                vuln.host,
                vuln.url,
                tuple(vuln.parameters),
                vuln.vulner_type,
            )

            if key not in existing:
                state.vulnerabilities.append(vuln)

    agent_output = None

    return {
        "agent_output": state.agent_output,
        "discovered_hosts": state.discovered_hosts,
        "host_memory": state.host_memory,
        "attack_vectors": state.attack_vectors,
        "vulnerabilities": state.vulnerabilities,
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
        
        NO MARKDOWN, NO EMOJIS!
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
        
        NO MARKDOWN, NO EMOJIS!
        """

    response = await llm.ainvoke(prompt)

    logMetadata(agent_name=AGENT_NAME, metadata=response.response_metadata)

    state.output_summary = response.content
    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f" {20 * "="} ORCHESTRATOR SUMMARY {20 * "="} ",
    )
    logReasoning(agentName=AGENT_NAME, reasoning=f"[SUMMARY] {response.content}")

    return {
        "output_summary": state.output_summary,
    }


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #


def extractNmapTargets(prompt: str):

    cidr = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}/\d{1,2}\b", prompt)
    ips = re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", prompt)
    cidr_ips = {c.split("/")[0] for c in cidr}
    filtered_ips = [ip for ip in ips if ip not in cidr_ips]
    targets = list(set(cidr + filtered_ips))

    return targets


def extractGobusterTargets(prompt: str):
    urls = re.findall(r"https?://[^\s]+", prompt)

    if urls:
        return list(set(urls))

    # fallback → IP:port
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b", prompt)

    return [f"http://{ip}" for ip in ips]


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
    workflow.add_node("router_node", routerNode)
    workflow.add_node("reasoning_node", reasoningNode)
    workflow.add_node("plan_expansion_node", planExpansionNode)
    workflow.add_node("agent_execution_node", agentExecutionNode)
    workflow.add_node("input_builder_node", inputBuilderNode)
    workflow.add_node("nmap_agent_node", nmapAgentNode)
    workflow.add_node("gobuster_agent_node", gobusterAgentNode)
    workflow.add_node("sqlmap_agent_node", sqlmapAgentNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("save_tool_output_node", saveToolOutputNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------

    workflow.add_edge(START, "planning_node")
    workflow.add_edge("planning_node", "router_node")
    workflow.add_edge("router_node", "reasoning_node")
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
    workflow.add_edge("plan_expansion_node", "router_node")
    workflow.add_edge("agent_execution_node", "input_builder_node")
    workflow.add_conditional_edges(
        "input_builder_node",
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
    workflow.add_edge("evaluate_node", "save_tool_output_node")
    workflow.add_edge("save_tool_output_node", "router_node")
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
    5. Confirm and exploit any SQL injection vulnerabilities.

    Summarize all discovered attack vectors and vulnerabilities.
    """

    asyncio.run(orchestratorRunner(prompt=testPrompt))
