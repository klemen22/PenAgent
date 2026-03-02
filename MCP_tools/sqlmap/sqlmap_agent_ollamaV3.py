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

load_dotenv()

from MCP_tools.sqlmap.sqlmap_tool import sqlmap_scan, sqlmapConfig

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

finalAgent = llm.bind_tools([sqlmap_scan])


# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


# ---------------- Plan ---------------- #
class agentPlanStep(BaseModel):
    description: str = Field(default="")
    method: Literal["GET", "POST"]
    params: List[str] = Field(default_factory=list)
    phase: Literal["detection", "exploitation"]

    model_config = ConfigDict(extra="forbid")


class agentPlanOutput(BaseModel):
    reasoning: str = Field(default="")
    steps: List[agentPlanStep] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class sqlmapToolSelection(BaseModel):
    url: str
    method: str
    data: Optional[str]

    level: Optional[int]
    risk: Optional[int]
    technique: Optional[str]
    threads: Optional[int]

    random_agent: Optional[bool]

    enumeration: Optional[List[str]]
    tamper: Optional[List[str]]

    reasoning: str
    confidence: float


# ---------------- Analysis ---------------- #
class agentFeedback(BaseModel):
    vulnerability_found: bool = Field(default=False)
    exploitation_possible: bool = Field(default=False)
    confidence: float = Field(default=0.0)
    reasoning: str = Field(default="")


# ---------------- State ---------------- #
class attackVectorMemory(BaseModel):
    vector_data: Dict[str, Any] = Field(
        default_factory=dict, description="Selected vector from the list."
    )
    plan: List[agentPlanStep] = Field(
        default=[], description="Formulated plan for the given attack vector."
    )
    step_index: int = Field(default=0)

    selected_command: Optional[sqlmapToolSelection] = Field(default=None)
    last_tool_result: Optional[Any] = Field(default=None)

    analysis: Optional[agentFeedback] = Field(
        default=None, description="Tool output analysis."
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence factor for deciding if replanning is needed.",
    )
    done: bool = Field(default=False)


class sqlmapAgentState(BaseModel):
    objective: str = Field(default="", description="Current agent objective.")

    attack_vectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Attack vectors list given by the orchestrator.",
    )

    vector_index: int = Field(default=0)

    vectors_memory: Dict[str, attackVectorMemory] = Field(default_factory=dict)

    iteration: int = Field(default=0)
    max_iteration: int = Field(default=50)

    decision: Optional[str] = Field(default=None)

    # logging
    agent_log: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #


async def planningNode(state: sqlmapAgentState):

    if state.vector_index >= len(state.attack_vectors):
        return {"decision": "stop"}

    currentMemory = getCurrentVector(state=state)

    if currentMemory.plan and state.decision != "replan":
        return {"decision": "continue"}

    print("\n[PLANNING]\n")

    prompt = f"""
    You are an autonomous SQL injection agent.

    Objective:
    {state.objective}

    Attack vector:
    Endpoint: {currentMemory.vector_data['endpoint']}
    Method: {currentMemory.vector_data['method']}
    Parameters: {currentMemory.vector_data['params']}

    STRICT RULES:
    1. Do NOT construct URLs.
    2. Do NOT include parameter values.
    3. Only specify parameter NAMES to test.
    4. Do NOT include SQL payloads.
    5. SQLMap performs injection automatically.
    
    IMPORTANT:
    You are working on ONLY this single endpoint.
    Do NOT consider any other endpoints.
    Do NOT reference other attack vectors.

    Return JSON:
    - reasoning
    - steps: list of
        - description
        - target_url
        - method
        - params
        - phase
    
    Return ONLY valid JSON matching the schema.
    """

    retries = 2

    for attempt in range(retries):

        try:
            outputPlan = await llm.with_structured_output(agentPlanOutput).ainvoke(
                prompt
            )

            log_data(state, "[PLANNING]")
            log_data(state, outputPlan.reasoning)

            currentMemory.plan = outputPlan.steps
            currentMemory.step_index = 0

            return {"vectors_memory": state.vectors_memory, "decision": "continue"}

        except Exception as e:

            print(f"[PLANNING ERROR] Attempt {attempt+1}")
            print("Raw LLM output:")
            print(e.llm_output)

            log_data(state, f"Planning JSON parse failed attempt {attempt+1}")

            prompt += "\n\nWARNING: Your previous output was not valid JSON. Return ONLY raw JSON. No explanations."

    log_data(state, "Planning failed after retries.")

    currentMemory.plan = []
    currentMemory.step_index = 0

    return {"vectors_memory": state.vectors_memory, "decision": "stop"}


async def selectActionNode(state: sqlmapAgentState):

    with open("MCP_tools\sqlmap\sqlmap_allowed_arguments.json") as f:
        allowedArguments = f.read()

    currentMemory = getCurrentVector(state=state)
    vector = currentMemory.vector_data

    print("\n[SELECT_ACTION]\n")

    if currentMemory.step_index >= len(currentMemory.plan):
        return {"decision": "replan"}

    step = currentMemory.plan[currentMemory.step_index]

    prompt = f"""
    Objective: {state.objective}

    Current attack vector:
    Endpoint: {currentMemory.vector_data['endpoint']}
    Method: {currentMemory.vector_data['method']}
    Parameters: {currentMemory.vector_data['params']}

    Current plan step:
    Description: {step.description}
    Phase: {step.phase}
    Parameters under test: {step.params}

    Previous analysis:
    {currentMemory.analysis.reasoning if currentMemory.analysis else "None"}

    Available sqlmap options:
    {allowedArguments}
    
    IMPORTANT:
    Do NOT modify parameter values.
    Do NOT inject payloads manually.
    Always return clean URL with normal parameter values.
    sqlmap will handle injection automatically.

    Decide:
    - which options are appropriate
    - do NOT escalate risk without reason
    - prefer minimal detection first
    - explain reasoning
    
    Rules:
    - Start with safe detection.
    - Escalate only if confidence < 0.5.
    - High risk exploitation allowed only if vulnerability confirmed.
    """

    selection = await llm.with_structured_output(sqlmapToolSelection).ainvoke(prompt)

    # print("\n[SELECT REASONING]")
    # print(selection.reasoning)

    log_data(state, message="\n[SELECT REASONING]")
    log_data(state, selection.reasoning)

    currentMemory.selected_command = selection

    return {
        "vectors_memory": state.vectors_memory,
        "decision": None,
    }


async def toolExecutionNode(state: sqlmapAgentState):
    print("\n[EXECUTE TOOL]\n")

    currentMemory = getCurrentVector(state=state)

    if not currentMemory.selected_command:
        return {"decision": "replan"}

    selection = currentMemory.selected_command

    config_payload = {
        "level": selection.level or 1,
        "risk": selection.risk or 1,
        "batch": True,
        "random_agent": selection.random_agent or False,
        "current_db": False,
        "enumerate_tables": False,
        "tamper": selection.tamper,
    }

    # enumeration mapping
    if selection.enumeration:
        if "current_db" in selection.enumeration:
            config_payload["current_db"] = True
        if "tables" in selection.enumeration:
            config_payload["enumerate_tables"] = True

    step = currentMemory.plan[currentMemory.step_index]
    vector = currentMemory.vector_data

    base_url = vector["endpoint"]

    clean_params = []

    for param in step.params:
        clean_params.append(f"{param}=1")

    if clean_params:
        clean_url = f"{base_url}?{'&'.join(clean_params)}"
    else:
        clean_url = base_url

    tool_payload = {
        "url": clean_url,
        "data": selection.data or "",
        "config": config_payload,
    }

    if selection.method != "POST":
        tool_payload["data"] = ""

    print(f"\n[TOOL PAYLOAD]\n{tool_payload}")
    log_data(state, "\n[TOOL PAYLOAD]")
    log_data(state, str(tool_payload))

    try:
        rawOutput = await sqlmap_scan(
            url=tool_payload["url"],
            data=tool_payload["data"],
            config=sqlmapConfig(**tool_payload["config"]),
        )
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }

    print(f"\n[LAST RAW TOOL RESULT]\n{rawOutput}")

    currentMemory.last_tool_result = rawOutput

    return {"vectors_memory": state.vectors_memory}


async def analyzeNode(state: sqlmapAgentState):
    print("\n[ANALYZE]\n")

    # print(f"\n[LAST TOOL RESULT]\n{state.last_tool_result}")

    currentMemory = getCurrentVector(state=state)
    toolResult = currentMemory.last_tool_result

    if not toolResult:
        log_data(state, "No tool result -> skip analysis")

        currentMemory.analysis = agentFeedback(
            vulnerability_found=False,
            exploitation_possible=False,
            confidence=0.0,
            reasoning="No tool result available.",
        )

        currentMemory.confidence = 0.0

        return {"vectors_memory": state.vectors_memory}

    prompt = f"""
    Objective:
    {state.objective}

    Tool result:
    {toolResult}
    
    Vector context:
    Endpoint: {currentMemory.vector_data['endpoint']}
    Method: {currentMemory.vector_data['method']}
    Parameter under test: {currentMemory.plan[currentMemory.step_index].params}

    You MUST analyze ONLY what is explicitly present in the tool result.

    If the tool result does not explicitly mention:
    - injectable
    - SQL injection
    - parameter is vulnerable
    - dbms fingerprint

    Then you MUST return:
    vulnerability_found = false
    confidence = 0.0
    reasoning = "No explicit SQL injection evidence found."

    Return JSON:
    - vulnerability_found
    - exploitation_possible
    - confidence (0-1)
    - reasoning
    """

    # response = await llm.ainvoke(prompt)
    # feedback = agentFeedback.model_validate_json(response.content)

    feedback = await llm.with_structured_output(agentFeedback).ainvoke(prompt)
    # print(f"Feedback:\n {feedback.reasoning}")
    # print(f"Confidence: {feedback.confidence}")

    log_data(state=state, message="\n[ANALYZE]")
    log_data(state=state, message=feedback.reasoning)
    log_data(state=state, message=f"Confidence: {feedback.confidence}")

    currentMemory.analysis = feedback
    currentMemory.confidence = feedback.confidence

    return {"vectors_memory": state.vectors_memory}


async def evaluateNode(state: sqlmapAgentState):
    print("\n[EVALUATE]\n")

    log_data(state=state, message="\n[EVALUATE]")

    currentMemory = getCurrentVector(state=state)
    newIteration = state.iteration + 1

    # plan exhausted
    if currentMemory.step_index + 1 >= len(currentMemory.plan):
        currentMemory.done = True
        state.vector_index += 1
        return {
            "vector_index": state.vector_index,
            "vectors_memory": state.vectors_memory,
            "decision": "plan",
            "iteration": newIteration,
        }

    # vulnerability was found
    if currentMemory.analysis and currentMemory.analysis.vulnerability_found:
        currentMemory.step_index += 1
        return {"decision": "continue", "vectors_memory": state.vectors_memory}

    # if confidence is too low
    if currentMemory.confidence < 0.3:
        return {"decision": "replan"}

    currentMemory.step_index += 1
    return {
        "vectors_memory": state.vectors_memory,
        "decision": "continue",
    }


def outputNode(state: sqlmapAgentState):
    print(
        json.dumps(
            {k: v.model_dump() for k, v in state.vectors_memory.items()}, indent=4
        )
    )


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #
def sqlmapOutputParser(toolOutput):

    stdout = ""
    keywords = [
        "parameter",
        "injectable",
        "dbms",
        "warning",
        "critical",
        "all tested parameters",
        "appears",
        "does not",
    ]

    filteredOutput = []

    if isinstance(toolOutput, tuple):
        result = toolOutput[1].get("result", {})
        stdout = result.get("stdout", "")

    elif isinstance(toolOutput, dict):
        stdout = toolOutput.get("stdout", "")

    if stdout != "":
        lines = stdout.splitlines()

        for line in lines:
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    filteredOutput.append(line)

        return "\n".join(filteredOutput)
    else:
        return ""


def evaluateNodeRouting(state: sqlmapAgentState):

    if state.decision == "continue":
        return "select_action"

    elif state.decision == "replan":
        return "plan"

    elif state.decision == "stop":
        return "output"

    return "output"


def setupLogger():
    logDir = Path("MCP_tools/sqlmap/logs")
    logFile = logDir / f"sqlmap_agent_log4.log"
    logger = logging.getLogger("sqlmap_agent")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def log_data(state, message: str):
    state.agent_log.append(message)
    logging.getLogger("sqlmap_agent").info(message)


def getCurrentVector(state: sqlmapAgentState) -> attackVectorMemory:
    if state.vector_index >= len(state.attack_vectors):
        return None

    vector = state.attack_vectors[state.vector_index]

    key = f"{vector['endpoint']}::{vector['method']}"

    return state.vectors_memory.get(key)


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


async def agentRunner(endpoints):
    agentState = sqlmapAgentState()
    logger = setupLogger()

    workflow = StateGraph(sqlmapAgentState)
    SESSION_ID = "default_session"

    # test initial prompt
    agentState.objective = "Analyze given attack vectors."

    # prepare attack vectors - filter out vectors without parameters
    for vector in endpoints:
        if len(vector.get("params", [])) > 0:
            agentState.attack_vectors.append(vector)

            key = f"{vector['endpoint']}::{vector['method']}"

            agentState.vectors_memory[key] = attackVectorMemory(vector_data=vector)

            print(f"[VALID ATTACK VECTOR]:\n\n {vector}")

    # agentState.attack_vectors = endpoints

    # -------------------------------
    # graph nodes
    # -------------------------------

    workflow.add_node("planning_node", planningNode)
    workflow.add_node("select_action_node", selectActionNode)
    workflow.add_node("tool_execution_node", toolExecutionNode)
    workflow.add_node("analyze_node", analyzeNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------

    workflow.add_edge(START, "planning_node")
    workflow.add_edge("planning_node", "select_action_node")
    workflow.add_edge("select_action_node", "tool_execution_node")
    workflow.add_edge("tool_execution_node", "analyze_node")
    workflow.add_edge("analyze_node", "evaluate_node")
    workflow.add_conditional_edges(
        "evaluate_node",
        evaluateNodeRouting,
        {
            "select_action": "select_action_node",
            "plan": "planning_node",
            "output": "output_node",
        },
    )
    workflow.add_edge("output_node", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "MCP_tools\sqlmap\sqlmap_agent_graphV3.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    await graph.ainvoke(
        agentState.model_dump(),
        config={"thread_id": SESSION_ID, "recursion_limit": 1000},
    )


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "SQLmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    with open("MCP_tools\gobuster\crawler_test_dump3.json", "r") as f:
        endpoints = json.load(f)

    result = asyncio.run(agentRunner(endpoints=endpoints))
