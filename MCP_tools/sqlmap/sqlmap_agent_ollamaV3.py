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

from MCP_tools.sqlmap.sqlmap_tool import sqlmap_scan

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
    target_url: str = Field(default="")
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
class sqlmapAgentState(BaseModel):
    objective: str = Field(default="", description="Current agent objective.")

    attack_vectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Attack vectors list given by the orchestrator.",
    )

    plan: List[agentPlanStep] = Field(
        default=[], description="Current formulated plan step by step."
    )
    step_index: int = Field(default=0, description="Current step in formulated plan.")

    selected_command: Optional[sqlmapToolSelection] = Field(default=None)

    last_tool_result: Optional[Any] = Field(
        default=None, description="Summary of last tool result."
    )
    analysis: Optional[agentFeedback] = Field(
        default=None, description="Analysis of last tool output."
    )

    confidence: float = Field(default=0.0, description="Confidence factor.")
    iteration: int = Field(default=0)
    max_iteration: int = Field(default=50)

    decision: Optional[str] = Field(default=None)

    # logging
    agent_log: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #


async def planningNode(state: sqlmapAgentState):

    print("\n[PLANNING]\n")

    prompt = f"""
    You are an autonomous SQL injection agent.

    Objective:
    {state.objective}

    Attack vectors:
    {state.attack_vectors}

    STRICT RULES:
    1. You MUST create exactly ONE detection step per attack vector that has parameters.
    2. Every attack vector with parameters must appear in the plan.
    3. Do NOT skip endpoints.
    4. Only create exploitation steps AFTER all detection steps.
    5. Detection phase must come first for ALL endpoints.
    6. Phase must be either: "detection" or "exploitation".

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

            return {
                "plan": outputPlan.steps,
                "step_index": 0,
                "decision": None,
                "agent_log": state.agent_log,
            }

        except Exception as e:

            print(f"[PLANNING ERROR] Attempt {attempt+1}")
            print("Raw LLM output:")
            print(e.llm_output)

            log_data(state, f"Planning JSON parse failed attempt {attempt+1}")

            prompt += "\n\nWARNING: Your previous output was not valid JSON. Return ONLY raw JSON. No explanations."

    log_data(state, "Planning failed after retries.")
    return {
        "plan": [],
        "decision": "stop",
        "agent_log": state.agent_log,
    }


async def selectActionNode(state: sqlmapAgentState):

    with open("MCP_tools\sqlmap\sqlmap_allowed_arguments.json") as f:
        allowedArguments = f.read()

    print("\n[SELECT_ACTION]\n")

    if state.step_index >= len(state.plan):
        print("Plan exhausted -> replan")
        return {"decision": "replan"}

    step = state.plan[state.step_index]

    prompt = f"""
    Objective: {state.objective}

    Current step:
    {step.description}

    Attack vectors:
    {state.attack_vectors}

    Available sqlmap options:
    {allowedArguments}

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

    return {
        "selected_command": selection,
        "decision": None,
        "agent_log": state.agent_log,
    }


async def toolExecutionNode(state: sqlmapAgentState):
    print("\n[EXECUTE TOOL]\n")

    if not state.selected_command:
        return {"decision": "replan"}

    selection = state.selected_command
    additional_args = []

    if selection.level:
        additional_args.append(f"--level={selection.level}")

    if selection.risk:
        additional_args.append(f"--risk={selection.risk}")

    if selection.technique:
        additional_args.append(f"--technique={selection.technique}")

    if selection.threads:
        additional_args.append(f"--threads={selection.threads}")

    if selection.enumeration:
        for enum in selection.enumeration:
            if enum.startswith("--"):
                additional_args.append(enum)
            else:
                additional_args.append(f"--{enum}")

    if selection.random_agent:
        additional_args.append("--random-agent")

    if selection.tamper:
        additional_args.append(f"--tamper={','.join(selection.tamper)}")

    additional_args.append("--batch")

    final_args = " ".join(additional_args)

    tool_payload = {
        "url": selection.url,
        "data": selection.data or "",
        "additional_args": final_args,
    }

    if selection.method != "POST":
        tool_payload["data"] = ""

    print(f"\n[TOOL PAYLOAD]\n{tool_payload}")
    log_data(state=state, message="\n[TOOL PAYLOAD]")
    log_data(state=state, message=tool_payload)

    try:
        rawOutput = await sqlmap_scan.ainvoke(tool_payload)
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }

    print(f"\n[LAST RAW TOOL RESULT]\n{rawOutput}")

    return {"last_tool_result": rawOutput}


async def analyzeNode(state: sqlmapAgentState):
    print("\n[ANALYZE]\n")

    # print(f"\n[LAST TOOL RESULT]\n{state.last_tool_result}")

    if not state.last_tool_result:
        log_data(state, "No tool result -> skip analysis")
        return {
            "analysis": agentFeedback(
                vulnerability_found=False,
                exploitation_possible=False,
                confidence=0.0,
                reasoning="No tool result available.",
            ),
            "confidence": 0.0,
        }

    prompt = f"""
    Objective:
    {state.objective}

    Tool result:
    {state.last_tool_result}

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

    return {
        "analysis": feedback,
        "confidence": feedback.confidence,
        "agent_log": state.agent_log,
    }


async def evaluateNode(state: sqlmapAgentState):
    print("\n[EVALUATE]\n")

    log_data(state=state, message="\n[EVALUATE]")

    newIteration = state.iteration + 1

    if newIteration >= state.max_iteration:
        # print("STOP -> max iterations reached")
        log_data(state=state, message="STOP -> max iterations reached")
        return {"decision": "stop", "iteration": newIteration}

    if state.step_index + 1 >= len(state.plan):
        # print("All plan steps executed")
        log_data(state=state, message="All plan steps executed")
        return {"decision": "stop", "iteration": newIteration}

    if state.analysis and state.analysis.vulnerability_found:
        # print("Vulnerability found -> continue scanning next endpoint")
        log_data(
            state=state,
            message="Vulnerability found -> continue scanning next endpoint",
        )
        return {
            "decision": "continue",
            "step_index": state.step_index + 1,
            "iteration": newIteration,
        }
    if state.confidence < 0.3:
        # print("REPLAN -> low confidence")
        log_data(state=state, message="REPLAN -> low confidence")
        return {"decision": "replan", "iteration": newIteration}

    # print("CONTINUE -> next step")
    log_data(state=state, message="CONTINUE -> next step")

    return {
        "decision": "continue",
        "step_index": state.step_index + 1,
        "iteration": newIteration,
        "agent_log": state.agent_log,
    }


def outputNode(state: sqlmapAgentState):
    print(json.dumps(state.model_dump(), indent=4))


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
    logFile = logDir / f"sqlmap_agent_log3.log"
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


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


async def agentRunner(endpoints):
    agentState = sqlmapAgentState()
    logger = setupLogger()

    workflow = StateGraph(sqlmapAgentState)
    SESSION_ID = "default_session"

    # prepare attack vectors - filter out vectors without parameters
    for vector in endpoints:
        if len(vector.get("params", [])) > 0:
            agentState.attack_vectors.append(vector)

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

    with open("MCP_tools\gobuster\crawler_test_dump2.json", "r") as f:
        endpoints = json.load(f)

    result = asyncio.run(agentRunner(endpoints=endpoints))
