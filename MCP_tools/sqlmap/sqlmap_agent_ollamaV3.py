from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any, Literal
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import json
import logging
from urllib.parse import urlparse
from pathlib import Path

load_dotenv()

from MCP_tools.sqlmap.sqlmap_tool import sqlmap_scan, sqlmapConfig
from Orchestrator.memory.agent_output import AgentOutput, vulnerability
from metadata.metadata_logger import setupMetadataLogger, logMetadata, logTotalTokens
from reasoning.reasoning_logger import setupReasoningLogger, logReasoning
from MCP_tools.sqlmap.sqlmap_data_manager import retrieveData, deleteHistory

# TODO: check if async is really needed or if you can convert code back to sync?
# TODO: decision check routing, planning node, add dataRetrieveNode, rework graph egdes

# ------------------------------------------------------------------------------- #
#                                  LLM setup                                      #
# ------------------------------------------------------------------------------- #

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")
TOKEN_WINDOW_SIZE = os.getenv(key="TOKEN_WINDOW_SIZE", default=4096)
AGENT_NAME = "sqlmap_agent"
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

finalAgent = llm.bind_tools([sqlmap_scan])

# log count
logDir = Path("MCP_tools/sqlmap/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1

with open("MCP_tools\sqlmap\sqlmap_allowed_arguments.json") as f:
    ALLOWED_ARGS = json.read()


setupMetadataLogger(agent_name=AGENT_NAME)
setupReasoningLogger(agentName=AGENT_NAME)

logReasoning(agentName=AGENT_NAME, reasoning=f" {20 * "="} NMAP REASONING {20 * "="} ")
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
    reasoning: str

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


# ---------------- Analysis ---------------- #
class agentFeedback(BaseModel):
    reasoning: str = Field(..., description="Agent's explanation for current decision")
    decision: Optional[Literal["conitnue", "replan", "finish_vector", "finish"]] = (
        Field(default=None)
    )
    confidence: float = Field(default=0.0)


class agentAnalysis(BaseModel):
    analysis: str = Field(..., description="Agent's analysis of the current situation.")
    vulnerability_found: bool = Field(default=False)
    exploitation_possible: bool = Field(default=False)
    confidence: float = Field(default=0.0)


class retrievedData(BaseModel):
    target: str
    status: Literal["success", "failed"] = Field(default="failed")
    message: str = Field(..., description="Information message from manager.")
    data_path: Optional[str] = Field(default=None)
    files: Optional[Any] = Field(default=None)


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

    feedback: Optional[agentFeedback] = Field(
        default=None, description="Evaluate node feedback."
    )

    analysis: Optional[agentAnalysis] = Field(
        default=None, description="Tool output analysis, instead of output parsing."
    )

    # replanning
    replan_reasoning: Optional[str] = Field(default=None)
    replan_count: int = Field(default=0)
    max_replans: int = 5
    replan_flag: bool = Field(default=False)

    done: bool = Field(default=False)


class sqlmapAgentState(BaseModel):
    objective: str = Field(default="", description="Current agent objective.")
    targets: List[str] = Field(default_factory=list)

    attack_vectors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Attack vectors list given by the orchestrator.",
    )

    vector_index: int = Field(default=0)

    vectors_memory: Dict[str, attackVectorMemory] = Field(default_factory=dict)

    iteration: int = Field(default=0)
    max_iteration: int = Field(default=100)

    decision: Optional[Literal["continue", "stop", "evaluate", "plan", "data"]] = Field(
        default="continue"
    )

    retrieved_data: List[retrievedData] = Field(default_factory=list)
    summary: Optional[str] = Field(default=None)

    agent_output: AgentOutput = Field(default_factory=AgentOutput)

    fail: bool = Field(default=False)
    fail_reason: Optional[str] = Field(default="")
    done: bool = Field(default=False)


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #


async def planningNode(state: sqlmapAgentState):
    logData(state=state, message=f"[PLANNING NODE] -> enter node")

    if not state.vectors_memory:
        for vector in state.attack_vectors:
            key = f"{vector['endpoint']}::{vector['method']}"
            state.vectors_memory[key] = attackVectorMemory(vector_data=vector)

    if state.vector_index >= len(state.attack_vectors):
        logData(
            message=f"[PLANNING NODE] -> all attack vectors were tested, stopping..."
        )
        return {
            "decision": "stop",
            "done": True,
        }

    currentMemory = getCurrentVector(state=state)

    if currentMemory.plan and not currentMemory.replan_flag:
        logData(
            state=state, message=f"[PLANNING NODE] -> exit node (all vectors are done)"
        )
        return {
            "decision": "continue",
        }

    logData(message="[PLANNING NODE] -> planning...")

    additionalPrompt = ""

    if currentMemory.replan_flag:
        additionalPrompt = f"""
        [WARNING]
        You must replan your actions for endpoint: {currentMemory.vector_data['endpoint']}
        Replan reason: {currentMemory.replan_reason}
        Last tool output: {currentMemory.last_tool_output}
        Previous feedback: {currentMemory.feedback.reasoning if currentMemory.feedback else "None"}        
        """

    prompt = f"""
    You are an autonomous SQL injection agent.
    Objective: {state.objective}
    

    Attack vector:
    Endpoint: {currentMemory.vector_data['endpoint']}
    Method: {currentMemory.vector_data['method']}
    Parameters: {currentMemory.vector_data['params']}
    
    Create a strategic scan plan. 
    You MUST always include:
    1. Detection phase (verify if vulnerable)
    2. Exploitation phase (extract data if vulnerable)

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

    finalPrompt = additionalPrompt + prompt

    retries = 3

    for attempt in range(retries):

        try:
            outputPlanFull = await llm.with_structured_output(
                agentPlanOutput, include_raw=True
            ).ainvoke(finalPrompt)

            outputPlan = outputPlanFull["parsed"]
            outputPlanRaw = outputPlanFull["raw"]

            logData(
                message=f"[PLANNING NODE] -> created following plan:{outputPlan.reasoning}",
            )
            logMetadata(agent_name=AGENT_NAME, metadata=outputPlanRaw.response_metadata)

            currentMemory.plan = outputPlan.steps
            currentMemory.step_index = 0
            logData(state=state, message=f"[PLANNING NODE] -> exit node")
            return {
                "vectors_memory": state.vectors_memory,
                "decision": "continue",
                "target": state.target,
            }

        except Exception as e:

            logData(message=f"[PLANNING NODE] -> planning attempt failed: {str(e)}")

            prompt += "\n\nWARNING: Your previous output was not valid JSON. Return ONLY raw JSON. No explanations."

    logData(message="[PLANNING NODE] -> planning failed, moving to output node")

    currentMemory.plan = []
    currentMemory.step_index = 0
    state.fail_reason = f"Planning failed after {attempt} attempts.\nLast recieved error: {str(e) if e else "None."} "

    logData(state=state, message=f"[PLANNING NODE] -> exit node")
    return {
        "vectors_memory": state.vectors_memory,
        "decision": "stop",
        "target": state.target,
        "fail": True,
        "fail_reason": state.fail_reason,
    }


async def selectActionNode(state: sqlmapAgentState):
    logData(state=state, message=f"[SELECT ACTION NODE] -> enter node")

    currentMemory = getCurrentVector(state=state)

    if currentMemory.step_index >= len(currentMemory.plan):
        logData(
            state=state, message="[SELECT ACTION NODE] -> exit node (replan needed!)"
        )
        currentMemory.replan_reasoning = (
            "Current step index exceeds lenght of the plan!"
        )
        currentMemory.replan_flag = True

        logData(message="[SELECT ACTION NODE] -> exit node")
        return {
            "decision": "plan",
            "vectors_memory": state.vectors_memory,
        }

    step = currentMemory.plan[currentMemory.step_index]

    if not step:
        logData(message="[SELECT ACTION NODE] -> recieved empty or invalid plan step")
        currentMemory.replan_reasoning = "Recieved empty or invalid plan step!"
        currentMemory.replan_flag = True

        logData(message="[SELECT ACTION NODE] -> exit node")
        return {
            "decision": "plan",
            "vectors_memory": state.vectors_memory,
        }

    basePrompt = f"""
    Objective: {state.objective}
    Current attack vector:
    Endpoint: {currentMemory.vector_data['endpoint']}
    Method: {currentMemory.vector_data['method']}
    Parameters: {currentMemory.vector_data['params']}

    Current plan step description: {step.description}
    Parameters under test: {step.params}

    Previous analysis:
    {currentMemory.analysis.reasoning if currentMemory.analysis else "None"}

    Available sqlmap options:
    {ALLOWED_ARGS}
    """

    if step.phase == "detection":
        additonalPrompt = f"""
        PHASE: DETECTION
        Your goal is to confirm if '{step.params}' are vulnerable.
        
        Strategy:
        1. Start with 'detection.safe' options.
        2. Use 'optimization' and 'evasion' (e.g., --random-agent, --batch) for better results.
        3. Only move to 'detection.aggressive' if previous safe attempts were inconclusive.
        4. Use flags like --banner or --current-db to verify a successful hit.
        
        STRICT RULE: Do NOT use anything from 'enumeration' or 'exploitation' categories yet.
        """
    else:
        additonalPrompt = f"""
        PHASE: EXPLOITATION
        Vulnerability is confirmed. Goal: {step.description}
        
        Strategy:
        1. Use 'enumeration.safe' to map the database structure (--dbs, --tables).
        2. Use 'enumeration.aggressive' (--dump) ONLY if you need to extract specific data.
        3. Use 'exploitation.high_risk' ONLY if the objective specifically requires OS access or file manipulation.
        4. Always include 'optimization' flags like --threads or --keep-alive for efficiency.
        
        STRICT RULE: Focus on extracting the information required to fulfill the objective.
        """

    additonalRules = f"""
    OUTPUT RULES:
    - You MUST explain which category of arguments you are using and why (e.g., "Using evasion because a WAF is suspected").
    - If you use 'aggressive' or 'high_risk' options, justify the risk in your reasoning.
    
    Return valid JSON:
        - reasoning
        - url
        - method
        - data (optional)
        - level (optional)
        - risk (optional)
        - technique (optional)
        - threads (optional)
        - random_agent (optional)
        - enumeration (optional)
        - tamper (optional)
    """

    finalPrompt = basePrompt + additonalPrompt + additonalRules

    for attempt in range(3):

        try:

            selectionFull = await llm.with_structured_output(
                sqlmapToolSelection, include_raw=True
            ).ainvoke(finalPrompt)

            selection = selectionFull["parsed"]
            selectionRaw = selectionFull["raw"]

            logData(
                state,
                message=f"[SELECT ACTION NODE] -> reasoning: {selection.reasoning}",
            )
            logData(state=state, message=f"[SELECT ACTION NODE -> exit node]")
            logMetadata(agent_name=AGENT_NAME, metadata=selectionRaw.response_metadata)

            currentMemory.selected_command = selection

            return {
                "vectors_memory": state.vectors_memory,
                "decision": "continue",
            }
        except Exception as e:
            logData(
                message=f"[SELECT TOOL CALL] -> failed creating tool call: {str(e)}"
            )
            prompt += f"\n\nWARNING: Your previous output produced an exception probably due to invalid JSON output.\nException recieved:{str(e) if e else "None"}"

    state.fail_reason = f"""
    Planning failed after {attempt} attempts
    
    Last step before failure:
    {step}
    
    Last error recieved:
    {str(e) if e else "None"}  
    """

    return {
        "fail": state.fail,
        "fail_reason": state.fail_reason,
        "decision": "evaluate",
    }


async def toolExecutionNode(state: sqlmapAgentState):
    print("\n[EXECUTE TOOL]\n")
    logData(state=state, message="[EXECUTE TOOL] -> enter node")
    currentMemory = getCurrentVector(state=state)

    if not currentMemory.selected_command:
        logData(
            state=state,
            message="[EXECUTE TOOL] -> no tool call present, skipping tool call...",
        )
        state.fail_reason = f"Recieved invalid or empty tool call parameters, tool exectuion was skipped!"

        return {
            "decision": "evaluate",
            "fail": True,
            "fail_reason": state.fail_reason,
        }

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

    step = currentMemory.plan[currentMemory.step_index]

    if step.phase == "detection":
        config_payload["current_db"] = False
        config_payload["enumerate_tables"] = False

    if step.phase == "exploitation":
        config_payload["current_db"] = True
        config_payload["enumerate_tables"] = True

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
    logData(state, f"[TOOL PAYLOAD]: {str(tool_payload)}")

    try:
        currentMemory.last_tool_result = await sqlmap_scan(
            url=tool_payload["url"],
            data=tool_payload["data"],
            config=sqlmapConfig(**tool_payload["config"]),
        )
    except Exception as e:
        currentMemory.last_tool_result = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }
        logData(
            message=f"[EXECUTE TOOL] -> tool call failed: {str(e) if e else "None."}"
        )
        return {
            "vectors_memory": state.vectors_memory,
            "decision": "evaluate",
        }

    logData(state=state, message="[EXECUTE TOOL] -> exit node")
    return {
        "vectors_memory": state.vectors_memory,
        "decision": "continue",
    }


async def analyzeNode(state: sqlmapAgentState):
    logData(state=state, message="[ANALYZE NODE] -> enter node")

    currentMemory = getCurrentVector(state=state)
    toolResult = currentMemory.last_tool_result

    if not toolResult:
        logData(
            message="[ANALYZE NODE] -> No tool result provided, skipping analysis..."
        )
        currentMemory.analysis = agentAnalysis(
            analysis="No valid output provided from sqlmap.",
            vulnerability_found=False,
            exploitation_possible=False,
            confidence=0.0,
        )

        return {"vectors_memory": state.vectors_memory}

    prompt = f"""
    You are a SQLmap Output Analyst. 
    Analyze the following tool output for endpoint: {currentMemory.vector_data['endpoint']}

    TOOL OUTPUT:
    {toolResult}

    OBJECTIVE:
    {state.objective}

    INSTRUCTIONS:
    1. Look for keywords: "injectable", "vulnerable", "back-end DBMS is...", "fetching tables".
    2. Set 'vulnerability_found' to True ONLY if sqlmap explicitly confirmed a vulnerability.
    3. Set 'exploitation_possible' to True if sqlmap confirmed vulnerability AND you see signs that data extraction (dumping, tables) is working.
    4. Provide a confidence score (0.0 - 1.0) based on how certain the results are.
    5. In 'analysis', summarize what was found (e.g., "Parameter 'id' is vulnerable to Union-based SQLi").

    If the output says "all parameters appear not to be injectable", vulnerability_found must be False.
    
    Return valid JSON:
    - analysis
    - vulnerability_found
    - exploitation_possible
    - confidence
    """

    try:
        analysisfull = await llm.with_structured_output(
            agentAnalysis, include_raw=True
        ).ainvoke(prompt)

        analysisRaw = analysisfull["raw"]
        analysis = analysisfull["parsed"]

        if analysis.confidence > 1.0:
            analysis.confidence = 1.0

        currentMemory.analysis = analysis

        logData(
            message=f"[ANALYZE NODE] -> analysis successful\n> found: {analysis.analysis}\n> confidence: {analysis.confidence}",
        )
        logMetadata(agent_name=AGENT_NAME, metadata=analysisRaw.response_metadata)

        logData(message="[ANALYZE NODE] -> exit node")
        return {
            "vectors_memory": state.vectors_memory,
        }

    except Exception as e:
        logData(message=f"[ANALYZE NODE] -> error: {str(e) if e else "None"}")

    state.fail_reason = f"Analysis of tool output failed.\nRecieved following exception: {str(e) if e else "None"}"

    return {
        "vectors_memory": state.vectors_memory,
        "fail": state.fail,
        "fail_reason": state.fail_reason,
    }


async def evaluateNode(state: sqlmapAgentState):
    logData(message="[EVALUATE NODE] -> enter node")
    state.iteration += 1

    currentMemory = getCurrentVector(state=state)
    plan = currentMemory.plan[currentMemory.step_index] if currentMemory.plan else None

    if state.fail:
        additionalPrompt = f"""
        [WARNING]
        Agent failed at performing designated task.
        
        [ERROR LOG]
        {state.fail_reason}        
        """
    else:
        additionalPrompt = ""

    if currentMemory.analysis.analysis:
        agentAnalysis = currentMemory.analysis

        analysisPrompt = f"""
        Vulnerability found: {agentAnalysis.vulnerability_found}
        Exploitation possible: {agentAnalysis.exploitation_possible}
        Details: {agentAnalysis.analysis}
        Confidence: {agentAnalysis.confidence}
        """
    else:
        analysisPrompt = "None."

    prompt = f"""
    Objective: {state.objective}
    Current vector: {currentMemory}
    Current phase: {plan.phase if plan else "unknown"}
    
    [ANALYSIS]
    {analysisPrompt}
    
    [PLAN STATUS]
    - current step: {currentMemory.step_index + 1} of {len(currentMemory.plan) if currentMemory.plan else 0}
    - replan count: {currentMemory.replan_count} / {currentMemory.max_replans}
    
    DECISION RULES:
    1. 'finish': If objective "{state.objective}" is fully achieved (e.g. data dumped).
    2. 'finish_vector': If this vector is confirmed NOT vulnerable after multiple steps, or we are done with it.
    3. 'replan': If the current plan isn't working (low confidence or error) but you think another approach (tamper script, higher level) might work.
    4. 'continue': If vulnerability is found and we need to proceed to exploitation, or if we need to finish the current plan steps.
    
    Return valid JSON:
    - reasoning <your reasoning for your decision and confidence value>
    - decision <one of [continue, replan, finish_vector, finish]>
    - confidence <number between 0.0 - 1.0>
    """

    finalPrompt = additionalPrompt + prompt

    try:
        feedbackFull = await llm.with_structured_output(
            agentFeedback, include_raw=True
        ).ainvoke(finalPrompt)

        feedback = feedbackFull["parsed"]
        feedbackRaw = feedbackFull["raw"]

        currentMemory.feedback = feedback
        decision = feedback.decision

        if decision == "finish_vector":
            state.vector_index += 1
            currentMemory.done = True

            logData(
                message="[EVLUATE NODE] -> attack vector finished, moving to planning node..."
            )
            logData(message="[EVALUATE NODE] -> exit node")
            return {
                "iteration": state.iteration,
                "decision": "plan",
                "vector_index": state.vector_index,
                "vectors_memory": state.vectors_memory,
            }

        if decision == "replan":
            currentMemory.replan_count += 1
            currentMemory.replan_flag = True

            if state.fail:
                state.fail = False
                state.fail_reason = None
                logData(
                    message="[EVLUATE NODE] -> current plan was not successful, moving to planning node..."
                )

            if currentMemory.replan_count >= currentMemory.max_replans:
                state.vector_index += 1
                logData(
                    message="[EVALUATE NODE] -> max. number of replans reached moving to next vector..."
                )

            logData(message="[EVALUATE NODE] -> exit node")
            return {
                "iteration": state.iteration,
                "vector_index": state.vector_index,
                "decision": "plan",
                "vectors_memory": state.vectors_memory,
                "fail": state.fail,
                "fail_reason": state.fail_reason,
            }

        if decision == "finish":

            if not state.fail:
                state.done = True

            logData(
                message="[EVALUATE NODE] -> all vectors are finished, moving to output node..."
            )
            logData(message="[EVALUATE NODE] -> exit node")
            return {
                "iteration": state.iteration,
                "vectors_memory": state.vectors_memory,
                "decision": "stop",
                "done": state.done,
            }

        # continue decision
        currentMemory.step_index += 1
        logData(message="[EVALUATE NODE] -> moving to next step of the plan...")
        logData(message="[EVALUATE NODE] -> exit node")
        return {
            "iteration": state.iteration,
            "decision": "continue",
            "vectors_memory": state.vectors_memory,
        }

    except Exception as e:
        logData(message=f"[EVALUATE NODE] -> error: {str(e) if e else "None."}")

    state.fail_reason = f"Agent encountered following exception while performing evaluation: {str(e) if e else "None."}"

    return {
        "iteration": state.iteration,
        "vectors_memory": state.vectors_memory,
        "decision": "stop",
        "fail": True,
        "fail_reason": state.fail_reason,
    }


async def dataRetrieveNode(state: sqlmapAgentState):

    targets = state.targets

    if not targets:
        logData(
            "[DATA RETRIEVE NODE] -> no targets provided, skipping data retrieval..."
        )
        logData(message="[DATA RETRIEVE NODE] -> exit node")
        return

    logData(
        message=f"[DATA RETRIEVE NODE] -> retrieving data for targets: {targets}..."
    )
    for target in targets:

        try:
            result = retrieveData()
            logData(
                message=f"[DATA RETRIEVE NODE] -> data retrieval for target {target} was successful"
            )
            state.retrieved_data.append(
                retrievedData(
                    target=target,
                    status=result.status,
                    message=result.message,
                    data_path=result.data_path,
                    files=result.files,
                )
            )
            # clear retrieved data from MCP server
            deleteHistory(targetAddress=target)

        except Exception as e:
            logData(
                f"[DATA RETRIEVE NODE] -> encoutered following exception during data retrieval process for {target}: {str(e) if e else "None."}"
            )
            state.retrieved_data.append(
                retrievedData(
                    target=target,
                    status="failed",
                    message=f"Encountered following exception: {str(e)}" if e else "",
                )
            )

    logData(message="[DATA RETRIEVE NODE] -> exit node")
    return {
        "retrieved_data": state.retrieved_data,
    }


async def outputNode(state: sqlmapAgentState):
    logData(state=state, message="[OUTPUT NODE] -> enter node")
    vectors = json.dumps(
        {k: v.model_dump() for k, v in state.vectors_memory.items()}, indent=4
    )
    if state.fail:
        prompt = f"""
        You failed at performing your task.
        Create a concise summary why that happened based on the bollow listed facts.
        
        Fail reason:
        {state.fail_reason}
        
        Agent state before failure:
        {state.model_dump_json()}
        """
    else:
        prompt = f"""
        [TASK]
            
        Create a concise summary based on the given initial objective and gathered infromation.
        
        [OBJECTIVE]
        {state.objective}
        
        [FINAL VECTORS]
        {vectors}
        """

    logData(state=state, message="[OUTPUT NODE] -> generating summary")
    response = await llm.ainvoke(prompt)
    state.summary = response.content

    logMetadata(agent_name=AGENT_NAME, metadata=response.response_metadata)
    logData(state=state, message="[OUTPUT NODE] -> exit node - summary done")
    logTotalTokens(agent_name=AGENT_NAME)

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
        return {
            "agent_output": AgentOutput(
                agent_name="sqlmap",
                success=True,
                vulnerabilities=sqlmapVectorConverter(state=state),
                summary=state.summary,
            )
        }


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
    # TODO: remove this junk
    if state.decision == "continue":
        return "select_action"

    elif state.decision == "replan":
        return "plan"

    elif state.decision == "plan":
        return "plan"

    elif state.decision == "stop":
        return "output"

    return "output"


def planningNodeRouting(state: sqlmapAgentState):
    if state.decision == "continue":
        return "select_action"
    else:
        return "output"


def setupLogger():
    logFile = logDir / f"sqlmap_agent_log{logCount}.log"
    logger = logging.getLogger("sqlmap_agent_log")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def logData(state, message: str):
    state.agent_log.append(message)
    logging.getLogger("sqlmap_agent_log").info(message)


def getCurrentVector(state: sqlmapAgentState) -> attackVectorMemory:
    if state.vector_index >= len(state.attack_vectors):
        return None

    vector = state.attack_vectors[state.vector_index]

    key = f"{vector['endpoint']}::{vector['method']}"

    return state.vectors_memory.get(key)


def sqlmapVectorConverter(state: sqlmapAgentState) -> List[vulnerability]:

    vectors_memory = state.vectors_memory
    vulnerabilities = []

    for vectorKey, vector in vectors_memory.items():

        analysis = vector.analysis

        if not analysis:
            continue

        if not analysis.vulnerability_found:
            continue

        data = vector.vector_data

        vulnerabilities.append(
            vulnerability(
                host=data.get("host", ""),
                url=data.get("url", ""),
                parameters=data.get("parameters", []),
                vulner_type="SQL Injection",
                severity=severityConverter(analysis.confidence),
                evidence=str(vector.last_tool_result),
            )
        )

    return vulnerabilities


def severityConverter(conf: float):

    if conf > 0.9:
        return "critical"

    if conf > 0.7:
        return "high"

    if conf > 0.4:
        return "medium"

    return "low"


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


def sqlmapBuilder():
    setupLogger()
    workflow = StateGraph(sqlmapAgentState)

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
    workflow.add_conditional_edges(
        "planning_node",
        planningNodeRouting,
        {
            "select_action": "select_action_node",
            "output": "output_node",
        },
    )
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

    # checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=False)

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "MCP_tools\sqlmap\sqlmap_agent_graph.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    return graph


async def agentRunner(prompt: str, endpoints):
    graph = sqlmapBuilder()

    state = sqlmapAgentState()
    state.objective = prompt
    state.attack_vectors = endpoints

    result = await graph.ainvoke(state.model_dump(), {"recursion_limit": 1000})

    print(f"[FINAL RESULT]:\n\n{result}")


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "SQLmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    with open("MCP_tools\gobuster\crawler_final_dump1.json", "r") as f:
        endpoints = json.load(f)

    result = asyncio.run(
        agentRunner(prompt="Analyze given attack vectors.", endpoints=endpoints)
    )
