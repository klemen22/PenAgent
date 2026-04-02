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
    ALLOWED_ARGS = json.load(f)


setupMetadataLogger(agent_name=AGENT_NAME)
setupReasoningLogger(agentName=AGENT_NAME)

logReasoning(
    agentName=AGENT_NAME, reasoning=f" {20 * "="} SQLMAP REASONING {20 * "="} "
)
# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


# ---------------- Plan ---------------- #
class agentPlanStep(BaseModel):
    description: str
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
    decision: Optional[
        Literal["conitnue", "replan", "finish_vector", "finish", "stop", "expand"]
    ] = Field(default=None)
    reasoning: str = Field(..., description="Agent's explanation for current decision")
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

    # plan expansion
    expand_flag: bool = Field(default=False)
    expand_count: int = Field(default=0)
    expand_max: int = 2

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
async def initNode(state: sqlmapAgentState):
    logData(message="[INIT NODE] -> enter node")

    if not state.attack_vectors:
        logData(
            message="[INIT NODE] -> error no attack vectors were given to the agent, stopping..."
        )
        state.fail_reason = "No attack vectors were given to the agent!"

        logData(message="[INIT NODE] -> exit node")
        return {
            "fail": True,
            "fail_reason": state.fail_reason,
            "decision": "stop",
        }

    hostnames = set()

    for vector in state.attack_vectors:
        endpoint = vector.get("endpoint", "")
        logData(message=f"[INIT NODE] -> parsing: {endpoint}")

        if endpoint:
            try:
                parsed = urlparse(endpoint)
                if parsed.hostname:
                    hostnames.add(parsed.hostname)

            except Exception as e:
                logData(
                    message=f"[INIT NODE] -> error parsing endpoint {endpoint}: {str(e) if e else "None"}"
                )
    state.targets = list(hostnames)
    logData(message=f"[INIT NODE] -> targets successfuly extracted: {state.targets}")

    logData(message="[INIT NODE] -> exit node")
    return {
        "targets": state.targets,
        "decision": "continue",
    }


async def planningNode(state: sqlmapAgentState):
    logData(message=f"[PLANNING NODE] -> enter node")

    if not state.vectors_memory:
        for vector in state.attack_vectors:
            key = f"{vector['endpoint']}::{vector['method']}"
            state.vectors_memory[key] = attackVectorMemory(vector_data=vector)

    if state.vector_index >= len(state.attack_vectors):
        logData(
            message=f"[PLANNING NODE] -> all attack vectors were tested, stopping..."
        )
        return {
            "decision": "data",
            "done": True,
        }

    currentMemory = getCurrentVector(state=state)

    if (
        currentMemory.plan
        and not currentMemory.replan_flag
        and not currentMemory.expand_flag
    ):
        logData(
            message=f"[PLANNING NODE] -> exit node (we already have a plan)",
        )
        return {
            "decision": "continue",
        }

    logData(message="[PLANNING NODE] -> planning...")

    if currentMemory.expand_flag:
        logData(message="[PLANNING NODE] -> performing plan expansion,...")
        history = "\n".join(
            [f"- {s.phase}: {s.description}" for s in currentMemory.plan]
        )

        prompt = f"""
        You are an autonomous SQL injection agent in EXPANSION mode.
        Vulnerability is CONFIRMED, but the global objective is not yet reached.
        
        Objective: {state.objective}
        Endpoint: {currentMemory.vector_data['endpoint']}
        
        HISTORY OF COMPLETED STEPS:
        {history}
        
        LAST FINDINGS:
        {currentMemory.analysis.analysis if currentMemory.analysis else "Vulnerability confirmed."}
        
        TASK:
        Generate 1 to 3 ADDITIONAL 'exploitation' steps to continue the attack.
        Focus on extracting specific data (dumping tables, users, or passwords) based on your findings.
        
        STRICT RULES:
        1. ALL new steps MUST be phase: 'exploitation'.
        2. 'params': MUST be ONLY: {currentMemory.vector_data['params']}.
        3. Be specific in descriptions (e.g., "Dump users table from database 'users_db'").

        Return valid JSON:
        {{
            "reasoning": "why more steps are needed and what you aim to extract",
            "steps": [
                {{
                    "description": "new exploitation step",
                    "method": "{currentMemory.vector_data['method']}",
                    "params": {currentMemory.vector_data['params']},
                    "phase": "exploitation"
                }}
            ]
        }}
        """

    elif currentMemory.replan_flag:
        logData(message="[PLANNING NODE] -> performing replan,...")

        prompt = f"""
        You are an autonomous SQL injection agent in REPLAN mode.
        The previous plan failed to confirm or exploit the vulnerability.
        
        Objective: {state.objective}
        Endpoint: {currentMemory.vector_data['endpoint']}
        
        LAST ANALYSIS:
        {currentMemory.analysis.analysis if currentMemory.analysis else "No analysis available."}
        
        TASK:
        Create a NEW strategic plan with a different approach. 
        Consider using higher 'level'/'risk' or suggesting 'tamper' scripts in your reasoning.
        The plan must still have at least one 'detection' and one 'exploitation' step.

        STRICT RULES:
        1. 'params': MUST be ONLY: {currentMemory.vector_data['params']}.
        2. Do NOT repeat the exact same failed strategy.
        
        Return valid JSON:
        {{
            "reasoning": "why the previous plan failed and what you are changing",
            "steps": [
                {{
                    "description": "step description",
                    "method": "{currentMemory.vector_data['method']}",
                    "params": {currentMemory.vector_data['params']},
                    "phase": "detection/exploitation"
                }}
            ]
        }}   
        """
    else:
        logData(
            message="[PLANNING NODE] -> generating initial plan for current attack vector,..."
        )

        prompt = f"""
        You are an autonomous SQL injection agent.
        Objective: {state.objective}
        
        Attack vector:
        Endpoint: {currentMemory.vector_data['endpoint']}
        Method: {currentMemory.vector_data['method']}
        Parameters to test: {currentMemory.vector_data['params']}
        
        Create a strategic scan plan. 
        You MUST include AT LEAST 2 steps:
        1. A 'detection' phase step: Use SQLMap to confirm if the parameter is vulnerable.
        2. An 'exploitation' phase step: If confirmed, perform initial enumeration (e.g., fetch current database name or banner).
        
        You can include up to 4 steps if you think initial enumeration requires separate actions.

        STRICT RULES:
        1. 'params': This MUST be a list containing ONLY the exact parameter names: {currentMemory.vector_data['params']}.
        2. 'phase': First step MUST be 'detection', others MUST be 'exploitation'.
        3. Do NOT invent new parameters or dummy values.
        
        Return valid JSON:
        {{
            "reasoning": "your reasoning for the initial strategy",
            "steps": [
                {{
                    "description": "step description",
                    "method": "{currentMemory.vector_data['method']}",
                    "params": ["param1"],
                    "phase": "detection/exploitation"
                }}
            ]
        }}
        """
    retries = 3

    for attempt in range(retries):

        try:
            outputPlanFull = await llm.with_structured_output(
                agentPlanOutput, include_raw=True
            ).ainvoke(prompt)

            outputPlan = outputPlanFull["parsed"]
            outputPlanRaw = outputPlanFull["raw"]

            if currentMemory.expand_flag:
                currentMemory.expand_flag = False
                currentMemory.plan.extend(outputPlan.steps)
                logData(
                    message=f"[PLANNING NODE] -> appended following additional plan steps: {outputPlan.steps}"
                )
                logData(
                    message=f"[PLANNING NODE] -> reasoning behind additional steps: {outputPlan.reasoning}"
                )
                return {
                    "vectors_memory": state.vectors_memory,
                    "decision": "continue",
                }

            elif len(outputPlan.steps) >= 2:
                logData(
                    message=f"[PLANNING NODE] -> created following plan:{outputPlan.steps}",
                )
                logData(
                    message=f"[PLANNING NODE] -> reasoning behind set plan: {outputPlan.reasoning}"
                )
                logMetadata(
                    agent_name=AGENT_NAME, metadata=outputPlanRaw.response_metadata
                )

                currentMemory.replan_flag = False
                currentMemory.plan = outputPlan.steps
                currentMemory.step_index = 0
                logData(message=f"[PLANNING NODE] -> exit node")
                return {
                    "vectors_memory": state.vectors_memory,
                    "decision": "continue",
                }

            else:
                logData(
                    message=f"[PLANNING NODE]: Generated plan with less than 2 steps, trying again..."
                )
                prompt += "\n\n[WARNING]: Plan was generated with less than 2 steps. Try again and make sure you create at least 2 STEP plan with: 'detection' and 'explotation' phase!"

        except Exception as e:

            logData(message=f"[PLANNING NODE] -> planning attempt failed: {str(e)}")

            prompt += "\n\n[WARNING]: Your previous output was not valid JSON. Return ONLY raw JSON. No explanations."

    logData(message="[PLANNING NODE] -> planning failed, moving to output node")

    currentMemory.plan = []
    currentMemory.step_index = 0
    state.fail_reason = f"Planning failed after {attempt} attempts.\nLast recieved error: {str(e) if e else "None."} "

    logData(message=f"[PLANNING NODE] -> exit node")
    return {
        "vectors_memory": state.vectors_memory,
        "decision": "stop",
        "fail": True,
        "fail_reason": state.fail_reason,
    }


async def selectActionNode(state: sqlmapAgentState):
    logData(message=f"[SELECT ACTION NODE] -> enter node")

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
    {currentMemory.analysis.analysis if currentMemory.analysis else "None"}

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
                message=f"[SELECT ACTION NODE] -> reasoning: {selection.reasoning}",
            )
            logData(message=f"[SELECT ACTION NODE -> exit node]")
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
            finalPrompt += f"\n\nWARNING: Your previous output produced an exception probably due to invalid JSON output.\nException recieved:{str(e) if e else "None"}"

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
    logData(message="[EXECUTE TOOL] -> enter node")
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
    logData(f"[TOOL PAYLOAD]: {str(tool_payload)}")

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

    logData(message="[EXECUTE TOOL] -> exit node")
    return {
        "vectors_memory": state.vectors_memory,
        "decision": "continue",
    }


async def analyzeNode(state: sqlmapAgentState):
    logData(message="[ANALYZE NODE] -> enter node")

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

    totalSteps = len(currentMemory.plan) if currentMemory.plan else 0
    currentStep = currentMemory.step_index + 1

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
    Current vector: {currentMemory.vector_data['endpoint']}
    Current phase: {plan.phase if plan else "unknown"}
    
    [ANALYSIS]
    {analysisPrompt}
    
    [PLAN STATUS]
    - current step: {currentStep} of {totalSteps}
    - replan count: {currentMemory.replan_count} / {currentMemory.max_replans}
    - expand count: {currentMemory.expand_count} / {currentMemory.expand_max}
    
    STRICT DECISION RULES (Read Carefully):
    1. 'continue': 
        - If you are on step {currentStep} of {totalSteps} and there are more steps remaining, you MUST select 'continue'.
        - If vulnerability is found in the 'detection' phase, you MUST select 'continue' to move to 'exploitation'.
    2. 'expand': If you are on the LAST step ({totalSteps} of {totalSteps}), the vulnerability IS confirmed, but you need MORE steps (like dumping specific tables or data) to fully achieve the global objective.
    3. 'finish_vector': 
        - ONLY select this if you are on the LAST step ({totalSteps} of {totalSteps}).
        - OR if you have reached maximum number of replans ({currentMemory.max_replans} of {currentMemory.max_replans}).
        - OR if you have reached maximum number of plan expansion ({currentMemory.expand_max} of {currentMemory.expand_max}).
        - OR if the vector is confirmed NOT vulnerable after multiple attempts.
        - NEVER select this if you are still on step 1 and vulnerability was found!
    4. 'replan': If the current plan isn't working (low confidence or errors) but another approach (tamper scripts, higher level/risk) might work.
    5. 'finish': If the global objective "{state.objective}" is fully achieved across all vectors.
    6. 'stop': Emergency exit only for critical unrecoverable agent errors.
    
    Return valid JSON:
    - decision <ONE of ["conitnue", "replan", "finish_vector", "finish", "stop", "expand"]>
    - reasoning <your reasoning for your decision>
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

        # decision override
        if decision == "finish_vector" and currentStep < totalSteps:
            if currentMemory.analysis and currentMemory.analysis.vulnerability_found:
                logData(
                    message="[EVALUATE NODE] -> OVERRIDE: llm tried to finish attack vector early without explotation phase!"
                )
                decision = "continue"

        logData(message=f"[EVALUATE NODE] -> evaluation done (decision: {decision})")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[EVALUATE] Decision: {decision}. Reason: {feedback.reasoning}",
        )
        logMetadata(agent_name=AGENT_NAME, metadata=feedbackRaw.response_metadata)

        if decision == "expand":
            currentMemory.expand_count += 1
            if currentMemory.expand_count >= currentMemory.expand_max:
                logData(
                    "[EVALUATE NODE] -> maximum number of expansions reached, moving to next vector..."
                )
                state.vector_index += 1
                currentMemory.done = True

                return {
                    "iteration": state.iteration,
                    "decision": "plan",
                    "vector_index": state.vector_index,
                    "vectors_memory": state.vectors_memory,
                }
            else:
                currentMemory.expand_flag = True
                logData(
                    message="[EVALUATE NODE] -> objective not done yet, plan expansion is needed"
                )
                return {
                    "decision": "plan",
                    "vectors_memory": state.vectors_memory,
                    "iteration": state.iteration,
                }

        if decision == "stop":
            logData(
                message="[EVALUATE NODE] -> agent encountered a critical error, moving to output node..."
            )
            state.fail_reason += f"\nProcess was forcefully stopped by evlaute agent with following reasoning: {feedback.reasoning}"

            logData(message="[EVALUATE NODE] -> exit node")
            return {
                "fail": True,
                "fail_reason": state.fail_reason,
                "decision": "stop",
                "vectors_memory": state.vectors_memory,
            }

        if decision == "finish_vector":
            state.vector_index += 1
            currentMemory.done = True

            logData(
                message="[EVALUATE NODE] -> attack vector finished, moving to planning node..."
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

            if state.fail:
                state.fail = False
                state.fail_reason = None
                logData(
                    message="[EVALUATE NODE] -> current plan was not successful, moving to planning node..."
                )

            if currentMemory.replan_count >= currentMemory.max_replans:
                state.vector_index += 1
                currentMemory.done = True
                logData(
                    message="[EVALUATE NODE] -> max. number of replans reached moving to next vector..."
                )

                logData(message="[EVALUATE NODE] -> exit node")
                return {
                    "vector_index": state.vector_index,
                    "decision": "plan",
                    "vectors_memory": state.vectors_memory,
                    "fail": state.fail,
                    "fail_reason": state.fail_reason,
                }

            else:
                currentMemory.replan_flag = True

                logData(message="[EVALUATE NODE] -> exit node")
                return {
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
                "decision": "data",
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
            result = retrieveData(targetAddress=target)
            logData(
                message=f"[DATA RETRIEVE NODE] -> data retrieval for target {target} was successful"
            )
            state.retrieved_data.append(
                retrievedData(
                    target=target,
                    status=result["status"],
                    message=result["message"],
                    data_path=result["data_path"],
                    files=result["files"],
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
    logData(message="[OUTPUT NODE] -> enter node")
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
        retrieved_data = state.retrieved_data

        prompt = f"""
        [TASK]
            
        Create a concise summary based on the given initial objective and gathered infromation.
        
        [OBJECTIVE]
        {state.objective}
        
        [FINAL VECTORS]
        {vectors}
        
        [RETRIEVED DATA]
        {retrieved_data if retrieved_data else "None"}
        """

    logData(message="[OUTPUT NODE] -> generating summary")
    response = await llm.ainvoke(prompt)
    state.summary = response.content

    logMetadata(agent_name=AGENT_NAME, metadata=response.response_metadata)
    logData(message="[OUTPUT NODE] -> exit node - summary done")
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


def setupLogger():
    logFile = logDir / f"sqlmap_agent_log{logCount}.log"
    logger = logging.getLogger("sqlmap_agent_log")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def logData(message: str):
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


def retrieveCurrentDecision(state: sqlmapAgentState):
    return state.decision


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


def sqlmapBuilder():
    setupLogger()
    workflow = StateGraph(sqlmapAgentState)

    # -------------------------------
    # graph nodes
    # -------------------------------

    workflow.add_node("init_node", initNode)
    workflow.add_node("planning_node", planningNode)
    workflow.add_node("select_action_node", selectActionNode)
    workflow.add_node("tool_execution_node", toolExecutionNode)
    workflow.add_node("analyze_node", analyzeNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("output_node", outputNode)
    workflow.add_node("data_retrieve_node", dataRetrieveNode)

    # -------------------------------
    # graph edges
    # -------------------------------

    workflow.add_edge(START, "init_node")
    workflow.add_conditional_edges(
        "init_node",
        retrieveCurrentDecision,
        {
            "continue": "planning_node",
            "stop": "output_node",
        },
    )
    workflow.add_conditional_edges(
        "planning_node",
        retrieveCurrentDecision,
        {
            "continue": "select_action_node",
            "stop": "output_node",
            "data": "data_retrieve_node",
        },
    )
    workflow.add_conditional_edges(
        "select_action_node",
        retrieveCurrentDecision,
        {
            "continue": "tool_execution_node",
            "plan": "planning_node",
            "evaluate": "evaluate_node",
        },
    )
    workflow.add_conditional_edges(
        "tool_execution_node",
        retrieveCurrentDecision,
        {
            "continue": "analyze_node",
            "evaluate": "evaluate_node",
        },
    )
    workflow.add_edge("analyze_node", "evaluate_node")
    workflow.add_conditional_edges(
        "evaluate_node",
        retrieveCurrentDecision,
        {
            "stop": "output_node",
            "data": "data_retrieve_node",
            "plan": "planning_node",
            "continue": "select_action_node",
        },
    )
    workflow.add_edge("data_retrieve_node", "output_node")
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

    print(f"[FINAL RESULT]:\n\n{result.get("agent_output")}")


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "SQLmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    with open("MCP_tools\gobuster\crawler_final_dump1.json", "r") as f:
        endpoints = json.load(f)

    result = asyncio.run(
        agentRunner(
            prompt="Analyze and exploit  given attack vectors.", endpoints=endpoints
        )
    )
