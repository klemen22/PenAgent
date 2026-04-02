from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Literal
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import logging
from pathlib import Path
import re
from collections import Counter
import json

load_dotenv()

from MCP_tools.gobuster.gobuster_toolV2 import gobuster_scan, gobusterInput
from MCP_tools.gobuster.crawler import main as crawlerMain
from Orchestrator.memory.agent_output import AgentOutput, attackVector
from metadata.metadata_logger import setupMetadataLogger, logMetadata, logTotalTokens
from reasoning.reasoning_logger import setupReasoningLogger, logReasoning

# ------------------------------------------------------------------------------- #
#                                  LLM setup                                      #
# ------------------------------------------------------------------------------- #

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")
TOKEN_WINDOW_SIZE = os.getenv(key="TOKEN_WINDOW_SIZE", default=4096)
AGENT_NAME = "gobuster_agent"
LLM_MODEL = os.getenv(key="LLM_MODEL", default="huihui_ai/qwen3-abliterated:8b")

llm = ChatOllama(
    name=AGENT_NAME,
    model=LLM_MODEL,
    num_ctx=TOKEN_WINDOW_SIZE,
    base_url=LM_API,
    temperature=0.1,
    format=None,
    system="You are a specialized cybersecurity assistant. You MUST always respond in English. Do not use any other languages under any circumstances.",
)

finalAgent = llm.bind_tools([gobuster_scan])

logDir = Path("MCP_tools/gobuster/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1


with open("MCP_tools/gobuster/gobuster_allowed_arguments.json") as f:
    ALLOWED_ARGS = json.load(f)

print(json.dumps(ALLOWED_ARGS, indent=2))
setupMetadataLogger(AGENT_NAME)
setupReasoningLogger(AGENT_NAME)

logReasoning(
    agentName=AGENT_NAME,
    reasoning=f" {20 * "="} GOBUSTER REASONING {20 * "="} ",
)

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class gobusterEndpoint(BaseModel):
    base_url: str = Field(default="")
    path: str = Field(default="")
    status: Optional[int] = Field(default=None)
    size: Optional[int] = Field(default=None)
    redirect: bool = Field(default=False)
    redirect_address: Optional[str] = Field(default=None)
    type: str = Field(default="")


class gobusterMemory(BaseModel):
    scans_performed: List[Dict[str, Any]] = Field(default_factory=list)
    endpoints: List[gobusterEndpoint] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    last_tool_output: Optional[Dict[str, Any]] = Field(default=None)


class gobusterToolCall(BaseModel):
    reasoning: Optional[str] = Field(default="")
    url: str = Field(default="")
    mode: str = Field(default="dir")
    additional_args: str = Field(default="")


class toolFeedback(BaseModel):
    reasoning: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.0, description="Number between 0.0 - 1.0")
    feedback: Optional[Literal["continue", "done", "error"]] = Field(default=None)


class gobusterOutput(BaseModel):
    summary: Optional[str] = Field(default=None)
    host_enum: Dict[str, Dict] = Field(default_factory=dict)
    crawler_result: Optional[Any] = Field(default=None)


class gobusterAgentState(BaseModel):
    objective: str = Field(
        default="", description="Objective given by the orchestartor"
    )
    target: List[str] = Field(default_factory=list)
    target_index: int = Field(default=0)
    tool_call: gobusterToolCall = Field(default_factory=gobusterToolCall)

    memory: gobusterMemory = Field(default_factory=gobusterMemory)

    iteration: int = Field(default=0)
    max_iterations: int = Field(default=10)

    decision: Optional[
        Literal["continue", "stop", "crawler", "done", "evaluate", "parse", "plan"]
    ] = Field(default=None)
    feedback: toolFeedback = Field(default_factory=toolFeedback)
    error_detected: bool = Field(
        default=False,
        description="A flag for seperating normal and error based feedback.",
    )

    # output
    gobuster_output: gobusterOutput = Field(default_factory=gobusterOutput)

    agent_output: AgentOutput = Field(default_factory=AgentOutput)

    # report
    fail: bool = Field(default=False)
    fail_reason: Optional[str] = Field(default=None)


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #


async def planningNode(state: gobusterAgentState):
    logData(message="[PLANNING NODE] -> enter node")
    feedback = state.feedback
    memory = state.memory

    if not state.target:
        urls = re.findall(r"https?://[^\s]+", state.objective)
        ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b", state.objective)

        all_targets = set(urls)
        for ip in ips:
            if not any(ip in u for u in urls):
                all_targets.add(f"http://{ip}")

        state.target = list(all_targets)
        logData(f"[PLANNING NODE] -> Initial targets found: {state.target}")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLANNING] Initial targets found: {state.target}.",
        )

    state.iteration += 1
    # check itterations
    if state.iteration >= state.max_iterations:
        state.fail_reason = f"Max number of iterations reached (number of iterations: {state.iteration})"
        logData(
            message="[PLANNING NODE] -> maximum number of iterations reached -> moving to output node"
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLANNING] Maximum number of iterations reached, moving to output node.",
        )

        return {
            "target": state.target,
            "fail": True,
            "fail_reason": state.fail_reason,
            "iteration": state.iteration,
            "decision": "stop",
        }

    if state.target_index >= len(state.target):
        logData("[PLANNING NODE] -> all targets done, moving to crawler...")
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLANNING] All targets done, moving to crawler.",
        )
        return {
            "decision": "crawler",
            "target_index": state.target_index,
            "iteration": state.iteration,
        }

    current_target = state.target[state.target_index]
    if not state.memory.scans_performed:
        logData(
            f"[PLANNING NODE] -> first agent run, initializing target: {current_target}"
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLANNING] Initializing target: {current_target}.",
        )
        state.feedback.feedback = None
        state.error_detected = False
    elif state.memory.scans_performed[-1]["url"] != current_target:
        logData(
            f"[PLANNING NODE] -> new target detected: {current_target}, resetting feedback for fresh start."
        )
        logReasoning(
            agentName=AGENT_NAME,
            reasoning=f"[PLANNING] Initializing target: {current_target}.",
        )
        state.feedback.feedback = None
        state.error_detected = False

    try:
        recent_endpoints = [
            e.path for e in memory.endpoints if e.base_url == current_target
        ][-15:]
        memory_summary = f"""
            - Current Target: {current_target}
            - Scans performed on this target: {len([s for s in memory.scans_performed if s['url'] == current_target])}
            - Endpoints found for this target: {len([e for e in memory.endpoints if e.base_url == current_target])}
            - Recent paths: {recent_endpoints}
            - Key signals: {memory.signals}
            """
    except:
        memory_summary = "None."

    logData(f"[PLANNING NODE] -> memory summary retrieved: {memory_summary}")
    logData("[PLANNING NODE] -> start planning...")

    print(f"FEEDBACK: {state.feedback.feedback}")

    if state.feedback.feedback is not None:

        if feedback.feedback == "continue":
            if state.error_detected:
                logData("[PLANNING NODE] -> tool encountered error, replanning...")
                prompt = f"""
                You are a Gobuster enumeration agent.
                
                Objective:
                {state.objective}
                
                Target:
                {current_target}
                
                [WARNING]
                
                Last tool call has encountered an error.
                
                Feedback:
                {feedback.reasoning}
                
                [MEMORY SUMMARY]

                {memory_summary}
                
                [TASK]
                
                Try to fix the problem and decide on the next gobuster tool call based on the feedback and warning.
                
                Allowed gobuster arguments - USE ONLY ARGUMENTS PROVIDED BELLOW:
                {ALLOWED_ARGS}
                
                IMPORTANT:
                Do NOT include -w or --wordlist arguments.
                The wordlist is automatically handled by the tool.
                Gobuster does NOT support recursion.
                Do NOT use --recursion.
                
                Return JSON:
                {{  
                    "reasoning": <Your reasoning for this tool call and arguments combination>
                    "url": <a valid target URL>
                    "mode": <gobuster mode -> default value is set to "dir">
                    "additional_args": <A combination of additional gobuster arguments - USE ONLY ARGUMENTS PROVIDED ABOVE>
                }}
                
                CORRECT EXAMPLE:
                {{
                    "reasoning": "Standard dir mode found nothing. I will now search for common web files by adding the -x flag.",
                    "url": "http://192.168.1.10",
                    "mode": "dir",
                    "additional_args": "-x php,txt,zip"
                }}
                
                """

                # reset error flag
                state.error_detected = False
            else:
                logData(f"[PLANNING NODE] -> creating new plan for {current_target}")
                prompt = f"""
                You are a Gobuster enumeration agent.
                
                Objective:
                {state.objective}
                
                Target:
                {current_target}
                
                [FEEDBACK]
                
                From feedback it was decided that additional scans are required.
                
                Last scan performed:
                {state.memory.scans_performed[-1]}
                
                Confidence from last scan:
                {feedback.confidence}
                
                Reasoning for confidence:
                {feedback.reasoning}
                
                [MEMORY SUMMARY]

                {memory_summary}
                
                [TASK]
                
                Decide next gobuster scan based on the above mentioned memory and feedback.
                        
                Allowed gobuster arguments - USE ONLY ARGUMENTS PROVIDED BELLOW:
                {ALLOWED_ARGS}
                
                IMPORTANT:
                Do NOT include -w or --wordlist arguments.
                The wordlist is automatically handled by the tool.
                Gobuster does NOT support recursion.
                Do NOT use --recursion.
                
                Return JSON:
                {{  
                    "reasoning": <Your reasoning for this tool call and arguments combination>
                    "url": <a valid target URL>
                    "mode": <gobuster mode -> default value is set to "dir">
                    "additional_args": <A combination of additional gobuster arguments - USE ONLY ARGUMENTS PROVIDED ABOVE>
                }}
                
                CORRECT EXAMPLE:
                {{
                    "reasoning": "Standard dir mode found nothing. I will now search for common web files by adding the -x flag.",
                    "url": "http://192.168.1.10",
                    "mode": "dir",
                    "additional_args": "-x php,txt,zip"
                }}
                """

    else:
        # initial prompt
        logData("[PLANNING NODE] -> creating initial plan...")
        prompt = f"""
        You are a Gobuster enumeration agent.
        
        Objective:
        {state.objective}
        
        Targets:
        {current_target}
        
        Decide on the initial gobuster scan to begin the enumeration process.
        
        Allowed gobuster arguments - USE ONLY ARGUMENTS PROVIDED BELLOW:
        {ALLOWED_ARGS}
        
        IMPORTANT:
        Do NOT include -w or --wordlist arguments.
        The wordlist is automatically handled by the tool.
        Gobuster does NOT support recursion.
        Do NOT use --recursion.
        
        Return JSON:
        {{  
            "reasoning": <Your reasoning for this tool call and arguments combination>
            "url": <a valid target URL>
            "mode": <gobuster mode -> default value is set to "dir">
            "additional_args": <A combination of additional gobuster arguments - USE ONLY ARGUMENTS PROVIDED ABOVE>
        }}
        
        CORRECT EXAMPLE:
        {{
            "reasoning": "Standard dir mode found nothing. I will now search for common web files by adding the -x flag.",
            "url": "http://192.168.1.10",
            "mode": "dir",
            "additional_args": "-x php,txt,zip"
        }}
        """

    retries = 3
    finalPrompt = prompt

    for attempt in range(retries):
        try:
            outputFull = await llm.with_structured_output(
                gobusterToolCall, include_raw=True
            ).ainvoke(finalPrompt)

            outputPlan = outputFull["parsed"]
            outputRaw = outputFull["raw"]

            logData(message=f"[PLANNING NODE] -> created next tool call: {outputPlan}")
            state.tool_call = outputPlan
            logMetadata(agent_name=AGENT_NAME, metadata=outputRaw.response_metadata)
            logReasoning(
                agentName=AGENT_NAME,
                reasoning=f"[PLANNING] {outputPlan.reasoning}",
            )

            logData(message="[PLANNING NODE] -> exit node")
            return {
                "target": state.target,
                "tool_call": state.tool_call,
                "decision": "continue",
                "iteration": state.iteration,
            }

        except Exception as e:
            logData(f"Planning failed: {str(e)}")
            finalPrompt += f"\n\n[WARNING]\n Your previous output was not a valid JSON or you encountered an error. Return ONLY raw JSON. Error message:\n{str(e)}"

    logData(
        message=f"[PLANNING NODE] -> WARNING: planning failed after {attempt} attempts -> moving to output node"
    )

    state.fail_reason = f"Planning failed after {attempt} attempts. Recieved following error:\n\n{str(e)}"

    return {
        "target": state.target,
        "fail": True,
        "fail_reason": state.fail_reason,
        "iteration": state.iteration,
        "decision": "stop",
    }


async def toolNode(state: gobusterAgentState):
    logData(message="[TOOL NODE] -> enter node")

    toolCall = state.tool_call
    memory = state.memory

    try:
        target = state.target[state.target_index]
    except IndexError:
        logData("[TOOL NODE] -> No more targets available.")
        return {"decision": "output", "error_detected": True}

    toolCall = validateToolCall(toolCall=toolCall, target=target)

    if state.tool_call != toolCall:
        logData(f"[TOOL NODE] -> initial tool call:\n {state.tool_call}")
        logData(f"[TOOL NODE] -> fixed tool call: \n{toolCall}")

    if not toolCall:
        return {
            "decision": "evaluate",
            "error_detected": True,
        }

    memory.scans_performed.append(
        {
            "url": target,
            "mode": toolCall.mode,
            "additional_args": toolCall.additional_args,
        }
    )

    try:
        rawOutput = await gobuster_scan(
            gobusterInput(
                url=toolCall.url,
                mode=toolCall.mode,
                additional_args=toolCall.additional_args,
            )
        )
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }
        logData(message=f"[TOOL NODE] -> Exception occured: {str(e)}")

    if isinstance(rawOutput, tuple):
        rawOutput = rawOutput[1]["result"]

    memory.last_tool_output = rawOutput

    stderr_content = str(rawOutput.get("stderr", "")).lower()
    fatal_errors = [
        "timeout",
        "connection refused",
        "no route to host",
        "context deadline exceeded",
        "connection attempt failed",
    ]

    if any(err in stderr_content for err in fatal_errors):
        logData(f"[TOOL NODE] -> FATAL ERROR detected for {target}: {stderr_content}")
        logData(
            f"[TOOL NODE] -> Host is unreachable. Incrementing target_index and skipping."
        )

        return {
            "target_index": state.target_index + 1,
            "decision": "plan",
            "memory": state.memory,
        }

    if rawOutput.get("stderr"):
        logData(
            f"[TOOL NODE] -> tool encountered following error: {str(rawOutput.get('stderr'))}"
        )

    logData(
        message=f"[TOOL NODE] -> exit node - tool call was successful: {rawOutput.get('success','')}"
    )

    return {
        "decision": "parse",
        "memory": state.memory,
    }


def parseOutputNode(state: gobusterAgentState):
    logData("[PARSE NODE] -> enter node")
    current_target = state.target[state.target_index]
    rawOutput = state.memory.last_tool_output

    if not rawOutput:
        return {"decision": "evaluate"}

    parsed = parseGobusterOutput(rawOutput)

    existing_paths = {e.base_url + e.path.rstrip("/") for e in state.memory.endpoints}

    for e in parsed["endpoints"]:
        e["base_url"] = current_target
        normalized = current_target + e["path"].rstrip("/")

        if normalized not in existing_paths:
            state.memory.endpoints.append(gobusterEndpoint(**e))
            existing_paths.add(normalized)

    state.memory.signals.extend(parsed["signals"])

    return {"memory": state.memory, "decision": "evaluate"}


async def evaluateNode(state: gobusterAgentState):
    logData(message="[EVALUATE NODE] -> enter node")
    memory = state.memory
    feedback = state.feedback

    currentTarget = state.target[state.target_index]

    recent_endpoints = [
        e.path for e in memory.endpoints if e.base_url == currentTarget
    ][-15:]
    memory_summary = f"""
        - Current Target: {currentTarget}
        - Scans performed on this target: {len([s for s in memory.scans_performed if s['url'] == currentTarget])}/5
        - Endpoints found for this target: {len([e for e in memory.endpoints if e.base_url == currentTarget])}
        - Recent paths: {recent_endpoints}
        - Key signals: {list(set(memory.signals))}
        """

    if state.error_detected:
        errorMsg = str(memory.last_tool_output.get("stderr", ""))[-500:]
        prompt = f"""
        [INFO]
        You are a profesional evaluator of the current state of host enumeration.
        
        [WARNING]
        During tool execution we encountered an error listed bellow:
        {errorMsg}
        
        [MEMORY SUMMARY]

        {memory_summary}
        
        [TASK]
        Your job is to do following 3 things:
            > Assign a confidence factor between 0.0 and 1.0 for the current situation.
            > Decide a feedback flag: "continue" or "error"
                * "continue" -> this flag should be prioritized as completing the job takes priority so you should prefer retrying.
                * "error" -> flag should be used in case the error is too severe to continue.
            > Give your reasoning for your decisions.
        
        Return ONLY valid JSON exactly in this format:
        {{
            "reasoning": "<Your explanation here>",
            "confidence": 0.5,
            "feedback": "<continue OR error>"
        }}
        """
    else:
        prompt = f"""
        [INFO]
        You are a profesional evaluator of the current state of host enumeration.
        
        [MEMORY SUMMARY]

        {memory_summary}
        
        [TASK]
        Your job is to do following 3 things:
            > Assign a confidence factor between 0.0 and 1.0 for the current situation.
            > Decide a feedback flag: "continue" or "done"
                * "continue" -> this flag should be used if additional scans are needed for the current situation.
                * "done" -> if all relevant information was collected.
            > Give your reasoning for your decisions.
            > AFTER 5 SCANS ON THE SAME TARGET YOU MUST RETURN FLAG "done".
                
        Return ONLY valid JSON exactly in this format:
        {{
            "reasoning": "<Your detailed reasoning here>",
            "confidence": 0.8,
            "feedback": "<continue OR done>"
        }}
        """

    feedbackFull = await llm.with_structured_output(
        toolFeedback, include_raw=True
    ).ainvoke(prompt)

    if feedbackFull["parsed"] is None:
        logData(
            "[EVALUATE NODE] -> Warning: LLM returned invalid JSON structure, forcing 'continue'"
        )
        feedback = toolFeedback(
            reasoning="Forced continue due to parsing error",
            confidence=0.5,
            feedback="continue",
        )
        feedbackRaw = feedbackFull["raw"]
    else:
        feedback = feedbackFull["parsed"]
        feedbackRaw = feedbackFull["raw"]

    if feedback.feedback:
        feedback.feedback = feedback.feedback.lower()

    if len([s for s in memory.scans_performed if s["url"] == currentTarget]) >= 5:
        feedback.feedback = "done"

    logReasoning(agentName=AGENT_NAME, reasoning=feedback.reasoning)
    logMetadata(agent_name=AGENT_NAME, metadata=feedbackRaw.response_metadata)
    state.feedback = feedback

    if feedback.feedback == "error":
        logData(
            f"[EVALUATE NODE] -> Critical error on {currentTarget}. Skipping to next."
        )
        return {
            "feedback": state.feedback,
            "target_index": state.target_index + 1,
            "decision": "continue",
            "error_detected": False,
        }

    if feedback.feedback == "done":
        logData(
            f"[EVALUATE NODE] -> Finished with {currentTarget}. Moving to next target."
        )
        return {
            "feedback": state.feedback,
            "target_index": state.target_index + 1,
            "decision": "continue",
            "error_detected": False,
        }

    return {
        "feedback": state.feedback,
        "decision": "continue",
        "error_detected": False,
    }


async def crawlerNode(state: gobusterAgentState):
    logData(message="[CRAWLER NODE] -> enter node")

    endpoints = state.memory.endpoints
    output = state.gobuster_output

    excluded = [
        ".css",
        ".js",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".log",
        ".bak",
        ".sql",
        ".conf",
        ".ini",
        ".cfg",
        ".key",
    ]

    if not endpoints:
        logData("[CRAWLER NODE] -> No endpoints found to crawl. Exiting.")
        return {
            "decision": "done",
            "gobuster_output": state.gobuster_output,
        }

    filteredEndpoints = []

    for endpoint in endpoints:
        path = endpoint.path.lower()

        filename = path.split("/")[-1]
        if filename.startswith("."):
            continue

        if any(path.endswith(ext) for ext in excluded):
            continue

        filteredEndpoints.append(endpoint)

    crawlerInput = {
        "endpoints": [e.model_dump() for e in filteredEndpoints],
    }

    print(f"[CRAWLER INPUT]:\n\n{crawlerInput}")

    logData(
        message=f"[CRAWLER NODE] -> filtered {len(endpoints)} endpoints down to {len(filteredEndpoints)}"
    )

    logData(
        message=f"[CRAWLER NODE] -> Processing {len(filteredEndpoints)} total endpoints from all targets."
    )

    try:
        crawlerOutput = await crawlerMain(payload=crawlerInput)
        output.crawler_result = crawlerOutput
        output.host_enum = crawlerInput

        logData(message="[CRAWLER NODE] -> Crawling successful for all targets.")
        return {
            "decision": "done",
            "gobuster_output": state.gobuster_output,
        }

    except Exception as e:
        state.fail_reason = f"Crawler error: {str(e)}"
        logData(message=f"[CRAWLER NODE] -> Error: {str(e)}")
        return {
            "decision": "done",
            "fail": True,
            "fail_reason": state.fail_reason,
        }


async def outputNode(state: gobusterAgentState):
    logData(message="[OUTPUT NODE] -> enter node")

    if state.fail:
        prompt = f"""
        [WARNING]
        
        You failed at performing your task.
        
        Fail reason:
        {state.fail_reason}
        
        Agent state before failure:
        {state.model_dump_json()}
        
        [TASK]
        Create a concise summary why that happened based on the above listed facts.
        
        NO MARKDOWN, NO EMOJIS!
        """

    else:
        prompt = f"""
        [TASK]
        
        Create a concise summary based on the given initial objective and gathered infromation.
        
        Objective:
        {state.objective}
        
        Crawler final result:
        {state.gobuster_output.crawler_result}
        
        NO MARKDOWN, NO EMOJIS!
        """

    logData(message="[OUTPUT NODE] -> generating summary")
    response = await llm.ainvoke(prompt)

    state.gobuster_output.summary = response.content
    logMetadata(agent_name=AGENT_NAME, metadata=response.response_metadata)
    logTotalTokens(agent_name=AGENT_NAME)

    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f" {20 * "="} GOBUSTER SUMMARY {20 * "="} ",
    )

    logReasoning(
        agentName=AGENT_NAME,
        reasoning=f"[SUMMARY] {response.content}",
    )

    if state.fail:
        logData(message="[OUTPUT NODE] -> exit node - summary done")
        return {
            "agent_output": AgentOutput(
                agent_name="gobuster",
                success=False,
                fail=True,
                fail_reason=state.fail_reason,
                summary=state.gobuster_output.summary,
            )
        }
    else:
        attack_vectors = []

        if state.gobuster_output.crawler_result:
            for vector in state.gobuster_output.crawler_result:
                attack_vectors.append(
                    attackVector(
                        endpoint=vector.get("url", ""),
                        method=vector.get("method", ""),
                        parameters=vector.get("parameters", []),
                        vector_type=vector.get("vector_type", ""),
                        confidence=vector.get("confidence", 0),
                        cookies=vector.get("cookies", {}),
                        origins=vector.get("origins", []),
                    )
                )

        logData(message="[OUTPUT NODE] -> exit node - summary done")
        return {
            "agent_output": AgentOutput(
                agent_name="gobuster",
                success=True,
                attack_vectors=attack_vectors,
                host_enum=state.gobuster_output.host_enum,
                summary=state.gobuster_output.summary,
            )
        }


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #
def setupLogger():
    logFile = logDir / f"gobuster_agent_log{logCount}.log"
    logger = logging.getLogger("gobuster_agent_log")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def logData(message: str):
    logging.getLogger("gobuster_agent_log").info(message)


def parseGobusterOutput(rawOutput):

    if isinstance(rawOutput, dict):
        stdout = rawOutput.get("stdout", "")
    elif isinstance(rawOutput, str):
        stdout = rawOutput
    else:
        stdout = str(rawOutput)

    # print(f"STDOUT:\n\n{stdout}")

    lines = stdout.split("\n")

    lines = [
        line.strip()
        for line in lines
        if line.strip() and not line.startswith("=") and "gobuster" not in line.lower()
    ]

    metadata = {}

    for line in lines:
        if line.startswith("[+]"):
            key, value = line[3:].split(":", 1)
            metadata[key.strip().lower().replace(" ", "_")] = value.strip()

    enumerateData = {}

    for line in lines:
        if "(Status:" in line and "(" in line:
            key, value = line.split("(", 1)

            path = key.strip().lower()
            if not path.startswith("/"):
                path = "/" + path

            if path not in enumerateData:
                enumerateData[path] = "(" + value.strip()

    endpoints = []

    for path, value in enumerateData.items():

        endpoint = {}

        endpoint["path"] = path

        status_match = re.search(r"Status:\s*(\d+)", value)
        endpoint["status"] = int(status_match.group(1)) if status_match else None

        size_match = re.search(r"Size:\s*(\d+)", value)
        endpoint["size"] = int(size_match.group(1)) if size_match else None

        redirect = re.search(r"\[\s*-->\s*(https?://[^\]\s]+)\s*\]", value)

        if redirect:
            endpoint["redirect"] = True
            endpoint["redirect_address"] = redirect.group(1)
        else:
            endpoint["redirect"] = False
            endpoint["redirect_address"] = None

        endpoint["type"] = classifyEndpoint(
            path=endpoint["path"],
            status=endpoint["status"],
            redirectAddress=endpoint["redirect_address"],
        )

        endpoints.append(endpoint)

    signals = {
        "has_php": any(e["path"].endswith(".php") for e in endpoints),
        "has_redirect_dirs": any(e["redirect"] for e in endpoints),
        "has_sensitive": any(
            k in e["path"]
            for e in endpoints
            for k in [".git", "config", "backup", ".bak"]
        ),
    }

    signals = [k for k, v in signals.items() if v]

    return {
        "metadata": metadata,
        "endpoints": endpoints,
        "signals": signals,
    }


def classifyEndpoint(path, status, redirectAddress):

    if redirectAddress and redirectAddress.endswith("/"):
        return "directory"

    if "." in path.split("/")[-1]:
        return "file"

    if status in [301, 302] and redirectAddress:
        return "directory"

    return "unknown"


def countData(endpoints):

    counter = Counter()

    for endpoint in endpoints:
        counter[endpoint.type] += 1

    return {
        "total_unique_endpoints": len(endpoints),
        "directories": counter.get("directory", 0),
        "files": counter.get("file", 0),
        "unknown": counter.get("unknown", 0),
    }


def retrieveCurrentDecision(state: gobusterAgentState):
    return state.decision


def extractArguments():
    allowedModes = {"dir", "dns", "vhost"}
    allowedFlags = set()

    for category, content in ALLOWED_ARGS.items():

        if isinstance(content, dict):
            for group in content.values():
                for item in group:

                    if isinstance(item, dict) and "name" in item:
                        allowedFlags.add(item["name"].split()[0])

                    elif isinstance(item, str):
                        allowedFlags.add(item.split()[0])

        elif isinstance(content, list):
            for item in content:

                if isinstance(item, dict) and "name" in item:
                    allowedFlags.add(item["name"].split()[0])

                elif isinstance(item, str):
                    allowedFlags.add(item.split()[0])

    return allowedModes, allowedFlags


def validateToolCall(toolCall: gobusterToolCall, target: str):

    allowedModes, allowedFlags = extractArguments()

    # URL check
    url = toolCall.url.strip()
    print(f"URL: {url}")
    print(f"TARGET: {target}")

    if target not in url:
        url = target

    # mode check
    mode = toolCall.mode.lower().strip()

    if mode not in allowedModes:
        mode = "dir"

    # additional args check
    args = toolCall.additional_args

    if not args:
        return gobusterToolCall(
            url=url,
            mode=mode,
            additional_args="",
            reasoning=toolCall.reasoning,
        )

    splitArgs = args.split()

    cleanArgs = []

    i = 0

    while i < len(splitArgs):
        arg = splitArgs[i]

        # remove wordlist
        if arg in ["-w", "--worldlist"]:
            i += 2
            continue

        # remove recursion
        if arg == "--recusrion":
            i += 1
            continue

        if arg.startswith("-"):

            if arg in allowedFlags:
                cleanArgs.append(arg)

                # check for values
                if i + 1 < len(splitArgs) and not splitArgs[i + 1].startswith("-"):
                    cleanArgs.append(splitArgs[i + 1])
                    i += 1

        i += 1

    finalArgs = " ".join(cleanArgs)

    return gobusterToolCall(
        url=url,
        mode=mode,
        additional_args=finalArgs,
        reasoning=toolCall.reasoning,
    )


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


def gobusterBuilder():
    setupLogger()

    workflow = StateGraph(gobusterAgentState)

    # -------------------------------
    # graph nodes
    # -------------------------------
    workflow.add_node("planning_node", planningNode)
    workflow.add_node("tool_node", toolNode)
    workflow.add_node("parse_output_node", parseOutputNode)
    workflow.add_node("evaluate_node", evaluateNode)
    workflow.add_node("crawler_node", crawlerNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------

    workflow.add_edge(START, "planning_node")
    workflow.add_conditional_edges(
        "planning_node",
        retrieveCurrentDecision,
        {
            "stop": "output_node",
            "crawler": "crawler_node",
            "continue": "tool_node",
        },
    )
    workflow.add_conditional_edges(
        "tool_node",
        retrieveCurrentDecision,
        {
            "evaluate": "evaluate_node",
            "parse": "parse_output_node",
            "plan": "planning_node",
        },
    )
    workflow.add_edge("parse_output_node", "evaluate_node")
    workflow.add_conditional_edges(
        "evaluate_node",
        retrieveCurrentDecision,
        {
            "stop": "output_node",
            "continue": "planning_node",
        },
    )
    workflow.add_edge("crawler_node", "output_node")
    workflow.add_edge("output_node", END)

    graph = workflow.compile()

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "MCP_tools/gobuster/gobuster_agent_graph.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    return graph


async def agentRunner(prompt: str):

    graph = gobusterBuilder()

    state = gobusterAgentState()
    state.objective = prompt

    result = await graph.ainvoke(state.model_dump(), {"recursion_limit": 1000})

    print(f"[FINAL RESULT]:\n\n{result}")


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "Gobuster_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    testPrompt = "Enumerate HTTP endpoints on http://192.168.157.136:8081"
    result = asyncio.run(agentRunner(prompt=testPrompt))
