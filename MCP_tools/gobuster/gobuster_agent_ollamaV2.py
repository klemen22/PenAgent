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

load_dotenv()

from MCP_tools.gobuster.gobuster_toolV2 import gobuster_scan, gobusterInput
from MCP_tools.gobuster.crawler import main as crawlerMain

# TODO: Create arguments and URL filter for consistency and reinforce communication between planning and evaluate nodes

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

finalAgent = llm.bind_tools([gobuster_scan])

logDir = Path("MCP_tools/gobuster/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1

with open("MCP_tools/gobuster/gobuster_allowed_arguments.json") as f:
    ALLOWED_ARGS = f.read()


# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class gobusterEndpoint(BaseModel):
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
    url: str = Field(default="")
    mode: str = Field(default="dir")
    additional_args: str = Field(default="")
    reasoning: Optional[str] = Field(default=None)


class toolFeedback(BaseModel):
    confidence: float = Field(default=0.0, description="Number between 0.0 - 1.0")
    feedback: Optional[Literal["continue", "done", "error"]] = Field(default=None)
    reasoning: Optional[str] = Field(default=None)


class gobusterOutput(BaseModel):
    summary: Optional[str] = Field(default=None)
    crawler_result: Optional[Any] = Field(default=None)


class gobusterAgentState(BaseModel):
    objective: str = Field(
        default="", description="Objective given by the orchestartor"
    )
    target: Optional[str] = Field(default=None)

    tool_call: gobusterToolCall = Field(default_factory=gobusterToolCall)

    memory: gobusterMemory = Field(default_factory=gobusterMemory)

    iteration: int = Field(default=0)
    max_iterations: int = Field(default=10)

    decision: Optional[
        Literal["continue", "stop", "crawler", "done", "evaluate", "parse"]
    ] = Field(default=None)
    feedback: toolFeedback = Field(default_factory=toolFeedback)
    error_detected: bool = Field(
        default=False,
        description="A flag for seperating normal and error based feedback.",
    )

    # output
    gobuster_output: gobusterOutput = Field(default_factory=gobusterOutput)

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

    # save target
    if not state.target:

        url_match = re.search(r"https?://[^\s]+", state.objective)

        if url_match:
            target = url_match.group(0)

        else:
            ip_match = re.search(
                r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?\b", state.objective
            )

            if ip_match:
                target = ip_match.group(0)

                # auto-add http if missing
                if not target.startswith("http"):
                    target = f"http://{target}"

            else:
                domain_match = re.search(
                    r"\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?::\d{1,5})?\b", state.objective
                )

                if domain_match:
                    target = domain_match.group(0)

                    if not target.startswith("http"):
                        target = f"http://{target}"

                else:
                    target = None

        if target:
            state.target = target

        state.iteration += 1

    # check itterations
    if state.iteration >= state.max_iterations:
        state.fail_reason = f"Max number of iterations reached (number of iterations: {state.iteration})"
        logData(
            message="[PLANNING NODE] -> maximum number of iterations reached -> moving to output node"
        )

        return {
            "fail": True,
            "fail_reason": state.fail_reason,
            "iteration": state.iteration,
            "decision": "stop",
        }

    if state.feedback.feedback is not None:
        if feedback.feedback == "done":
            return {
                "decision": "crawler",
                "iteration": state.iteration,
            }

        if feedback.feedback == "continue":

            if state.error_detected:
                prompt = f"""
                You are a Gobuster enumeration agent.
                
                Objective:
                {state.objective}
                
                Target:
                {state.target}
                
                [WARNING]
                
                Last tool call has encountered an error.
                
                Feedback:
                {feedback.reasoning}
                
                [MEMORY]

                Already known facts stored in memory:
                {memory if memory else "None"}
                
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
                    "url": <a valid target URL>
                    "mode": <gobuster mode -> default value is set to "dir">
                    "additional_args": <A combination of additional gobuster arguments - USE ONLY ARGUMENTS PROVIDED ABOVE>
                    "reasoning": <Your reasoning for this tool call and arguments combination>
                }}
                
                """

                # reset error flag
                state.error_detected = False
            else:
                prompt = f"""
                You are a Gobuster enumeration agent.
                
                Objective:
                {state.objective}
                
                Target:
                {state.target}
                
                [FEEDBACK]
                
                From feedback it was decided that additional scans are required.
                
                Last scan performed:
                {state.memory.scans_performed[-1]}
                
                Confidence from last scan:
                {feedback.confidence}
                
                Reasoning for confidence:
                {feedback.reasoning}
                
                [MEMORY]
                
                Already known facts stored in memory:
                {memory if memory else "None"}
                
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
                    "url": <a valid target URL>
                    "mode": <gobuster mode -> default value is set to "dir">
                    "additional_args": <A combination of additional gobuster arguments - USE ONLY ARGUMENTS PROVIDED ABOVE>
                    "reasoning": <Your reasoning for this tool call and arguments combination>
                }}
                """

    else:
        # initial prompt
        prompt = f"""
        You are a Gobuster enumeration agent.
        
        Objective:
        {state.objective}
        
        Target:
        {state.target}
        
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
            "url": <a valid target URL>
            "mode": <gobuster mode -> default value is set to "dir">
            "additional_args": <A combination of additional gobuster arguments - USE ONLY ARGUMENTS PROVIDED ABOVE>
            "reasoning": <Your reasoning for this tool call and arguments combination>
        }}

        """

    retries = 2
    finalPrompt = prompt

    for attempt in range(retries):
        try:
            output = await llm.with_structured_output(gobusterToolCall).ainvoke(
                finalPrompt
            )
            logData(message=f"[PLANNING NODE] -> created next tool call: {output}")
            state.tool_call = output
            logData(message="[PLANNING NODE] -> exit node")

            return {
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
        "fail": True,
        "fail_reason": state.fail_reason,
        "iteration": state.iteration,
        "decision": "stop",
    }


async def toolNode(state: gobusterAgentState):
    logData(message="[TOOL NODE] -> enter node")

    # TODO: create tool arg filter and fallback

    toolCall = state.tool_call
    memory = state.memory

    if not toolCall:
        return {
            "decision": "evaluate",
            "error_detected": True,
        }

    memory.scans_performed.append(
        {
            "url": state.target,
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
        logData(message="[TOOL NODE] -> Execption occured!")
        memory.last_tool_output = rawOutput

        return {
            "memory": state.memory,
            "decision": "evaluate",
            "error_detected": True,
        }

    if isinstance(rawOutput, tuple):
        rawOutput = rawOutput[1]["result"]

    memory.last_tool_output = rawOutput
    if rawOutput.get("stderr"):
        logData(
            f"[TOOL NODE] -> tool encountered following error: {str(rawOutput.get("stderr"))}"
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

    rawOutput = state.memory.last_tool_output

    if not rawOutput:
        return {"decision": "evaluate"}

    parsed = parseGobusterOutput(rawOutput)

    endpoints = parsed["endpoints"]

    state.memory.endpoints.extend([gobusterEndpoint(**e) for e in endpoints])

    state.memory.signals.extend(parsed["signals"])

    return {
        "memory": state.memory,
        "decision": "evaluate",
    }


async def evaluateNode(state: gobusterAgentState):
    logData(message="[EVALUATE NODE] -> enter node")
    memory = state.memory
    feedback = state.feedback

    if state.error_detected:
        prompt = f"""
        [INFO]
        You are a profesional evaluator of the current state of host enumeration.
        
        [WARNING]
        During tool execution we encountered an error listed bellow:
        {memory.last_tool_output}
        
        [TASK]
        Your job is to do following 3 things:
            > Assign a confidence factor between 0.0 and 1.0 for the current situation.
            > Decide a feedback flag: "continue" or "error"
                * "continue" -> this flag should be prioritized as completing the job takes priority so you should prefer retrying.
                * "error" -> flag should be used in case the error is to sever to continue (try to avoid this flag).
            > Give your reasoning for your decisions.
        
        Return a valid JSON:
        - confidence
        - feedback
        - reasoning
        """
    else:
        prompt = f"""
        [INFO]
        You are a profesional evaluator of the current state of host enumeration.
        
        [MEMORY]
        {memory}
        
        [TASK]
        Your job is to do following 3 things:
            > Assign a confidence factor between 0.0 and 1.0 for the current situation.
            > Decide a feedback flag: "continue", "done" or "error"
                * "continue" -> this flag should be used if additional scans are needed for the current situation.
                * "done" -> if all relevant information was collected.
            > Give your reasoning for your decisions.
                
        Return a valid JSON:
        - confidence
        - feedback
        - reasoning
        """

    feedback = await llm.with_structured_output(toolFeedback).ainvoke(prompt)
    state.feedback = feedback

    if feedback.feedback == "error":
        state.fail_reason = f"""
        Error encountered was too severe.
        
        Reasoning:
        {feedback.reasoning} 
        """

        return {
            "feedback": state.feedback,
            "fail": True,
            "fail_reason": state.fail_reason,
            "decision": "stop",
        }

    else:
        return {
            "feedback": state.feedback,
            "decision": "continue",
        }


async def crawlerNode(state: gobusterAgentState):
    logData(message="[CRAWLER NODE] -> enter node")

    endpoints = state.memory.endpoints
    # count = countData(endpoints=endpoints)
    output = state.gobuster_output

    target = state.target

    if not target:
        match = re.search(r"https?://[^\s]+", state.objective)
        if match:
            target = match.group(0)

        else:
            raise ValueError("Crawler cannot run because target is None")

    crawlerInput = {
        "target": target,
        "endpoints": [e.model_dump() for e in endpoints],
    }
    logData(message=f"[CRAWLER NODE] -> initial endpoints:\n\n{endpoints}")
    logData(message=f"[CRAWLER NODE] -> crawler input:\n\n{crawlerInput}")
    logData(message="[CRAWLER NODE] -> starting crawler...")

    try:

        crawlerOutput = await crawlerMain(payload=crawlerInput)
        output.crawler_result = crawlerOutput

        logData(message="[CRAWLER NODE] -> crawling successful -> exit node")
        return {
            "decision": "done",
            "gobuster_output": state.gobuster_output,
        }

    except Exception as e:
        state.fail_reason = f"""
        Crawler encountered following error:
        
        {str(e)}
        """
        logData(
            message=f"[CRAWLER NODE] -> crawler encountered following error:{str(e)}"
        )
        logData(message="[CRAWLER NODE] -> crawling failed -> exit node")
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
        """

    else:
        prompt = f"""
        [TASK]
        
        Create a concise summary based on the given initial objective and gathered infromation.
        
        Objective:
        {state.objective}
        
        Crawler final result:
        {state.gobuster_output.crawler_result}
        """

    logData(message="[OUTPUT NODE] -> generating summary")
    response = await llm.ainvoke(prompt)
    state.gobuster_output.summary = response.content

    logData(message="[OUTPUT NODE] -> exit node - summary done")
    return {
        "gobuster_output": state.gobuster_output,
    }


# ------------------------------------------------------------------------------- #
#                                 Helper fucntion                                 #
# ------------------------------------------------------------------------------- #
def setupLogger():
    logFile = logDir / f"gobuster_agent_log{logCount}.log"
    logger = logging.getLogger("gobuster_agent")
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logFile, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def logData(message: str):
    logging.getLogger("gobuster_agent").info(message)


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

    # print(f"FINAL ENDPOINTS:\n\n{endpoints}")
    # print(f"FINAL METADATA:\n\n{metadata}")
    # print(f"FINAL SIGNALS:\n\n{signals}")

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


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


async def agentRunner(prompt):
    agentState = gobusterAgentState()
    setupLogger()

    workflow = StateGraph(gobusterAgentState)
    SESSIN_ID = "default_session"

    # test prompt placeholder
    agentState.objective = prompt

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

    result = await graph.ainvoke(
        agentState.model_dump(),
        config={"thread_id": SESSIN_ID, "recursion_limit": 1000},
    )

    return {
        "summary": result["gobuster_output"].summary,
        "attack_vectors": result["gobuster_output"].crawler_result,
    }


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "Gobuster_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    testPrompt = "Enumerate HTTP endpoints on http://192.168.157.136:8081"
    result = asyncio.run(agentRunner(prompt=testPrompt))

    print(f"[FINAL RESULT]:\n\n{result}")
