import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import json
import re
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import Field, BaseModel
from langchain.messages import SystemMessage, ToolCall, ToolMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from MCP_tools.gobuster.gobuster_tool import gobuster_scan, returnGobusterToolCall
from datetime import datetime
from collections import Counter

load_dotenv()

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


# ------------------------------------------------------------------------------- #
#                                  Agent setup                                    #
# ------------------------------------------------------------------------------- #

# TODO: read it through and adjust some stuff
context = """
You are GOBUSTER-AGENT, a deterministic sub-agent.
Your only tool is "gobuster_scan".
Your job is to enumerate HTTP endpoints on the assigned target.

OBJECTIVE:
1. Receive exactly one HTTP target (URL).
2. Perform directory and file enumeration using controlled gobuster scans.
3. Discover:
   - existing paths
   - HTTP status codes
   - response sizes
   - redirects
   - directory vs file hints

SCOPE RULES:
- Operate ONLY on the provided target URL.
- Never modify protocol, host, port, or base path.
- Never brute-force parameters, forms, or credentials.
- Never attempt authentication, login bypass, or exploitation.
- Enumeration only.

TOOL USAGE RULES:
- Use ONLY the "gobuster_scan" tool.
- Default mode is "dir".
- Use the default wordlist unless explicitly instructed otherwise.
- Do NOT invent arguments.
- Do NOT repeat an identical tool call.

TOOL CALL FORMAT (exact):
CALL_TOOL: {
  "tool": "gobuster_scan",
  "args": {
    "url": "<target url>",
    "mode": "dir",
    "additional_args": "<optional gobuster flags or empty string>"
  }
}

ENUMERATION STRATEGY:
1. Always begin with ONE baseline directory enumeration run.
2. After the first successful run, inspect CURRENT STATE memory.
3. You MAY perform additional scans ONLY if signals exist.

SIGNALS:
- A discovered endpoint ends with ".php"
- A discovered endpoint has redirect=true and type=directory
- Sensitive indicators exist (e.g. ".git", "config", "backup", ".bak")

ALLOWED FOLLOW-UP SCANS:
Choose at most TWO of the following:
- Add extensions: "-x php,txt,bak,conf"
- Force directory handling: "-f"
- Exclude common noise by status or size if needed

LIMITS:
- Maximum total tool calls: 3
- Never recurse into discovered paths
- Never scan subdirectories individually
- Never change the target URL

OUTPUT RULES:
Each response MUST be exactly ONE of:
1. CALL_TOOL: <JSON>
2. FINAL_ANSWER: <plain text>
3. ERROR: <plain text>

No markdown.
No emojis.
No explanations.
No extra formatting.

BREAK / TERMINATION RULE:
Stop execution and return FINAL_ANSWER when ANY of the following is true:
- Maximum tool call limit is reached
- No new meaningful endpoints are discovered
- No valid signals exist after a scan
- Enumeration goals are satisfied

IMPORTANT:
FINAL_ANSWER must be plain text and MUST NOT start with "CALL_TOOL:".
This output signals the supervisor to stop execution.

BEHAVIOR RULES:
- Never hallucinate endpoints.
- Never guess paths.
- Never summarize results in text.
- Trust tool output as ground truth.
- Re-evaluate memory after each scan before deciding to continue.

MEMORY HANDLING:
SYSTEM MESSAGE may contain:
- YOUR TASK
- CURRENT STATE
- LAST TOOL CALL
- LAST TOOL OUTPUT

If a field is missing, ignore it.
Never recreate or overwrite memory manually.

ERROR HANDLING:
- On tool failure caused by wildcard responses or server behavior:
  - Retry ONCE with adjusted arguments.
- If failure persists:
  - Return ERROR with a short explanation.

FINAL OUTPUT:
When finished, return exactly:
FINAL_ANSWER: Enumeration completed.

===========================
       END OF CONTEXT
===========================

"""


# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class customAgentState(BaseModel):
    target: Optional[str] = Field(
        default=None, description="Host target given by the orchestrator."
    )

    memory: List[Dict[str, Any]] = Field(
        default_factory=list, description="List with tool results."
    )


# ------------------------------------------------------------------------------- #
#                                Agent definition                                 #
# ------------------------------------------------------------------------------- #


@task
async def callModel(
    messages: List[BaseMessage], customAgentState: customAgentState, toolOutput: None
):

    state_snapshot = json.dumps(
        {
            "target": customAgentState.target,
            "memory": customAgentState.memory,
        }
    )

    lastToolCall = await returnGobusterToolCall(mode="read")

    if toolOutput:
        customMessage = f"""
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        
        LAST TOOL CALL:
        {lastToolCall}
        
        LAST TOOL OUTPUT:
        {toolOutput}
        """
    else:
        customMessage = f"""
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        """

    # agent state dump
    print("\n\n============= AGENT_STATE DUMP =============")
    print(
        json.dumps(
            {
                "target": customAgentState.target,
                "memory": customAgentState.memory,
            },
            indent=4,
            ensure_ascii=False,
        )
    )
    print("============================================\n\n")

    return await finalAgent.ainvoke(
        [SystemMessage(content=context), SystemMessage(content=customMessage)],
        config={"recursion_limit": 40},
    )


@task
async def callTool(toolCall: List[ToolCall]):
    tool_call = toolCall[0]

    try:
        rawOutput = await gobuster_scan.arun(tool_call["args"])
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }

    return ToolMessage(
        content=rawOutput, name="gobuster_scan", tool_call_id=tool_call["id"]
    )


@task
async def updateState(toolOutput: ToolMessage, customAgentState: customAgentState):

    # TODO: double check this
    if isinstance(toolOutput.content, dict):
        toolSplitLines = toolOutput.content.get("stdout", "")
    else:
        toolContent = toolOutput.content
        toolSplitLines = toolContent[0].split("stdout")

    toolSplitLinesTemp = toolSplitLines[1].split("\\n")
    metadata = {}

    for line in toolSplitLinesTemp:
        line = line.strip("\\")
        if line.startswith("[+]"):
            key, value = line[3:].split(":", 1)
            metadata[key.strip().lower().replace(" ", "_")] = value.strip()

    enumerateData = {}
    for line in toolSplitLinesTemp:
        line = line.strip("\\")

        if line.startswith("/"):
            key, value = line.split("(", 1)
            enumerateData[key.strip().lower()] = f"({value.strip()}"

    memory = {}

    toolArgs = await returnGobusterToolCall("read")

    memory["tool"] = "gobuster_scan"
    memory["tool_args"] = toolArgs
    memory["metadata"] = metadata
    memory["endpoint"] = []

    for key in enumerateData:
        # /.hta -> key
        endpoint = {}
        value = enumerateData[key]

        endpoint["path"] = key

        status_match = re.search(r"Status:\s*(\d+)", value)
        if status_match:
            endpoint["status"] = int(status_match.group(1))
        else:
            endpoint["status"] = None

        size_match = re.search(r"Size:\s*(\d+)", value)
        if size_match:
            endpoint["size"] = int(size_match.group(1))
        else:
            endpoint["size"] = None

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

        memory["endpoint"].append(endpoint)

        memory["signals"] = {
            "has_php": any(e["path"].endswith(".php") for e in memory["endpoint"]),
            "has_redirect_dirs": any(e["redirect"] for e in memory["endpoint"]),
            "has_sensitive": any(
                k in e["path"]
                for e in memory["endpoint"]
                for k in [".git", "config", "backup", ".bak"]
            ),
        }

    memory["timestamp"] = datetime.now().isoformat()
    print("\n\n" + "=" * 40 + "\n" + "Final memory data output:\n")
    print(memory)

    customAgentState.memory.append(memory)

    return


@entrypoint()
async def agent(message: List[BaseMessage]):
    agent_state = customAgentState()

    # get target
    text = message[-1].content.lower()
    url_match = re.search(r"https?://[^\s]+", text)

    if url_match:
        url = url_match.group(0)
    else:
        url = "Unknown target."

    agent_state.target = url

    # initial invoke
    response = await callModel(message, customAgentState=agent_state, toolOutput=None)

    while True:
        if not response.tool_calls:
            return_answer = getattr(response, "content", None)
            if (
                not return_answer.strip().startswith("CALL_TOOL:")
                and return_answer.strip() != ""
            ):

                return formatAgentOutput(agentState=agent_state)

        toolOutput = await callTool(response.tool_calls)

        await updateState(toolOutput=toolOutput, customAgentState=agent_state)

        response = await callModel(
            message, customAgentState=agent_state, toolOutput=toolOutput
        )


# ------------------------------------------------------------------------------- #
#                                Helper functions                                 #
# ------------------------------------------------------------------------------- #


def classifyEndpoint(path, status, redirectAddress):

    if redirectAddress and redirectAddress.endswith("/"):
        return "directory"

    if "." in path.split("/")[-1]:
        return "file"

    if status in [301, 302] and redirectAddress:
        return "directory"

    return "unknown"


def formatAgentOutput(agentState):
    agentOutput = agentFinalOutput()

    agentOutput.target = agentState.target
    agentMemory = agentState.memory

    # get all endpoints from agent state
    allEndpoints = []
    for scan in agentMemory:
        allEndpoints.extend(scan.get("endpoint", []))

    # deduplicate endpoints
    cleanedEndpoints = {}
    for endpoint in allEndpoints:
        path = endpoint.get("path")
        if path:
            cleanedEndpoints[path] = endpoint

    uniqueEndpoints = list(cleanedEndpoints.values())

    agentOutput.summary = createSummary(uniqueEndpoints=uniqueEndpoints)
    agentOutput.endpoints = uniqueEndpoints

    allSignals = []
    for scan in agentMemory:
        allSignals.extend(scan.get("signals", []))

    print("\n\n============= AGENT_SIGNALS DUMP =============")
    print(f"All signals: {allSignals}")
    print("\n\n==============================================")

    allSignals = list(dict.fromkeys(allSignals))
    agentOutput.signals = allSignals
    agentOutput.finished = True

    return agentOutput


def createSummary(uniqueEndpoints):

    counter = Counter()

    for endpoint in uniqueEndpoints:
        endpointType = endpoint.get("type", "unknown")
        counter[endpointType] += 1

    # you have 3 main types in agent state: directory, file and unknown for everything else

    summary = {
        "total_unique_endpoints": len(uniqueEndpoints),
        "directories": counter.get("directory", 0),
        "files": counter.get("file", 0),
        "unknown": counter.get("unknown", 0),
    }

    return summary


# ------------------------------------------------------------------------------- #
#                                 Agent runners                                   #
# ------------------------------------------------------------------------------- #


async def agentRunner(message):

    response = await agent.ainvoke(input=message, config={"recursion_limit": 40})

    if response:
        print("\n" + "=" * 80)
        print("FINAL OUTPUT DEBUG\n\n")
        print(response)
        print("\n" + "=" * 80)
    else:
        return "Tool agent didn't return anything!"


# ------------------------------------------------------------------------------- #
#                               Output formatting                                 #
# ------------------------------------------------------------------------------- #


class agentFinalOutput(BaseModel):
    agent: str = "gobuster"
    target: str = Field(
        default_factory=str, description="Target givin by the orchestrator."
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary containing total number of endpoints, directories and files.",
    )
    endpoints: List[str] = Field(
        default_factory=list, description="List of discovered endpoints."
    )
    signals: List[str] = Field(default_factory=list, description="List of signals.")
    finished: bool = False


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "Gobuster_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    while True:
        supervisorInput = input("\n> User: ").strip()

        if supervisorInput.lower() in ["quit", "exit"]:
            print("Ending...")
            break

        message = [HumanMessage(content=supervisorInput)]

        asyncio.run(agentRunner(message=message))
