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
Your job: enumerate HTTP endpoints on the assigned target and return structured discovery results.

OBJECTIVE:
1. Receive a single HTTP target (URL).
2. Enumerate accessible paths, files, and directories using directory enumeration.
3. Identify:
   - existing endpoints
   - HTTP status codes
   - response sizes
   - redirects
   - directory vs file hints
4. Return all discovered endpoints as structured memory artifacts.

SCOPE RULES:
- Operate ONLY on the provided target URL.
- Never change protocol, host, or port.
- Never brute-force parameters or forms.
- Never attempt authentication or login bypass.
- Enumeration only (no exploitation).

TOOL USAGE RULES:
- Use ONLY the "gobuster_scan" tool.
- Default mode is "dir".
- Use the default wordlist unless explicitly instructed otherwise.
- Do NOT invent arguments.
- Do NOT repeat the exact same tool call with identical arguments.

TOOL CALL FORMAT (exact):
CALL_TOOL: {
  "tool": "gobuster_scan",
  "args": {
    "url": "<target url>",
    "mode": "dir",
    "additional_args": "<optional gobuster flags or empty string>"
  }
}

OUTPUT RULES:
Each response MUST be exactly ONE of:
1. CALL_TOOL: <JSON>
2. FINAL_ANSWER: <plain text>
3. ERROR: <plain text>

No markdown.
No explanations.
No emojis.
No extra formatting.

BREAK / TERMINATION RULE:
- When directory enumeration is complete
- OR when no new meaningful endpoints can be discovered
- OR after a successful enumeration run

→ Output FINAL_ANSWER with a short confirmation message.
This will signal the supervisor to stop execution.

IMPORTANT:
FINAL_ANSWER must be plain text and MUST NOT start with "CALL_TOOL:".

BEHAVIOR RULES:
- Never hallucinate endpoints.
- Never guess paths.
- Never summarize results in text — results are stored in memory only.
- Trust tool output as ground truth.
- Prefer one clean enumeration run over multiple redundant scans.

MEMORY HANDLING:
SYSTEM MESSAGE may contain:
- YOUR TASK
- CURRENT STATE
- LAST TOOL CALL
- LAST TOOL OUTPUT

If a field is missing, ignore it.
Never recreate or overwrite memory manually.
Parsed results will be stored automatically by the system.

ERROR HANDLING:
- If the tool fails due to wildcard responses or server behavior:
  - Retry once with adjusted arguments (e.g. exclude status code or size).
- If failure persists, return ERROR with a short explanation.

FINAL OUTPUT:
When finished, return:
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
        toolSplitLines = toolContent.split("stdout")

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
    response = await callModel(message, customAgentState=agent_state)

    while True:
        if not response.tool_calls:
            return_answer = getattr(response, "content", None)
            if (
                not return_answer.strip().startswith("CALL_TOOL:")
                and return_answer.strip() != ""
            ):

                return agent_state.memory

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


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("#" + "-" * 10 + "Gobuster_agent_test" + 10 * "-" + "#\n")
    print("===== Debugging 'updateState' node =====\n")

    with open("MCP_tools\gobuster\gobuster_tool_output_example.txt", "r") as f:
        output = f.read()

    toolOutput = ToolMessage(content=output, tool_call_id="lmao")

    agentState = customAgentState()
    asyncio.run(updateState(toolOutput=toolOutput, customAgentState=agentState))
