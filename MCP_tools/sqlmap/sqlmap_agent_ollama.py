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
from MCP_tools.sqlmap.sqlmap_tool import sqlmap_scan, returnSqlmapToolCall

load_dotenv()
# -------------------------------------------------------------------------------#
#                                  LLM setup                                     #
# -------------------------------------------------------------------------------#

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")

llm = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b",
    base_url=LM_API,
    temperature=0.2,
    format=None,
)
# TODO: check the temp
finalAgent = llm.bind_tools([sqlmap_scan])


summarizerAgent = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b",
    base_url=LM_API,
    temperature=0.0,
    format=None,
)


# -------------------------------------------------------------------------------#
#                                  Agent setup                                   #
# -------------------------------------------------------------------------------#

# TODO: move context into a seperate file
context = """
You are SQLMAP-AGENT, a deterministic sub-agent.
Your only tool is "sqlmap_scan".
Your job: methodically assess the assigned web target for SQL injection vulnerabilities and extract meaningful results.

OBJECTIVE:
Perform a multi-stage SQL injection assessment using sqlmap.
Progress from low-impact detection to higher-impact exploitation only when justified by evidence.

SCAN STAGES (in order):

STAGE 1 - INITIAL DETECTION:
- Perform a basic sqlmap scan to detect possible injection points.
- Use minimal flags (e.g. --batch, --level=1, --risk=1).
- Identify injectable parameters, DBMS type, and basic injection techniques.

STAGE 2 - CONFIRMATION & ENUMERATION:
- Only if injection is detected.
- Increase confidence and detail using moderate flags.
- Enumerate:
  - Current database
  - Available databases
  - Basic DBMS information
- Typical flags: --dbs, --current-db, --level=2-3, --risk=2

STAGE 3 - TARGETED EXTRACTION:
- Only if databases or tables are confirmed.
- Enumerate tables and columns for relevant databases.
- Avoid dumping everything blindly.
- Typical flags: --tables, --columns, --dump with specific tables.

STAGE 4 - ADVANCED / AGGRESSIVE:
- Attempt advanced features only if:
  - Injection is stable
  - DBMS and permissions allow it
- Examples:
  - --os-shell
  - --file-read / --file-write
  - Privilege escalation features

NEVER skip stages.
NEVER repeat the exact same sqlmap command.

OUTPUT RULES:
Each step outputs exactly ONE of:
1. CALL_TOOL: <JSON>
2. FINAL_ANSWER: <text>
3. ERROR: <text>

No markdown, no emojis, no explanations outside FINAL_ANSWER.

TOOL CALL FORMAT (exact):
CALL_TOOL: {
  "tool": "sqlmap_scan",
  "args": {
    "url": "<target URL>",
    "data": "<POST data or empty>",
    "additional_args": "<sqlmap arguments>"
  }
}

BEHAVIOR RULES:
- Stay strictly within the assigned target URL.
- Never invent parameters, databases, tables, or vulnerabilities.
- Base every next scan decision on memory and last tool output.
- Prefer incremental, controlled scans over aggressive one-shot scans.
- Do not dump large datasets unless explicitly relevant.
- Stop scanning when:
  - No injection is found, OR
  - The vulnerability has been sufficiently characterized.

MEMORY:
SYSTEM MESSAGE may contain:
- YOUR TASK
- CURRENT STATE
- LAST TOOL CALL
- LAST FORMATTED TOOL OUTPUT

If a field is missing, ignore it.
Never recreate or hallucinate missing memory.

ERROR HANDLING:
- If sqlmap fails, adjust arguments and retry once.
- If repeated failure occurs, stop and report in FINAL_ANSWER.

FINAL REPORT:
When finished, output FINAL_ANSWER summarizing:
- Whether injection was found
- Injection type and affected parameters
- DBMS identified
- Extracted data (high-level, not raw dumps)
- Security impact assessment

No markdown.
No JSON except inside CALL_TOOL.

===========================
       END OF CONTEXT
===========================
"""


summarizeContext = """
You are SQLMAP-SUMMARIZER, a minimal deterministic assistant.
Your task is to summarize raw sqlmap tool output.

OBJECTIVE:
Extract only actionable, high-signal information from sqlmap output.
Remove all noise, banners, progress logs, and repeated text.

FOCUS ON:
- Whether SQL injection was detected (YES/NO)
- Affected parameter(s)
- Injection technique(s)
- Identified DBMS and version (if available)
- Discovered databases, tables, or columns
- Any errors or limitations reported by sqlmap

IGNORE:
- ASCII banners
- Progress indicators
- Repeated status messages
- Explanations, advice, or warnings unless critical

OUTPUT RULES:
- Output plain text only.
- No markdown.
- No bullet symbols.
- No emojis.
- Be concise and factual.
- Do not speculate or infer beyond the tool output.

If no useful information is present, output:
"No actionable findings."

DO NOT include sqlmap command suggestions.
DO NOT include mitigation advice.

===========================
       END OF CONTEXT
===========================
"""

# -------------------------------------------------------------------------------#
#                                 Custom agent state                             #
# -------------------------------------------------------------------------------#


# basic structure idea
class customAgentState(BaseModel):

    target: Optional[str] = Field(
        default=None, description="Host target given by the orchestrator."
    )

    phase: str = Field(default="init", description="Current sqlmap assessment phase.")

    memory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Dictionary with organized results of previous scans.",
    )


# SQLmap assessments phases (state machine):
#   > Init
#   > Detect
#   > Enum
#   > Extract
#   > Done


# -------------------------------------------------------------------------------#
#                                Agent definition                                #
# -------------------------------------------------------------------------------#


@task
async def callModel(
    messages: List[BaseMessage],
    customAgentState: customAgentState,
    toolResult=None,
):

    state_snapshot = json.dumps(
        {
            "target": customAgentState.target,
            "phase": customAgentState.phase,
            "memory": customAgentState.memory,
        }
    )

    lastToolCall = await returnSqlmapToolCall(mode="read")

    if toolResult:
        customMessage = f"""
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        
        LAST TOOL CALL:
        {lastToolCall}
        
        LAST FORMATED TOOL OUTPUT:
        {toolResult}        
        """
    else:
        customMessage = f"""
        
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        """

    print("\n\n============= AGENT_STATE DUMP =============")
    print(
        json.dumps(
            {
                "target": customAgentState.target,
                "phase": customAgentState.phase,
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
        rawOutput = await sqlmap_scan.arun(tool_call["args"])
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }

    return ToolMessage(
        content=rawOutput, name="sqlmap_scan", tool_call_id=tool_call["id"]
    )


@task
async def summarizeToolOutput(toolMessage: ToolMessage):
    content = toolMessage.content
    lastToolCall = await returnSqlmapToolCall(mode="read")

    customMessage = f"""
    YOUR TASK:
    Create a concise summary of the last tool output.
    
    IMPORTANT:
    Append one or more TAGS at the end of the summary:
    
    Allowed TAGS (uppercase, exact match):
    
    - TAG:INJECTABLE
    - TAG:NOT_INJECTABLE
    - TAG:DB_ENUM_AVAILABLE
    - TAG:DATA_EXTRACTED
    - TAG:ERROR
    
    
    Only include a TAG if it is explicitly supported by the tool output.
    Do NOT infer.
    Do NOT invent tags.
    
    LAST TOOL CALL:
    {lastToolCall}
    
    LAST TOOL OUTPUT:
    {content}
    """

    return await summarizerAgent.ainvoke(
        [SystemMessage(content=summarizeContext), SystemMessage(content=customMessage)]
    )


@task
async def updateState(toolOutput: BaseMessage, customAgentState: customAgentState):
    toolSummary = getattr(toolOutput, "content", None).strip()
    lastToolCall = await returnSqlmapToolCall(mode="read")

    if not toolSummary:
        return

    target = customAgentState.target

    if not target:
        target = "unknown"

    print("\n" + "=" * 40)
    print(f"\nSUMMARY PASSED TO AGENT\n\n{toolSummary}")
    print("\n" + "=" * 40)

    customAgentState.memory.append(
        {
            "iteration": len(customAgentState.memory) + 1,
            "target": target,
            "phase": customAgentState.phase,
            "tool_call": lastToolCall,
            "summary": toolSummary,
        }
    )

    addedTag = re.findall(r"TAG:[A-Z_]+", toolSummary)

    if customAgentState.phase == "init":
        customAgentState.phase = "detect"

    if customAgentState.phase == "detect":
        if "SIGNAL:INJECTABLE" in addedTag:
            customAgentState.phase = "enum"
        elif "SIGNAL:NOT_INJECTABLE" in addedTag:
            customAgentState.phase = "done"

    elif customAgentState.phase == "enum":
        customAgentState.phase = "extract"

    elif customAgentState.phase == "extract":
        if "SIGNAL:DATA_EXTRACTED" in addedTag:
            customAgentState.phase = "done"


@entrypoint()
async def agent(message: list[BaseMessage]):
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
            final_answer = getattr(response, "content", None)

            if (
                not final_answer.strip().startswith("CALL_TOOL:")
                and final_answer.strip() != ""
            ):
                result = agentFinalOutput(
                    agent_finished=True,
                    agent_report=final_answer,
                )
                return {"final_result": result}

        rawToolResult = await callTool(response.tool_calls)

        summarizedToolResult = await summarizeToolOutput(toolMessage=rawToolResult)

        await updateState(toolOutput=summarizedToolResult, customAgentState=agent_state)

        response = await callModel(
            message, customAgentState=agent_state, toolResult=summarizedToolResult
        )


# -------------------------------------------------------------------------------#
#                                  Agent runner                                  #
# -------------------------------------------------------------------------------#


async def agentRunner(message):
    response = await agent.ainvoke(input=message, config={"recursion_limit": 40})

    try:
        if isinstance(response, dict):
            finalAgentOutput = response.get("final_result")
        else:
            finalAgentOutput = getattr(response, "final_result", None)

        if finalAgentOutput:
            print("\n" + "=" * 80)
            print("FINAL OUTPUT DEBUG\n\n")
            print(finalAgentOutput)
            print("\n" + "=" * 80)
            return finalAgentOutput
        else:
            return "Tool agent didn't return anything!"

    except Exception as e:
        print(f"Error: {str(e)}")


# -------------------------------------------------------------------------------#
#                               Output formatting                                #
# -------------------------------------------------------------------------------#


class agentFinalOutput(BaseModel):
    agent_finished: bool = False
    agent_report: str = Field(
        default_factory=str, description="Final report written by the tool agent."
    )


# -------------------------------------------------------------------------------#
#                                   Main loop                                    #
# -------------------------------------------------------------------------------#

# only for debugging
if __name__ == "__main__":
    print("#" + "-" * 10 + "SQLmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    while True:
        supervisorInput = input("\nUser: ").strip()

        if supervisorInput.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        message = [HumanMessage(content=supervisorInput)]

        asyncio.run(agentRunner(message=message))
