import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import json
import re
import asyncio
from typing import Dict, Any, List
from pydantic import Field, BaseModel
from langchain.messages import SystemMessage, ToolCall, ToolMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task

# temp solution for importing bullshit when calling this .py file
try:
    from MCP_tools.nmap_tool import nmap_scan, returnToolCall
except Exception:
    from nmap_tool import nmap_scan, returnToolCall


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

finalAgent = llm.bind_tools([nmap_scan])

# -------------------------------------------------------------------------------#
#                                  Agent setup                                   #
# -------------------------------------------------------------------------------#

# Intial context
context = """
You are NMAP-AGENT, a deterministic sub-agent.  
Your only tool is "nmap_scan".  
Your job: analyze the assigned target, profile hosts with multiple *focused* scans, and then produce a final report.

OBJECTIVE:
1. If the target is a network (CIDR):
   - Discover live hosts in the network.
2. If the target is a single host (IP):
   - Treat the given host as the only scan target.
3. For each target host perform several distinct scans:
   - Discovery: -sn (network targets only)
   - Port/service: -sS or -sT on small to medium port sets (e.g. "22,80,443" or a short list)
   - Version detection: -sV on already discovered open ports
   - Aggressive profiling: -sC, -A, or --script vuln

Aggressive scanning DOES NOT mean scanning all 65535 ports.  
Never attempt a full-range scan (e.g. "-p 1-65535").

PORT RULES:
- Never scan more than ~20 ports in a single tool call.
- Prefer short port lists (e.g. "21,22,80,443,3306") or targeted ports from memory.
- If ports are unknown, probe a small common-port set.
- Never use "-p 1-65535" or any full-range scan.

OUTPUT RULES:
Each step outputs exactly ONE of:
1. CALL_TOOL: <JSON>
2. FINAL_ANSWER: <text>
3. ERROR: <text>

No markdown, no emojis, no explanations.

TOOL CALL FORMAT (exact):
CALL_TOOL: {
  "tool": "nmap_scan",
  "args": {
    "target": "<ip or cidr>",
    "scan_type": "<e.g. -sn, -sS, -sV, -sC, -A>",
    "ports": "<short list or empty>",
    "additional_args": "<string>"
  }
}

Do not repeat the exact same scan on a host.  
Do not add extra fields.

BEHAVIOR RULES:
- Stay strictly inside the assigned target scope.
- Scan only hosts inside discovered_hosts.
- If the target is a single host and discovered_hosts is empty, initialize it with that host.
- Prefer many small scans over one big scan.
- After each scan, select the host with the fewest known facts.
- Never guess hosts or facts.
- Stop only when memory clearly shows all target hosts are fully profiled.

MEMORY:
SYSTEM MESSAGE may contain YOUR TASK, CURRENT STATE, LAST TOOL CALL, LAST TOOL OUTPUT.  
If a field is missing, ignore it. Never recreate missing memory.

ERROR HANDLING:
- On tool error, retry with adjusted parameters.
- If stuck, return FINAL_ANSWER with a short explanation.

FINAL REPORT:
When finished, output FINAL_ANSWER summarizing each host and overall findings.  
No markdown. No JSON except inside CALL_TOOL.

===========================
       END OF CONTEXT
===========================
"""


# -------------------------------------------------------------------------------#
#                                 Custom agent state                             #
# -------------------------------------------------------------------------------#


class customAgentState(BaseModel):

    discovered_hosts: List[str] = Field(
        description="A list of discovered hosts that are in queue to be scanned.",
        default_factory=list,
    )
    memory: Dict[str, Any] = Field(
        description="Dictionary with individual hosts and their coresponding facts.",
        default_factory=dict,
    )


# -------------------------------------------------------------------------------#
#                                Agent definition                                #
# -------------------------------------------------------------------------------#


@task
async def callModel(
    messages: list[BaseMessage], customAgentState: customAgentState, toolResult=None
):
    state_snapshot = json.dumps(
        {
            "discovered_hosts": customAgentState.discovered_hosts,
            "memory": customAgentState.memory,
        }
    )

    lastToolCall = await returnToolCall(mode="read")

    print("\n" + "=" * 40)  # additional debug stuff
    print(f"Last tool call:\n{lastToolCall}")
    print("\n" + "=" * 40)

    if toolResult:
        custom_message = f"""
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        
        LAST TOOL CALL:
        {lastToolCall}
        
        LAST TOOL OUTPUT:
        {toolResult}
        """
    else:
        custom_message = f"""
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        """

    # Agent state dump for debugging
    print("\n\n============= AGENT_STATE DUMP =============")
    print(
        json.dumps(
            {
                "discovered_hosts": customAgentState.discovered_hosts,
                "memory": customAgentState.memory,
            },
            indent=4,
            ensure_ascii=False,
        )
    )
    print("============================================\n\n")

    return await finalAgent.ainvoke(
        [SystemMessage(content=context), SystemMessage(content=custom_message)],
        config={"recursion_limit": 40},
    )


@task
async def callTool(tool_calls: List[ToolCall]):
    tool_call = tool_calls[0]

    try:
        rawOutput = await nmap_scan.arun(tool_call["args"])
    except Exception as e:
        rawOutput = {
            "stdout": "",
            "stderr": str(e),
            "success": False,
            "target": tool_call["args"].get("target", ""),
        }

    return ToolMessage(
        content=rawOutput,
        name="nmap_scan",
        tool_call_id=tool_call["id"],
    )


@task
async def updateState(
    toolMessage: ToolMessage, customAgentState: customAgentState, targetIP=None
):
    content = toolMessage.content
    lastToolCall = await returnToolCall(mode="read")

    if isinstance(content, list) and len(content) > 0:
        item = content[0]
        if hasattr(item, "text"):
            stdoutRaw = item.text
        else:
            stdoutRaw = str(item)
    elif isinstance(content, str):
        stdoutRaw = content
    elif isinstance(content, dict):
        stdoutRaw = content.get("stdout", "")
    else:
        stdoutRaw = ""

    stdout = stdoutRaw

    textTemp1 = str(stdoutRaw.split("stdout")[1])
    textTemp2 = textTemp1.split("\\\\n")
    del textTemp2[-1]

    print("\n" + "=" * 40)
    print("\nDATA PASSED TO AGENT\n")
    finalText = "\n".join(textTemp2[2:])
    finalText = finalText.encode("utf-8").decode("unicode_escape")
    print(f"Final text:\n{finalText}")
    print("\n" + "=" * 40)

    # extract ips and update state
    if stdout and targetIP:
        outputBlock = stdout.split("Nmap scan report for ")[1:]
        for block in outputBlock:
            addr = re.match(r"([0-9]{1,3}(?:\.[0-9]{1,3}){3})", block.strip())
            if addr:
                ip = addr.group(1)
                if ip not in customAgentState.discovered_hosts:
                    customAgentState.discovered_hosts.append(ip)

        customAgentState.memory.setdefault(targetIP, {"facts": []})
        customAgentState.memory[targetIP]["facts"].append(
            {
                "scan_type": lastToolCall.get("scan_type", ""),
                "ports": lastToolCall.get("ports", ""),
                "additional_args": lastToolCall.get("additional_args"),
                "notes": f"With the scan of {targetIP} we've learnt following facts:\n\n{finalText}",
            }
        )


@entrypoint()
async def agent(message: list[BaseMessage]):

    agent_state = customAgentState()
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

        toolResult = await callTool(response.tool_calls)

        targetIP = None
        try:
            args = response.tool_calls[0].get("args", {})
            targetIP = args.get("target")
        except Exception:
            pass

        await updateState(
            toolMessage=toolResult,
            customAgentState=agent_state,
            targetIP=targetIP,
        )

        response = await callModel(
            message, customAgentState=agent_state, toolResult=toolResult
        )


# -------------------------------------------------------------------------------#
#                                 Agent runners                                  #
# -------------------------------------------------------------------------------#

""" ONLY FOR DEBUGGING
async def runner(message):
    final_answer = None
    async for chunk in agent.astream(
        input=message, stream_mode="updates", config={"recursion_limit": 40}
    ):
        print("\n[DEBUG CHUNK]")
        print(chunk)

        if isinstance(chunk, dict):
            if "finalOutput" in chunk:
                final_answer = chunk["finalOutput"]
            elif getattr(chunk, "finalOutput", None):
                final_answer = getattr(chunk, "finalOutput")
            elif getattr(chunk, "content", None) and isinstance(chunk.content, str):
                final_answer = chunk.content

    if final_answer:
        print("\n\n# FINAL_ANSWER:")
        print(final_answer)
"""


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
    agent_finished: bool
    agent_report: str = Field(
        default_factory=str, description="Final report written by the tool agent."
    )


# -------------------------------------------------------------------------------#
#                                   Main loop                                    #
# -------------------------------------------------------------------------------#

if __name__ == "__main__":
    print("#" + "-" * 10 + "Agent_1_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    while True:
        supervisorInput = input("\nUser: ").strip()

        if supervisorInput.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        message = [HumanMessage(content=supervisorInput)]

        asyncio.run(agentRunner(message=message))
