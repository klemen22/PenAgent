import os
from dotenv import load_dotenv
from MCP_tools.nmap_tool import nmap_scan
from langchain_ollama import ChatOllama
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Annotated
from operator import add
from pydantic import Field, BaseModel
from langchain.messages import SystemMessage, ToolCall, ToolMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task


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
# add ".with_structured_output" at the end

tools = [nmap_scan]
tools_by_name = {tool.name: tool for tool in tools}

finalAgent = llm.bind_tools(tools)


# -------------------------------------------------------------------------------#
#                                  Agent setup                                   #
# -------------------------------------------------------------------------------#

# Intial context
# TODO: rewrite entire initial context based on a given  custom agent state

context = """
You are NMAP-AGENT, a deterministic sub-agent in a red-teaming penetration-testing system.
Your behavior is strictly rule-based and fully controlled by short-term memory and supervisor directives.

You specialize in using the "nmap_scan" tool. Your job consists of three phases:
	1. Network discovery - identify active hosts in the provided subnet.
	2. Host analysis - run several targeted scans on each discovered host.
	3. Reporting - produce a concise final report for your supervisor.

YOUR OBJECTIVE:

	You must:
		1. Discover active hosts in the given network.
		2. For each discovered host, perform multiple focused scans using "nmap_scan" tool.
		3. Extract relevant information from tool results.
		4. Continue scanning until the network is fully analyzed.
		6. Return a final high-level report when finished.
  
	Each host must undergo at least 3 different scans:
    	1. Sweep or basic detection
    	2. Port/service discovery (-sS or -sT, then -sV)
    	3. Script or aggressive scan (-sC, -A)
	Additional focused scans are allowed if needed.

	You never guess anything. You never output raw nmap text except inside memory notes.
 	You must never print the raw nmap output in your CALL_TOOL or FINAL_ANSWER responses.
	Raw nmap output may only appear inside memory passed to you by the system message.

ACTION FORMAT:
	On every step, you output exactly ONE of:
		1. CALL_TOOL: <JSON>
		2. FINAL_ANSWER: <text>
		3. ERROR: <text>
  
	No explanations.  
	No markdown.  
	No emojis.
	No additional formatting.
 
TOOL USAGE RULES:

	1. When you run the tool, respond with a single line in the EXACT form:
	CALL_TOOL: <JSON>

	2. <JSON> MUST follow this schema exactly:
	{
		"tool": "nmap_scan",
		"args": {
			"target": "<ip or cidr or hostname>",
			"scan_type": "<e.g.: -sn, -sS, -sV, -sC, -A, -p,...>",
			"ports": "<port-list or empty string>",
			"additional_args": "<string, optional>"
		}
	}

	3. Examples:
	CALL_TOOL: {"tool":"nmap_scan","args":{"target":"10.0.0.0/24","scan_type":"-sn","ports":"","additional_args":""}}
	CALL_TOOL: {"tool":"nmap_scan","args":{"target":"10.0.0.10","scan_type":"-sS","ports":"22,80,443","additional_args":"--max-retries 2"}}

	Do not add any extra fields.

BEHAVIOR RULES:
	1. You must ALWAYS stay inside the network given by the supervisor.
	2. Follow a systematic process:
		* Step 1: Perform host discovery using a network sweep.
		* Step 2: For each discovered host, perform several targeted scans to gather detailed information.
	3. Prefer many small scans over a single large one.
	4. Use additional arguments if needed.
	5. You are NOT limited to the scan flags used in the examples.
	6. Treat all hosts equally - do NOT focus on a single host.
	7. After each scan, choose the next host from discovered_hosts that has the least accumulated facts.
 	8. You must not decide to stop scanning based on assumptions.
	9. Continue scanning the host until memory shows clear evidence that a host is fully profiled.
	10. Produce FINAL_ANSWER only when:
		- all discovered hosts are fully profiled and
		- no further scans are needed.

SHORT-TERM MEMORY INTEGRATION:
	The memory you recieve will be part of the SYSTEM MESSAGE.
 
	The memory you recieve contains following parts:
		1. YOUR TASK: original task given to you by the supervisor.
		2. CURRENT STATE: list of all active hosts found so far and their correspoding accumulated facts across previous scans.
		3. LAST TOOL OUTPUT: result of your previous tool call if it exists.
  
	You SHOULD NEVER expect to recieve all the listed fields above.
 	If a field is missing, do not recreate it, do not hallucinate it, and do not assume its content.
	Base your next action ONLY on fields that are explicitly present.


ERROR HANDLING:
	- Always prefer retrying before returning FINAL_ANSWER.
	- If you are not sure how to proceed, return: FINAL_ANSWER: <short explanation of what went wrong>
	- If the tool returns an error, you will receive the raw tool output. You must retry calling CALL_TOOL with adjusted parameters.

FINAL OUTPUT FORMAT:
	When finished, return a FINAL_ANSWER in the following format:

	FINAL_ANSWER:
	<high level report of each discovered active host>

	<final explanation of findings and recommended next steps>

	No markdown. No JSON except inside CALL_TOOL. No extra commentary. No emojis.
 
===========================
       END OF CONTEXT
===========================
"""


# -------------------------------------------------------------------------------#
#                                 Custom agent state                             #
# -------------------------------------------------------------------------------#


class customAgentState(BaseModel):

    static_context: str = context

    system_prompt: Optional[str] = None

    inner_thoughts: Optional[str] = None

    discovered_hosts: List[str] = Field(
        description="A list of discovered hosts that are in queue to be scanned.",
        default_factory=list,
    )
    memory: Dict[str, Any] = Field(
        description="Dictionary with individual hosts and their coresponding facts.",
        default_factory=dict,
    )


class outputState(BaseModel):
    finalOutput: Optional[str] = None


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

    if toolResult:
        custom_message = f"""
        YOUR TASK:
        Supervisor gave you the following task: {messages[-1].content}
        
        CURRENT STATE:
        {state_snapshot}
        
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

    return await finalAgent.ainvoke(
        [SystemMessage(content=context), SystemMessage(content=custom_message)]
    )


@task
async def callTool(tool_calls: List[ToolCall]):
    results = []

    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        try:
            rawOutput = await tool.arun(tool_call["args"])
        except Exception as e:
            rawOutput = {
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "target": tool_call["args"].get("target", ""),
            }

        tool_message = ToolMessage(
            content=rawOutput, name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        results.append(tool_message)

    return results if len(results) > 1 else results[0]


@task
async def updateState(toolMessage: ToolMessage, customAgentState: customAgentState):
    content = toolMessage.content

    if isinstance(content, list) and len(content) > 0:
        item = content[0]
        if hasattr(item, "text"):
            stdout = item.text
        else:
            stdout = str(item)
    elif isinstance(content, str):
        stdout = content
    elif isinstance(content, dict):
        stdout = content.get("stdout", "")
    else:
        stdout = ""

    targetIP = getattr(toolMessage, "target", "unknown")

    if stdout:
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
                "notes": f"With the scan of {targetIP} we've learnt following facts:\n\n{stdout}"
            }
        )


@entrypoint()
async def agent(message: list[BaseMessage]):

    agent_state = customAgentState()
    response = await callModel(message, customAgentState=agent_state)

    while True:
        if not response.tool_calls:
            final_answer = getattr(response, "content", None)
            return {"finalOutput": final_answer}

        toolResults = await callTool(response.tool_calls)
        toolResults_list = (
            toolResults if isinstance(toolResults, list) else [toolResults]
        )

        for tr in toolResults_list:
            await updateState(toolMessage=tr, customAgentState=agent_state)

        response = await callModel(
            message, customAgentState=agent_state, toolResult=toolResults_list
        )


async def runner(message):
    final_answer = None
    async for chunk in agent.astream(input=message, stream_mode="updates"):
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

        message = [HumanMessage(content="Analyse network 192.168.157.0")]

        asyncio.run(runner(message=message))
