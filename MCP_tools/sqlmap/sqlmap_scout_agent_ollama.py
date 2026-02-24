import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import json
import asyncio
from typing import Dict, Any, List
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
Your job:
    - Identify and confirm SQL injection vulnerabilities on the assigned endpoints.
    - Escalate when justified.
    - Build structured knowledge in memory.

OBJECTIVE:
    1. Analyze provided attack vectors (URLs, parameters, methods).
    2. Identify potentially injectable parameters.
    3. Confirm injection through sqlmap evidence.
    4. Build structured knowledge in memory for each endpoint.
    
SCOPE RULES:
    - Only scan endpoints provided in attack_vectors.
    - Never invent new URLs.
    - Never modify domains.
    - Never scan unrelated targets.
    
SCAN RULES:
    - Prefer small, focused scans over aggressive full scans.
    - Start with detection-oriented scans.
    - If injection is suspected, confirm it before escalating.
    - Do not immediately use full enumeration flags.
    - Avoid unnecessary heavy exploitation. 
    
TOOL CALL FORMAT (exact):
    CALL_TOOL:{
        "tool": "sqlmap_scan",
        "args": {
            "url": "<full target url>",
            "data": "<POST body or empty string>",
            "additional_args": "<full sqlmap CLI arguments as string>"
        }
    }

Important:
    - "data" must be an empty string for GET requests.
    - All SQLMap flags go inside "additional_args".
    - Do not add extra fields.
    - Do not wrap JSON in markdown.
    
BEHAVIOR RULES:
    - Do not repeat identical scans on the same endpoint.
    - If a scan produces no injection evidence, adjust parameters carefully.
    - Escalate level and risk gradually.
    - Prefer many small iterations over one aggressive scan.
    - Use memory and known facts to avoid redundant scans.
    - STOP only when:
        * All given attack vectors were tested.
        * No further meaningful scans are possible.

OUTPUT RULES:
Each step must output exactly ONE of:
    1. CALL_TOOL: <JSON>
    2. FINAL_ANSWER: <text>
    3. ERROR: <text>

No markdown, no explanations, no emojis, no extra fields.

MEMORY:
You may receive:
    - YOUR TASK
    - CURRENT STATE
    - LAST TOOL CALL
    - LAST SUMMARIZED TOOL OUTPUT

If any field is missing, ignore it.
Never recreate or hallucinate missing memory.
"""

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class endpointScan(BaseModel):
    itteration: int = Field(default=0)
    tool_command: Dict[str, Any] = Field(
        default_factory=dict, description="Tool command used for this iterration."
    )
    facts: str = Field(
        default="", description="Potential facts learnt in this iterration."
    )


class customAgentState(BaseModel):

    message: str = Field(default="", description="Base task given by the orchestrator.")

    attack_vectors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Attack vectors given by the orchestrator."
    )
    memory: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Dictionary with organized results of sqlmap scans.",
    )


class agentInput(BaseModel):
    message: List[BaseMessage] = Field(
        default="", description="Initial task given by the orchestrator"
    )
    endpoints: List[Dict] = Field(
        default_factory=list, description="Given attack vectors"
    )


# ------------------------------------------------------------------------------- #
#                                Agent definition                                 #
# ------------------------------------------------------------------------------- #


@task
async def callModel(
    customAgentState: customAgentState,
    toolResult=None,
):
    print("placeholder")

    lastToolCall = await returnSqlmapToolCall(mode="read")

    currentState = json.dumps({"memory": customAgentState.memory})

    if toolResult:
        customMessage = f"""
        YOUR TASK:
        Supervisor gave you the following task: {customAgentState.message}
        
        CURRENT STATE:
        {currentState}
        
        LAST TOOL CALL:
        {lastToolCall}
        
        LAST SUMMARIZED TOOL OUTPUT:
        {toolResult}
        """
    else:
        customMessage = f"""
        YOUR TASK:
        Supervisor gave you the following task: {customAgentState.message}
        
        CURRENT STATE:
        {currentState}
        """

    print("\n\n============= AGENT_STATE DUMP =============")
    print(json.dumps({"memory": customAgentState.memory}, indent=4))
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
async def summarizeToolOutput(toolOutput):
    print("\n> Summarizing tool output...")

    filteredOutput = sqlmapOutputParser(toolOutput=toolOutput)

    customMessage = f"""
    Analyze the following filtered SQLMap CLI output.
    Extract structured findings.
    
    SQLMap output:
    {filteredOutput}
    """

    result = await summarizerAgent.ainvoke([SystemMessage(content=customMessage)])

    return result


@task
async def updateState(toolOutputSummary, customAgentState: customAgentState):
    toolCommand = await returnSqlmapToolCall(mode="read")

    endpoint = toolCommand.get("url", "")
    customAgentState.memory.setdefault(endpoint, [])

    customAgentState.memory[endpoint].append(
        endpointScan(
            itteration=len(customAgentState.memory[endpoint]) + 1,
            tool_command=toolCommand,
            facts=toolOutputSummary,
        )
    )


@entrypoint()
async def agent(input: agentInput):
    agent_state = customAgentState()

    agent_state.attack_vectors = input.endpoints
    agent_state.message = input.message

    response = await callModel(customAgentState=agent_state)

    while True:
        if not response.tool_calls:
            final_answer = getattr(response, "content", None)

            if (
                not final_answer.strip().startswith("CALL_TOOL:")
                and final_answer.strip() != ""
            ):
                return agent_state.memory

        toolResult = await callTool(response.tool_calls)
        toolResultSummary = await summarizeToolOutput(toolResult)
        await updateState(toolOutput=toolResultSummary, customAgentState=agent_state)

        response = await callModel(
            customAgentState=agent_state, toolResult=toolResultSummary
        )


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


# ------------------------------------------------------------------------------- #
#                                  Agent runner                                   #
# ------------------------------------------------------------------------------- #


async def agentRunner(message, endpoints):
    input = agentInput()

    input.message = message
    input.endpoints = endpoints

    response = await agent.ainvoke(input=input, config={"recursion_limit": 40})
    return response


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

# only for debugging
if __name__ == "__main__":
    print("#" + "-" * 10 + "SQLmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    with open("MCP_tools\gobuster\crawler_test_dump.json", "r") as f:
        endpoints = json.load(f)

    print(f"\n\nATTACK VECTORS:\n\n{json.dumps(endpoints, indent=4)}")

    while True:
        supervisorInput = input("\nUser: ").strip()

        if supervisorInput.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        message = [HumanMessage(content=supervisorInput)]

        asyncio.run(agentRunner(message=message, endpoints=endpoints))
