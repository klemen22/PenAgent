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
import shlex

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
    -STOP when:
        * Injection confirmed AND
        * At least one successful enumeration step completed OR
        * No injection confirmed after max escalation
        
EXPLOITATION RULES:
If injectable=true is confirmed:
    - Identify injection type and DBMS.
    - Escalate to enumeration mode.
    - Extract:
        * current database name
        * available databases
        * tables in current database
        * at least one table dump (limited rows)

Use sqlmap flags gradually:
    --current-db
    --dbs
    --tables
    --columns
    --dump (limit rows if possible)

Do not repeat detection scans after injection is confirmed.
Switch to exploitation phase.

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
    urls: List[str] = Field(
        default_factory=list,
        description="A list of valid URLs extracted from attack vectors for checking.",
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

    lastToolCall = await returnSqlmapToolCall(mode="read")

    # currentState = customAgentState.memory

    currentState = json.dumps(
        {
            endpoint: [scan.model_dump() for scan in scans]
            for endpoint, scans in customAgentState.memory.items()
        },
        indent=4,
    )

    if toolResult:
        customMessage = f"""
        YOUR TASK:
        Supervisor gave you the following task: {customAgentState.message}
        
        ATTACK VECTORS:
        {customAgentState.attack_vectors}
        
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
        
        ATTACK VECTORS:
        {customAgentState.attack_vectors}
        
        CURRENT STATE:
        {currentState}
        """

    print("\n\n============= AGENT_STATE DUMP =============")
    print(
        json.dumps(
            {
                endpoint: [scan.model_dump() for scan in scans]
                for endpoint, scans in customAgentState.memory.items()
            },
            indent=4,
        )
    )
    print("============================================\n\n")

    return await finalAgent.ainvoke(
        [SystemMessage(content=context), SystemMessage(content=customMessage)],
        config={"recursion_limit": 40},
    )


@task
async def callTool(toolCall: List[ToolCall], customAgentState: customAgentState):

    tool_call = toolCall[0]

    print("\n\n================ TOOL CALL ================")
    print(tool_call)
    print("============================================\n\n")

    # validate URLs and arguments
    try:
        validateURL(url=tool_call["args"]["url"], validUrls=customAgentState.urls)
    except ValueError as e:
        return {"stdout": "", "stderr": str(e), "success": False}

    try:
        validateArguments(tool_call["args"]["additional_args"])
    except ValueError as e:
        return {"stdout": "", "stderr": str(e), "success": False}

    vector = next(
        (
            v
            for v in customAgentState.attack_vectors
            if v["endpoint"] == tool_call["args"]["url"]
        ),
        None,
    )

    if vector:
        method = vector.get("method")
        params = vector.get("params", [])

        if method == "POST" and params:
            post_body = "&".join([f"{p}=1" for p in params])
            tool_call["args"]["data"] = post_body

            param_list = ",".join(params)
            tool_call["args"]["additional_args"] += f" -p {param_list}"

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
    print("\n\n============== TOOL OUTPUT ==============")
    print(toolOutput)
    print("============================================\n\n")

    print("\n> Summarizing tool output...")

    if not toolOutput or toolOutput == "":
        return ""

    # Temp. remove output filtering
    # filteredOutput = sqlmapOutputParser(toolOutput=toolOutput)

    filteredOutput = toolOutput

    customMessage = f"""
    You are a SQLMap output interpreter.

    STRICT RULES:
        - Use ONLY information present in the SQLMap output below.
        - DO NOT invent URLs.
        - DO NOT invent parameters.
        - DO NOT invent payloads.
        - If output says parameters are NOT injectable, explicitly state: "No injectable parameters found."
        - If no DBMS is mentioned, return dbms as null.
        - If no injection type is mentioned, return injection_type as null.
    
    CONSISTENCY RULES:
        - If injectable=false, then:
        - severity MUST be "none"
        - parameter MUST be null
        - dbms MUST be null
        - injection_type MUST be null

    Return result in this exact JSON format:

    {{
    "injectable": true/false,
    "parameter": "<parameter name or null>",
    "dbms": "<dbms or null>",
    "injection_type": "<type or null>",
    "severity": "none | suspected | confirmed",
    "reason": "<short explanation>"
    }}

    SQLMap output:
    {filteredOutput}
    """

    result = await summarizerAgent.ainvoke([SystemMessage(content=customMessage)])

    print("\n\n================ SUMMARY ================")
    print(result.content)
    print("============================================\n\n")
    return result.content


@task
async def updateState(toolOutputSummary, customAgentState: customAgentState):
    toolCommand = await returnSqlmapToolCall(mode="read")

    endpoint = toolCommand.get("url", "")
    customAgentState.memory.setdefault(endpoint, [])

    existingCommand = [scan.tool_command for scan in customAgentState.memory[endpoint]]

    if toolCommand in existingCommand:
        return

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

    # filter out attack vectors without parameters
    agent_state.attack_vectors = [
        v for v in input.endpoints if len(v.get("params", [])) > 0
    ]
    agent_state.message = input.message

    for vector in agent_state.attack_vectors:
        agent_state.urls.append(vector.get("endpoint", ""))

    print(f"VALID URLs list: {agent_state.urls}")

    print("> Initial agent invoke...")
    response = await callModel(customAgentState=agent_state)

    n = 0
    while True:
        if not response.tool_calls:
            final_answer = getattr(response, "content", None)

            if (
                not final_answer.strip().startswith("CALL_TOOL:")
                and final_answer.strip() != ""
            ):
                return json.dumps(
                    {
                        endpoint: [scan.model_dump() for scan in scans]
                        for endpoint, scans in agent_state.memory.items()
                    },
                    indent=4,
                )

        toolResult = await callTool(response.tool_calls, customAgentState=agent_state)
        toolResultSummary = await summarizeToolOutput(toolResult)
        await updateState(
            toolOutputSummary=toolResultSummary, customAgentState=agent_state
        )

        n += 1
        print(f"> Agent loop invoke: itteration = {n}")
        response = await callModel(
            customAgentState=agent_state, toolResult=toolResultSummary
        )
        print(f"> Agent response for itteration = {n}:\n\n{response.tool_calls}")


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


def validateURL(url: str, validUrls: list):

    if url in validUrls:
        return True
    else:
        raise ValueError(f"Invalid URL: {url}")


def validateArguments(additionalArgs: str):

    allowedArgs = [
        "--risk=",
        "--level=",
        "--technique=",
        "--method=",
        "--current-db",
        "--current-user",
        "--passwords",
        "--dbs",
        "--tables",
        "--columns",
        "--schema",
        "--dump",
        "--dump-all",
        "--fingerprint",
        "--batch",
        "--os-shell",
        "--os-pwn",
        "--flush-session",
        "-D",
        "-T",
        "-C",
        "-p",
        "-a",
        "-b",
        "-f",
    ]

    tokens = shlex.split(additionalArgs)

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # exact match
        if token in [
            "--current-db",
            "--dbs",
            "--tables",
            "--columns",
            "--fingerprint",
            "--dump",
            "--dump-all",
            "--batch",
            "--passwords",
            "--schema",
            "--os-shell",
            "--os-pwn",
        ]:
            i += 1
            continue

        # prefix match
        if any(token.startswith(prefix) for prefix in allowedArgs):
            i += 1
            continue

        # special cases
        if token in ["-D", "-T", "-C", "-p"]:
            if i + 1 >= len(tokens):
                raise ValueError(f"Missing value for: {token}")

            i += 2
            continue

        raise ValueError(f"Invalid sqlmap flag used: {token}")

    return True


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

    with open("MCP_tools\gobuster\crawler_test_dump2.json", "r") as f:
        endpoints = json.load(f)

    # print(f"\n\nATTACK VECTORS:\n\n{json.dumps(endpoints, indent=4)}")

    # TEST PROMPT: Perform SQLmap scans on the given attack vectors.
    while True:
        supervisorInput = input("\nUser: ").strip()

        if supervisorInput.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        message = [HumanMessage(content=supervisorInput)]

        result = asyncio.run(agentRunner(message=message, endpoints=endpoints))

        print("\n\n================ RESULT ================")
        print(result)
        print("============================================\n\n")
