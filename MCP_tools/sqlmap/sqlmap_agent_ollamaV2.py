from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage, ToolCall, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
import json

load_dotenv()

from MCP_tools.sqlmap.sqlmap_tool import sqlmap_scan

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

finalAgent = llm.bind_tools([sqlmap_scan])

# ------------------------------------------------------------------------------- #
#                                  Agent setup                                    #
# ------------------------------------------------------------------------------- #

context = """"""

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


class vector(BaseModel):
    target_url: str = Field(default="", description="Targeted URL for scan.")
    method: str = Field(default="GET")
    params: List[str] = Field(
        default=[], description="List of arguments for tool scan."
    )
    post_data: str = Field(default="", description="Additional data to post.")

    phase: str = Field(
        default="detection",
        description="Flag seperating detection and exploitation scans.",
    )
    injection_found: bool = Field(default=False)

    raw_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Field for saving raw tool results if neccessary."
    )
    parsed_result: Optional[str] = Field(
        default=None, description="Field for saving summary of tool output."
    )

    finished: bool = Field(default=False)


class sqlmapAgentState(BaseModel):
    message: str = Field(default="", description="Task given by the orchestrator.")
    attack_vectors: List[vector] = Field(
        default_factory=list, description="List of given attack vectors."
    )
    index: int = Field(
        default=0, description="Index of an attack vector in attack vectors list."
    )

    next_action: Optional[str] = Field(
        default="reasoning", description="Planned next action for node routing."
    )
    planned_phase: Optional[str] = Field(
        default="detection", description="Planned phase for tool usage."
    )

    finished: bool = Field(
        default=False,
        description="Flag for signalizing if agent is done with the given task.",
    )


# ------------------------------------------------------------------------------- #
#                                 Agent nodes                                     #
# ------------------------------------------------------------------------------- #

# current node setup idea:
#   > reasoning node - can be devided into: check escalation, select action nodes
#   > planning node
#   > tool execution node
#   > parser node
#   > update state node


def reasoningNode(state: sqlmapAgentState):
    print("\n[REASONING NODE]\n")

    if state.finished:
        return state.model_dump()

    if state.index >= len(state.attack_vectors):
        state.finished = True
        return state.model_dump()

    current = state.attack_vectors[state.index]

    customMessage = f"""
    You are a SQL injection agent.
    
    Current Endpoint:
    URL: {current.target_url}
    Method: {current.method}
    Params: {current.params}
    Phase: {current.phase}
    Injection found: {current.injection_found}
    
    Decide next action.
    You MUST only respond with one of the following words:
    
    - detect
    - exploit
    - next
    - finish
    """

    decision = llm.invoke(input=customMessage).content.strip().lower()

    print(f"Agent made for target {current.target_url} following decision: {decision}")
    print(100 * "=")

    state.next_action = decision
    return state.model_dump()


def planningNode(state: sqlmapAgentState):
    print("\n[PLANNING NODE]\n")

    action = state.next_action

    if action == "detect":
        state.planned_phase = "detection"
    elif action == "exploit":
        state.planned_phase = "exploitation"
    elif action == "next":
        state.index += 1
    elif action == "finish":
        state.finished = True

    return state.model_dump()


async def toolExectionNode(state: sqlmapAgentState):
    print("\n[TOOL EXECTUION]\n")

    if state.finished or state.planned_phase is None:
        return state.model_dump()

    current = state.attack_vectors[state.index]

    if state.planned_phase == "detection":
        additional_args = "--level=1 --risk=1"

    elif state.planned_phase == "exploitation":
        additional_args = "--dbs"

    result = await sqlmap_scan.arun(
        {
            "url": current.target_url,
            "data": current.post_data,
            "additional_args": additional_args,
        }
    )

    # print(f"\n[RAW TOOL RESULT]\n\n{result}")

    if isinstance(result, tuple):
        result = result[1].get("result", {})

    elif isinstance(result, dict) and "result" in result:
        result = result["result"]

    current.raw_result = result

    return state.model_dump()


def parserNode(state: sqlmapAgentState):
    print("\n[PARSER NODE]\n")

    current = state.attack_vectors[state.index]

    current.parsed_result = sqlmapOutputParser(current.raw_result)

    current.raw_result = {"status": "already parsed"}

    if current.parsed_result and "is vulnerable" in current.parsed_result.lower():
        current.injection_found = True

    return state.model_dump()


def updateStateNode(state: sqlmapAgentState):
    print("\n[UPDATE STATE]\n")

    current = state.attack_vectors[state.index]

    if current.phase == "detection" and current.injection_found:
        current.phase = "exploitation"
    elif current.phase == "exploitation":
        current.finished = True
        state.index += 1
    else:
        current.finished = True
        state.index += 1

    return state.model_dump()


def outputNode(state: sqlmapAgentState):
    print(json.dumps(state.model_dump(), indent=4))

    return state.model_dump()


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


def shouldContinue(state: sqlmapAgentState):
    if state.index >= len(state.attack_vectors):
        return "output"

    for scan in state.attack_vectors:
        if not scan.finished:
            return "reasoning"

    return "output"


def prepareVectors(state: sqlmapAgentState, attackVectors):

    for initialVector in attackVectors:
        if len(initialVector.get("params", [])) > 0:
            state.attack_vectors.append(
                vector(
                    target_url=initialVector.get("endpoint"),
                    method=initialVector.get("method", ""),
                    params=initialVector.get("params"),
                )
            )

    return


# ------------------------------------------------------------------------------- #
#                                    Graph                                        #
# ------------------------------------------------------------------------------- #


async def agentRunner(endpoints):
    agentState = sqlmapAgentState()

    workflow = StateGraph(sqlmapAgentState)
    SESSION_ID = "default_session"

    prepareVectors(state=agentState, attackVectors=endpoints)

    # -------------------------------
    # graph nodes
    # -------------------------------

    workflow.add_node("reasoning_node", reasoningNode)
    workflow.add_node("planner_node", planningNode)
    workflow.add_node("tool_node", toolExectionNode)
    workflow.add_node("parser_node", parserNode)
    workflow.add_node("update_state_node", updateStateNode)
    workflow.add_node("output_node", outputNode)

    # -------------------------------
    # graph edges
    # -------------------------------

    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "reasoning_node")
    workflow.add_edge("reasoning_node", "tool_node")
    workflow.add_edge("tool_node", "parser_node")
    workflow.add_edge("parser_node", "update_state_node")
    workflow.add_conditional_edges(
        "update_state_node",
        shouldContinue,
        {"reasoning": "reasoning_node", "output": "output_node"},
    )
    workflow.add_edge("output_node", END)

    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    # display workflow
    pngBytes = graph.get_graph().draw_mermaid_png()
    pngPath = "MCP_tools\sqlmap\sqlmap_agent_graph.png"

    with open(pngPath, "wb") as f:
        f.write(pngBytes)

    await graph.ainvoke(
        agentState.model_dump(),
        config={"thread_id": SESSION_ID, "recursion_limit": 100},
    )


# ------------------------------------------------------------------------------- #
#                                   Main loop                                     #
# ------------------------------------------------------------------------------- #

# only for debugging
if __name__ == "__main__":
    print("#" + "-" * 10 + "SQLmap_agent_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    with open("MCP_tools\gobuster\crawler_test_dump2.json", "r") as f:
        endpoints = json.load(f)

    result = asyncio.run(agentRunner(endpoints=endpoints))
