from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Interrupt, RetryPolicy
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import MCP_tools.nmap_agent_ollama as nmap_agent
import os
import uuid
from datetime import datetime
from IPython.display import Image, display


load_dotenv()

# -------------------------------------------------------------------------------#
#                                  LLM setup                                     #
# -------------------------------------------------------------------------------#

# TODO: you probably need a seperate model for orchestrator!
# TODO: check "create_supervisor"
# TODO: add fallback
# TODO: add interrupts

LM_API = os.getenv(key="OLLAMA_API", default="http://127.0.0.1:11434")

memory_llm = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b",
    base_url=LM_API,
    temperature=0.2,
    format=None,
)

reasoning_llm = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b",
    base_url=LM_API,
    temperature=0.2,
    format=None,
)

planner_llm = ChatOllama(
    model="huihui_ai/qwen3-abliterated:8b",
    base_url=LM_API,
    temperature=0.2,
    format=None,
)

# -------------------------------------------------------------------------------#
#                                    Rules                                       #
# -------------------------------------------------------------------------------#

globalRules = """
- Never fabricate hosts, services, vulnerabilities, or findings.
- Prefer incremental, evidence-driven actions over speculative ones.
- Prefer trying again before terminating the assessment.
- Do not escalate to aggressive techniques unless justified by prior findings.
- Always aim to minimize noise and footprint.
- Terminate the assessment once the task objectives are clearly satisfied.
"""


# -------------------------------------------------------------------------------#
#                                 Agent state                                    #
# -------------------------------------------------------------------------------#
class taskState(BaseModel):
    task: str = Field(default_factory=str)
    command: Optional[str] = Field(
        default_factory=None,
        description="Command given to tool agent by planner agent.",
    )
    subagent: str = Field(
        default_factory=str,
        description="Name of the tool agent performing task.",
    )
    status: str = Field(default_factory=str, description="Status of the tool agent.")


class orchestratorState(BaseModel):
    # Task&Context
    rules: str = globalRules
    task: List[str] = Field(
        default_factory=list,
        description="List of initial tasks provided to orchestrator upon invocation",
    )
    # Reasoning
    next_action: Optional[str] = Field(
        default_factory=None,
        description="Next agent step decided by the reasoning node",
    )
    # Tools
    current_task: taskState
    tool_result: Optional[str] = Field(
        default_factory=None,
        description="Written summary of last tool output.",
    )
    # Output
    report: Optional[str] = Field(
        default_factory=None, description="Written report from orchestrator."
    )
    finished: bool = False


# -------------------------------------------------------------------------------#
#                                 Agent nodes                                    #
# -------------------------------------------------------------------------------#

# all available actions:
# - reason
# - nmap
# - sqlmap
# --------------------
# - store_memory
# - retrieve_memory
# - form_output


def reportNode(state: orchestratorState, *, config):  # report building and formatting
    # create invoke for formating a final report
    debugFunc(node="REPORT NODE - (entry)")

    id = config["configurable"]["user_id"]

    memory = inMemoryStore.search((id, "memories"), limit=20)

    prompt = f"""
    You are a reporter agent for penetration testing results.
    
    Your job is to write a final concise report for a penetration test based on the saved memories provided below.
    
    MEMORIES:
    {memory}
    
    Write a concise final report.
    No markdown.
    No direct tool outputs.
    No recommendations or additional commentary unless explicitly asked for.
    No emojis.
    """

    agentReport = reasoning_llm.invoke(prompt).content.strip()

    debugFunc(node="REPORT NODE - (exit)")
    return {"report": agentReport, "finished": True}


def outputNode(state: orchestratorState):

    # output exporting for debugging
    outputDir = "orchestrator_outputs"

    if not os.path.exists(outputDir):
        os.makedirs(outputDir, exist_ok=True)

    # ------------------------------
    # report exporting
    # ------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseName = f"orchestrator_report_{timestamp}.txt"

    filePath = os.path.join(outputDir, baseName)
    with open(file=filePath, mode="w", encoding="utf-8") as f:
        f.write(state.report or "")

    # ------------------------------
    # agent state exporting
    # ------------------------------

    baseName = f"orchestrator_state_{timestamp}.json"
    filePath = os.path.join(outputDir, baseName)
    with open(file=filePath, mode="w", encoding="utf-8") as f:
        f.write(state.model_dump_json(indent=2))

    return {}


def memoryNode(state: orchestratorState, *, config):  # node for memory
    debugFunc(node="MEMORY NODE - (entry)")

    # node will be used for long term saving
    id = config["configurable"]["user_id"]

    # add myb a seperate invoke wehere you ask an agent is it worth saving new facts at all
    memory = inMemoryStore.search((id, "memories"), limit=10)

    if not state.tool_result:
        return {}

    promptDecision = f"""
    You are a memory agent who decides whether the following information is worth saving into long-term memory.
    You will be provided with the current task state, the last tool output, and existing memory.
    Some of the fields below can be empty depending on the situation.
    
    CURRENT TASK:
    {state.current_task}
    
    LAST PARSED TOOL OUTPUT:
    {state.tool_result}
    
    EXISTING MEMORIES:
    {memory}
    
    You can ONLY respond with YES or NO!
    """
    agentDecision = memory_llm.invoke(promptDecision).content.strip().upper()

    debugFunc(
        node="MEMORY NODE - (decision)",
        message=f"Memory agent decision for saving new facts: {agentDecision}",
    )

    if agentDecision != "YES":
        return {}

    promptSummary = f"""
    You are an agent who specializes in writing concise and high-quality summaries for long-term storage.
    
    Write a concise summary based on the following facts listed below:
    
    CURRENT STEP:
    {state.current_task}
    
    LAST PARSED TOOL OUTPUT:
    {state.tool_result}
    """

    agentSummary = memory_llm.invoke(promptSummary)

    summary = agentSummary.content.strip()
    if summary:
        memoryNamespace = (id, "memories")
        memoryID = str(uuid.uuid4())
        memory = {"findings": summary}
        inMemoryStore.put(memoryNamespace, memoryID, memory)

    debugFunc(node="MEMORY NODE - (exit & memory dump)", memoryFlag=True)
    return {}


def reasoningNode(state: orchestratorState, config):
    debugFunc(node="REASONING NODE - (entry)")

    id = config["configurable"]["user_id"]
    # look at the current situation and decide on 1 of the allowed actions:
    # all available actions:
    # - nmap
    # - sqlmap
    # --------------------
    # - memory
    # - output

    memory = inMemoryStore.search((id, "memories"), limit=10)

    prompt = f"""
    You are a professional penetration testing agent who uses given tools to perform penetration testing tasks and red teaming scenarios.
    Your current position is a supervisor who manages a penetration test, tool agents, and decides what to do next.
    
    GIVEN TASK:
    {state.task}
    
    GLOBAL RULES:
    {state.rules}
    
    MEMORY:
    {memory}
    
    CURRENT STEP:
    {state.current_task}
    
    LAST PARSED TOOL OUTPUT:
    {state.tool_result}
    
    Based on the given information you MUST decide what to do next. You respond with one word only!
    Words that can be used: nmap, sqlmap, memory, output. Following words should be used according to the scenarios described below:
    
    I. nmap
        * "nmap" should be returned when usage of the nmap tool is needed.
        
    II. sqlmap
        * "sqlmap" should be returned when usage of the sqlmap tool is needed.
        
    III. memory
        * "memory" should be returned when tool results should be added to memory.
    
    IV. output
        * "output" should be returned when the task goal was achieved and it's time to form a final report.
    
    ANY OTHER WORDS ARE NOT ALLOWED!
    """
    decision = reasoning_llm.invoke(prompt).content.strip().lower()

    debugFunc(
        node="REASONING NODE - (exit & state dump)",
        message=f"Reasoning agent decision: {decision}",
        state=state,
    )

    # add fallback
    return {"next_action": decision}


def plannerNode(state: orchestratorState):

    debugFunc(node="PLANNER NODE - (entry)")

    prompt = f"""
    You are a planner agent for a penetration testing agent.
    Your job is to translate an intent into a concrete command for a tool agent.
    You make your translation based on the facts listed below (facts provided to you can be empty):
    
    NEW INTENT:
    {state.next_action}
    
    PREVIOUS TASK:
    {state.task}
    
    LAST TOOL OUTPUT:
    {state.tool_result}
    
    Respond with a SINGLE sentence command for the tool agent.
    Your single sentence command should be formed in plain text with natural language.
    
    DO NOT CREATE MULTIPLE SENTENCES AND DO NOT EXPLAIN!
    """

    newCommand = planner_llm.invoke(prompt).content.strip()

    debugFunc(
        node="PLANNER NODE - (exit & command)",
        message=f"Planner node command: {newCommand}",
    )

    return {
        "current_task": {
            **state.current_task.model_dump(),
            "command": newCommand,
        }
    }


async def nmapAgentNode(state: orchestratorState):
    command = state.current_task.command
    response = await nmap_agent.agentRunner(message=command)

    debugFunc(node="NMAP AGENT NODE - (entry)")

    if response.agent_finished:
        return {
            "next_action": "reason",
            "current_task": {
                **state.current_task.model_dump(),
                "task": "Nmap scan.",
                "subagent": "nmap_agent",
                "status": "Completed.",
            },
            "tool_result": extractReport(text=response.agent_report),
        }

    else:
        return {
            "next_action": "reason",
            "current_task": {
                **state.current_task.model_dump(),
                "task": "Nmap scan.",
                "subagent": "nmap_agent",
                "status": "Failed to complete the task.",
            },
            "tool_result": "Agent failed to return usable output.",
        }


async def sqlmapAgentNode(state: orchestratorState, command):
    # create sqlmapAgent tool first bozo
    return {}


# -------------------------------------------------------------------------------#
#                               Helper functions                                 #
# -------------------------------------------------------------------------------#


def extractReport(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if text.upper().startswith("FINAL_ANSWER:"):
        return text.split("FINAL_ANSWER:", 1)[1].strip()

    return text


def routingFunction(state: orchestratorState) -> str:
    print(f"- [ROUTUER] entry")

    if not state.next_action:
        return "reasoning"

    nextAction = state.next_action.lower()

    print(f"- [ROUTUER] next action: {nextAction}")

    match nextAction:
        case "nmap":
            return "nmap"
        case "sqlmap":
            return "sqlmap"
        case "memory":
            return "memory"
        case "output":
            return "output"

    print("- [ROUTER] unknown action, fallback to reasoning")
    return "reasoning"


# -------------------------------------------------------------------------------#
#                                    Graph                                       #
# -------------------------------------------------------------------------------#

if __name__ == "__main__":
    workflow = StateGraph(orchestratorState)

    # ----------------------
    # graph nodes
    # ----------------------

    workflow.add_node("reasoning_node", reasoningNode)
    workflow.add_node("planner_node", plannerNode)
    workflow.add_node("nmap_agent_node", nmapAgentNode)
    workflow.add_node("sqlmap_agent_node", sqlmapAgentNode)
    workflow.add_node("memory_node", memoryNode)
    workflow.add_node("report_node", reportNode)
    workflow.add_node("output_node", outputNode)

    # ----------------------
    # graph edges
    # ----------------------

    workflow.add_edge(START, "reasoning_node")
    workflow.add_conditional_edges(
        "reasoning_node",
        routingFunction,
        {
            "nmap": "planner_node",
            "sqlmap": "planner_node",
            "memory": "memory_node",
            "output": "report_node",
            "reasoning": "reasoning_node",
        },
    )
    workflow.add_conditional_edges(
        "planner_node",
        routingFunction,
        {
            "nmap": "nmap_agent_node",
            "sqlmap": "sqlmap_agent_node",
        },
    )
    workflow.add_edge("nmap_agent_node", "reasoning_node")
    workflow.add_edge("sqlmap_agent_node", "reasoning_node")
    workflow.add_edge("memory_node", "reasoning_node")
    workflow.add_edge("report_node", "output_node")
    workflow.add_edge("output_node", END)

    checkpointer = InMemorySaver()
    inMemoryStore = InMemoryStore()
    graph = workflow.compile(checkpointer=checkpointer, store=inMemoryStore)

    # display the workflow
    display(Image(graph.get_graph().draw_mermaid_png()))

    while True:
        print("\n" + "=" * 80)
        print("PEN TEST DOMINATOR")
        print("(Type 'exit' to close the chat)")
        userInput = input("\n> user: ")

        if userInput.strip().lower() == ["exit", "quit"]:
            break

        graph.invoke({"task": userInput})

        state = orchestratorState()

        if state.finished:
            print("\n" + "=" * 80)
            print("FINAL REPORT")
            print("\n" + "=" * 80)
            print(state.report)
            print("\n" + "=" * 80)

# -------------------------------------------------------------------------------#
#                                    Debug                                       #
# -------------------------------------------------------------------------------#


def debugFunc(
    node: str,
    state: orchestratorState | None = None,
    message: str | None = None,
    memoryFlag: bool = False,
    config=None,
):

    print("\n" + "-" * 80)
    print(f"-NODE: {node}")

    if message:
        print(f"-MESSAGE: {message}")

    if state:
        print(f"-STATE SNAPSHOT: {state}")

    if memoryFlag and config:
        id = config["configurable"]["user_id"]
        memories = inMemoryStore.search((id, "memories"), limit=50)
        print(f"-LONG TERM MEMORY SNAPSHOT")

        for i, memory in enumerate(memories, 1):
            print(f"{i}. {memory}")

    print("\n" + "-" * 80)

    return
