from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Interrupt, RetryPolicy
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import Literal
import MCP_tools.nmap_agent_ollama as nmap_agent
import os
import uuid


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
#                                   Context                                      #
# -------------------------------------------------------------------------------#

# TODO
rules = "placeholder"


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
    RULES = rules
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


def entryNode(state: orchestratorState):  # entry node
    if state.next_action == None:
        return {"next_action": "reason"}


def outputNode():  # output building and formatting
    # create invoke for formating a final report
    return


def memoryNode(state: orchestratorState, *, config):  # node for memory
    # node will be used for long term saving
    id = config["configurable"]["user_id"]

    # add myb a seperate invoke wehere you ask an agent is it worth saving new facts at all

    prompt = f"""
    You are an agent who specializes in making good and concise summaries for long term storing.
    
    Write a concise summary based on the following facts listed bellow:
    
    CURRENT STEP:
    {state.current_task}
    
    LAST PARSED TOOL OUTPUT:
    {state.tool_result}
    """

    agentSummary = memory_llm.invoke(prompt)

    summary = agentSummary.content
    if summary:
        memoryNamespace = (id, "memories")
        memoryID = str(uuid.uuid4())
        memory = {"findings": summary}
        inMemoryStore.put(memoryNamespace, memoryID, memory)

    return {}


def reasoningNode(state: orchestratorState, config):
    id = config["configurable"]["user_id"]
    # look at the current situation and decide on 1 of the allowed actions:
    # all available actions:
    # - nmap
    # - sqlmap
    # --------------------
    # - memory
    # - output

    memory = inMemoryStore.search((id, "findings"), limit=10)

    prompt = f"""
    You are a professional penetration testing agent who uses given tools to perform penetration testing tasks and red teaming scenarios.
    Your current position is a supervisor who manages penetration test, tool agents and decides what to do next.
    
    GIVEN TASK:
    {state.task}
    
    RULES:
    {state.RULES}
    
    MEMORY:
    {memory}
    
    CURRENT STEP:
    {state.current_task}
    
    LAST PARSED TOOL OUTPUT:
    {state.tool_result}
    
    Based on the given information you MUST decide what to do next. You respond with one word only!
    Words that can be used: nmap, sqlmap, memory, output. Following words should be used according to bellow described scenarios:
    
    I. nmap
        * "nmap" word should be returned when usage of nmap tool is needed.
        
    II. sqlmap
        * "sqlmap" word should be returned when useg of sqlmap tool is needed.
        
    III. memory
        * "memory" word should be returned when tool results should be added to the memory.
    
    IV. output
        * "output" word should be returned when the task goal was achieved and it't time to form a final report.
    
    ANY OTHER WORDS ARE NOT ALLOWED!
    """
    decision = reasoning_llm.invoke(prompt).content.strip().lower()

    # add fallback

    return {"next_action": decision}


def plannerNode(state: orchestratorState):

    prompt = f"""
    You are a planner agent for a penetration testing agent.
    Your job is to translate an intent into a concrete command for a tool agent.
    You are making your translation based on the facts listed bellow (facts provided to you can be empty):
    
    NEW INTENT:
    {state.next_action}
    
    PREVIOUS TASK:
    {state.task}
    
    LAST TOOL OUTPUT:
    {state.tool_result}
    
    Respond with a SINGLE sentence command for the tool agent. 
    Your single sentence command should be formed in a plain text with natural language.
    
    DO NOT CREATE MULTI SENTENCES OUTPUT AND DO NOT EXPLAIN!   
    """

    newCommand = planner_llm.invoke(input=prompt).content.strip()

    return {"current_task": {"command": newCommand}}


async def nmapAgent(state: orchestratorState):
    command = state.current_task.command
    response = await nmap_agent.agentRunner(message=command)

    if response.agent_finished:
        return {
            "next_action": "reason",
            "current_task": {
                "task": "Nmap scan.",
                "subagent": "nmap_agent",
                "status": "Completed.",
            },
            "tool_result": extractReport(text=response),
        }

    else:
        return {
            "next_action": "reason",
            "current_task": {
                "task": "Nmap scan.",
                "subagent": "nmap_agent",
                "status": "Failed to complete the task.",
            },
            "tool_result": "Agent field to return any information.",
        }


async def sqlmapAgent(state: orchestratorState, command):
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


# -------------------------------------------------------------------------------#
#                                    Graph                                       #
# -------------------------------------------------------------------------------#

workflow = StateGraph(orchestratorState)

# add nodes & edges

checkpointer = InMemorySaver()
inMemoryStore = InMemoryStore()
graph = workflow.compile(checkpointer=checkpointer, store=inMemoryStore)

# -------------------------------------------------------------------------------#
#                                    Main                                        #
# -------------------------------------------------------------------------------#
