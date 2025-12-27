import os
from dotenv import load_dotenv
from MCP_tools.sqlmap_tool import sqlmap_scan, returnSqlmapToolCall
from langchain_ollama import ChatOllama
import json
import re
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import Field, BaseModel
from langchain.messages import SystemMessage, ToolCall, ToolMessage, HumanMessage
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint, task
from datetime import datetime

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

# fix this jank from tutorial
tools = [sqlmap_scan]
tools_by_name = {tool.name: tool for tool in tools}

final_agent = llm.bind_tools(tools)

# -------------------------------------------------------------------------------#
#                                  Agent setup                                   #
# -------------------------------------------------------------------------------#

# initial context
context = """"""

# -------------------------------------------------------------------------------#
#                                 Custom agent state                             #
# -------------------------------------------------------------------------------#


# basic structure idea
class customAgentState(BaseModel):

    static_context: str = context

    system_prompt: Optional[str] = Field(
        default_factory=None, description="Command given by the orchestrator"
    )

    memory: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary with organized results of previous scans.",
    )


# -------------------------------------------------------------------------------#
#                                Agent definition                                #
# -------------------------------------------------------------------------------#


@task
async def callModdel(
    messages: List[BaseMessage],
    customAgentState: customAgentState,
    toolResult=None,
):

    return


@task
async def callTool(toolCall: ToolCall):
    return


@task
async def summarizeToolOutput(toolMessage: ToolMessage):
    return


@task
async def updateState(toolOutput, customAgentState: customAgentState):
    return


@entrypoint()
async def agent(message: list[BaseMessage]):
    return
