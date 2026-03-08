from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any, Literal
import asyncio
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import json
import logging
from pathlib import Path

# tool agents
from MCP_tools.nmap.nmap_agent_ollama import agentRunner as nmapAgent
from MCP_tools.gobuster.gobuster_agent_ollama import agentRunner as gobusterAgent
from MCP_tools.crawler import main as crawlerMain
from MCP_tools.sqlmap.sqlmap_agent_ollamaV3 import agentRunner as sqlmapAgent

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

logDir = Path("MCP_tools/Orchestartor/logs")
logCount = 0

for log in os.listdir(logDir):
    if os.path.isfile(os.path.join(logDir, log)):
        logCount += 1

# ------------------------------------------------------------------------------- #
#                                 Custom agent state                              #
# ------------------------------------------------------------------------------- #


# ----------------- State ----------------- #
class orchestartorState(BaseModel):
    objective: str = Field(default="", description="Main objective given by the human.")

    ## TODO: fix nmap agent
    # -------- nmap outputs -------- #
    hosts: Optional[List[str]] = Field(
        default_factory=list, description="List of hosts discovered by nmap agent."
    )

    ports: Optional[Dict[str, List[Any]]] = Field(
        default_factory=dict, description="Discovered ports for corresponding hosts."
    )

    # -------- gobuster + crawler output -------- #
