# universal tool agent output schema

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from orchestartor_agent_ollamaV2 import attackVector, vulnerability, hostMemory


class AgentOutput(BaseModel):
    agent_name: str = Field(default="")

    summary: Optional[str] = Field(default=None, description="Tool agent summary.")

    # flags
    success: bool = Field(default=False)
    fail: bool = Field(default=False)
    fail_reason: Optional[str] = Field(default=None)

    # nmap
    discovered_hosts: List[str] = Field(default=[])
    host_memory: Dict[str, hostMemory] = Field(default={})

    # gobuster + crawler
    attack_vectors: List[attackVector] = Field(default=[])

    # other agents - sqlmap,...
    vulnerabilities: List[vulnerability] = Field(default=[])

    raw_output: Optional[str] = None
