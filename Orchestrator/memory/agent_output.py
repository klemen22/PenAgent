# universal tool agent output schema
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal

# ---------------------
# nmap info - basic host info
# ---------------------


class portInfo(BaseModel):
    port: Optional[int] = Field(default=None)
    service: Optional[str] = Field(default=None)
    version: Optional[str] = Field(default=None)
    state: str = Field(default="unknown")


class HostMemory(BaseModel):
    ip: str = Field(default="")
    status: str = Field(default="unknown")
    open_ports: List[portInfo] = Field(default=[])
    os_guess: Optional[str] = Field(default=None)


# ---------------------
# crawler info - attack vector
# ---------------------


class attackVector(BaseModel):
    endpoint: str = Field(default="")
    method: str = Field(default="")
    parameters: List[str] = Field(default_factory=list)
    vector_type: str = Field(default="")
    confidence: int = Field(default=0)
    cookies: Optional[Dict[str, Any]] = Field(default=dict)
    origins: List[str] = Field(default_factory=list)

    tested: bool = Field(default=False)


# ---------------------
# vulnerabilities
# ---------------------


class vulnerability(BaseModel):
    source_agent: str = Field(default="")
    host: str = Field(default="")
    url: str = Field(default="")
    parameters: List[str] = Field(default_factory=list)
    vulner_type: str = Field(default="")
    severity: str = Field(default="")
    evidence: str = Field(default="")


# ---------------------
# main schema
# ---------------------


class AgentOutput(BaseModel):
    agent_name: Optional[Literal["nmap", "sqlmap", "gobuster"]] = Field(default=None)

    summary: Optional[str] = Field(default=None, description="Tool agent summary.")

    # flags
    success: bool = Field(default=False)
    fail: bool = Field(default=False)
    fail_reason: Optional[str] = Field(default=None)

    # nmap
    discovered_hosts: List[str] = Field(default=[])
    host_memory: Dict[str, HostMemory] = Field(default={})

    # gobuster
    host_enum: Dict[str, Dict] = Field(default={})

    # gobuster + crawler
    attack_vectors: List[attackVector] = Field(default=[])

    # other agents - sqlmap,...
    vulnerabilities: List[vulnerability] = Field(default=[])
