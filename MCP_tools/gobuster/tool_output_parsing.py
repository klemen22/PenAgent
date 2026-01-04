from pydantic import BaseModel, Field
from typing import Any, Dict, List


class customAgentState(BaseModel):
    memory: List[Dict[str, Any]] = Field(
        default_factory=list, description="Test state."
    )


def createStateTest(toolOutput):
    stdout = str(toolOutput[0].split("stdout")[1])

    i = 0
    print(f"len: {len(stdout.split("\\n"))}")
    while i < len(stdout.split("\\n")):
        print(f"{i}: {stdout.split("\\n")[i]}")
        i = i + 1


if __name__ == "__main__":
    with open("MCP_tools\gobuster\gobuster_tool_output_example.txt", "r") as f:
        output = f.readlines()

    createStateTest(output)
