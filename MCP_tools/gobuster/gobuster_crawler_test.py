# simple test script for testing compatibility between gobuster agent and crawler

from langchain.messages import SystemMessage
from gobuster_agent_ollama import agentRunner as gobusterRunner
from MCP_tools.crawler import main as crawlerRunner
import asyncio
import json

command = "Enumerate HTTP endpoints on http://192.168.157.133"
gobusterPath = "MCP_tools/gobuster/gobuster_test_dump.json"
crawlerPath = "MCP_tools/gobuster/crawler_test_dump.json"

if __name__ == "__main__":
    print("\n" + 20 * "-" + "\n")
    print("Gobuster + crawler test")
    print(20 * "-" + "\n")

    print("\nStarting gobuster agent...")
    print(f"\nTest command:{command}")
    gobusterResult = asyncio.run(
        gobusterRunner(message=[SystemMessage(content=command)])
    )
    print(f"\n Gobuster has finished\n")

    # modelDump = json.dumps(gobusterResult, indent=4)

    print(f"\nGobuster output:\n{gobusterResult}")
    with open(gobusterPath, "w") as f:
        f.write(json.dumps(gobusterResult, indent=4))

    print(f"\nStarting crawler...\n")
    crawlerResult = asyncio.run(crawlerRunner(payload=gobusterResult))
    print(f"\nCrawler has finished...\n")

    crawlerDump = json.dumps(
        [vector.model_dump() for vector in crawlerResult], indent=4
    )
    with open(crawlerPath, "w") as f:
        f.write(crawlerDump)
