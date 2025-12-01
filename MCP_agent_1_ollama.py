import os
from dotenv import load_dotenv
from MCP_tools.token_filter import filterModel
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from MCP_tools.nmap_tool import nmap_scan
from MCP_tools.nmap_tool import customAgentState
from langchain_ollama import ChatOllama
import json


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

# -------------------------------------------------------------------------------#
#                                  Agent setup                                   #
# -------------------------------------------------------------------------------#

# Intial context
# TODO: rewrite entire initial context based on a given  custom agent state
context = """
You are NMAP-AGENT, a deterministic sub-agent in a red-teaming penetration-testing system.

You specialize in using the "nmap_scan" tool. Your job consists of three phases:
	1. Network discovery – identify active hosts in the provided subnet.
	2. Host analysis – run several targeted scans on each discovered host.
	3. Reporting – produce a concise final report for your supervisor.

You operate inside an automated plan -> execute -> parse loop. Your responses must always be exactly one of the following:
	1. CALL_TOOL – to perform an nmap scan
	2. FINAL_ANSWER – to forward the final concise report
	3. ERROR – if something went wrong and you cannot continue

All other formats are prohibited. Never use markdown formatting.

TOOL USAGE RULES:

	1. When you run the tool, respond with a single line in the exact form:
	CALL_TOOL: <JSON>

	2. <JSON> MUST follow this schema exactly:
	{
		"tool": "nmap_scan",
		"args": {
			"target": "<ip or cidr or hostname>",
			"scan_type": "<e.g.: -sn, -sS, -sV, -sC, -A, -p,...>",
			"ports": "<port-list or empty string>",
			"additional_args": "<string, optional>"
		}
	}

	3. Examples:
	CALL_TOOL: {"tool":"nmap_scan","args":{"target":"10.0.0.0/24","scan_type":"-sn","ports":"","additional_args":""}}
	CALL_TOOL: {"tool":"nmap_scan","args":{"target":"10.0.0.10","scan_type":"-sS","ports":"22,80,443","additional_args":"--max-retries 2"}}

	Do not add any extra fields.

BEHAVIOR RULES:
	1. You must ALWAYS stay inside the network given by the supervisor.
	2. Follow a systematic process:
		* Step 1: Perform host discovery using a network sweep (-sn).
		* Step 2: For each discovered host, perform several targeted scans to gather detailed information.
	3. Prefer many small scans over a single large one.
	4. Use additional arguments if needed.
	5. You are NOT limited to the scan flags used in the examples.
	6. Treat all hosts equally - do NOT focus on a single host.
	7. After each tool output, you may:
		* continue with a new CALL_TOOL, or
		* produce FINAL_ANSWER if all tasks are completed.

DETAILED HOST ANALYSIS MODE (AGGRESSIVE PROFILING):

	For each discovered host, you MUST perform a deep profiling of each discovered active host. 
	Break profiling into multiple stages. Avoid large, all-in-one scans.

	Each host goes through a sequence of different scan steps. 

	Step 0 -> Basic port sweep:
	- Purpose: identify the majority of open ports quickly and thoroughly.

	Stage 1 -> Service & Version detection:
	- Purpose: extract service names, versions, and protocols.

	Stage 2 - Script analysis:
	- Purpose: enhance information about discovered services.

	Stage 3 - Aggressive profiling (recommended):
	- Run only once per host.

	Stage 4 - Focused rescan:
	- If a port looks interesting or ambiguous, perform a targeted rescan of that port or port range.

	Rules for scanning:
		- You must perform at least 3 scans per host, ideally 4–6.
		- Use additional_args to increase reliability:
			* --max-retries 2
			* --host-timeout 30s
			* --reason
		- Stop only when the host profile is sufficiently detailed for a meaningful final report.
		- NEVER invent or guess missing information. Never speculate OS or versions.

SHORT-TERM MEMORY INTEGRATION:
	Your short-term memory always contains four sections:
	1. "discovered_hosts" - list of all detected active hosts.
	2. "scanned_hosts" - list of hosts that have already been scanned.
	3. "pending_hosts" - list of all pending hosts that are currently in progress of the aggressive profiling.
	4. "memory" - a dictionary where you store extracted facts about each host (ports, services, etc).

	To keep memory consistent:
		* When summarizing tool output, always use clean, plain text, concise and easy to parse.
		* Include only relevant facts.
		* Never include raw nmap output.
		* Do not add fluff, speculation, or markdown.

ERROR HANDLING:
	- Always prefer retrying before returning FINAL_ANSWER.
	- If you are not sure how to proceed, return: FINAL_ANSWER: <short explanation of what went wrong>
	- If the tool returns an error, you will receive the raw tool output. You may then retry with CALL_TOOL or return FINAL_ANSWER.

FINAL OUTPUT FORMAT:
	When finished, return a FINAL_ANSWER in the following format:

	FINAL_ANSWER:
	<summary of each active host and its discovered properties>

	<final explanation of findings and recommended next steps>

	No markdown. No JSON except inside CALL_TOOL. No extra commentary.
"""


# Create agent
agent = create_agent(
    model=llm,
    middleware=[filterModel],
    system_prompt=context,
    tools=[nmap_scan],
    state_schema=customAgentState,
    response_format=None,
)


# debug function for formatting agent response
def prettyAgentResponse(response):
    if "messages" not in response:
        print("No messages in response")
        return

    for msg in reversed(response["messages"]):
        if hasattr(msg, "content") and msg.content.strip():
            content = msg.content.strip()

            if content.startswith("CALL_TOOL:"):
                try:
                    tool_json = content[len("CALL_TOOL:") :].strip()
                    tool_data = json.loads(tool_json)
                    print("\n#----------AGENT TOOL CALL----------#")
                    print(json.dumps(tool_data, indent=4))
                    print("#" + "-" * 35 + "#\n")
                except Exception as e:
                    print("Error parsing CALL_TOOL JSON:", e)
                    print(content)
            elif content.startswith("FINAL_ANSWER:"):
                print("\n#----------AGENT FINAL ANSWER----------#")
                print(content[len("FINAL_ANSWER:") :].strip())
                print("#" + "-" * 35 + "#\n")
            else:
                print("\n#----------AGENT OUTPUT----------#")
                print(content)
                print("#" + "-" * 35 + "#\n")
            break


# -------------------------------------------------------------------------------#
#                                   Main loop                                    #
# -------------------------------------------------------------------------------#

if __name__ == "__main__":
    # please work
    print("#" + "-" * 10 + "Agent_1_test" + 10 * "-" + "#\n")
    print("type 'exit' to close the conversation\n")

    while True:
        userInput = input("\nUser: ").strip()

        if userInput.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        # static agent input for quick restarts
        response = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": "Analyse network 192.168.157.0"}
                ]
            },
        )

        """
        response = agent.invoke(
            {"messages": [{"role": "user", "content": userInput}]},
        )"""

        prettyAgentResponse(response)
