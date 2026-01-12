import os
import asyncio
from dotenv import load_dotenv
from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
from urllib.parse import urljoin
import json

load_dotenv()

# -------------------------------------------------------------------------------#
#                                      Config                                    #
# -------------------------------------------------------------------------------#


KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")
client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)
savedPayload = {}

# default values for now
HAKRAWLER_DEPTH = 2
HAKRAWLER_THREADS = 8

# -------------------------------------------------------------------------------#
#                                  Helper functions                              #
# -------------------------------------------------------------------------------#


async def execute_command(command: str):
    return await mcp.call_tool(name="execute_command", arguments={"command": command})


async def serverHealth():
    return await mcp.call_tool(name="server_health", arguments={})


def filterEndpoints(goBusterData: dict):
    endpointsData = goBusterData["endpoints"]
    endpointTarget = goBusterData["target"]

    endpointsDir = []
    endpointsFile = []

    for endpoint in endpointsData:
        status = endpoint["status"]
        endType = endpoint["type"]

        if status >= 400:
            continue

        if endType == "directory" and status in [200, 201, 301, 302]:
            if endpoint["redirect"]:
                address = endpoint["redirect_address"]
            else:
                address = "".join(endpointTarget, endpoint["path"])

            endpointsDir.append(address)

        if endType == "file" and status in [200, 302]:
            address = "".join((endpointTarget, endpoint["path"]))

            endpointsFile.append(address)

    return {
        "target": endpointTarget,
        "directories": endpointsDir,
        "files": endpointsFile,
    }


async def runHakrawler(url: dict):
    dirEndpoints = url["directories"]

    # first run for 1
    command = f"hakrawler -d {HAKRAWLER_DEPTH} -json -t {HAKRAWLER_THREADS} -u"

    # TODO: chnage crawler HAKRAWLER is bad

    return


# -------------------------------------------------------------------------------#
#                                       Main test                                #
# -------------------------------------------------------------------------------#
if __name__ == "__main__":
    print("\n" + "-" * 20)
    print("Crawler test\n")

    # hakrawler test
    # result = asyncio.run(execute_command("hakrawler --help"))
    # print(f"execute_command output:\n{result}")

    with open("MCP_tools\gobuster\gobuster_final_output_example.json", "r") as f:
        payload = json.load(f)

    print(f"Payload:\n\n{payload}\n\n")
    print(filterEndpoints(goBusterData=payload))
