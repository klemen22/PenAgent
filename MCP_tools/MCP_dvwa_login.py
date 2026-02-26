from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
import os
from dotenv import load_dotenv
import asyncio

# ------------------------------------------------------------------------------- #
#                                       Config                                    #
# ------------------------------------------------------------------------------- #
load_dotenv()

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.137:5000")
URL = os.getenv(key="TEST_TARGET", default="http://192.168.157.136")

client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)

# ------------------------------------------------------------------------------- #
#                                   Login                                         #
# ------------------------------------------------------------------------------- #


async def dvwa_login(baseURL):
    command = f"/home/kali/DVWA_login/venv/bin/python /home/kali/DVWA_login/dvwa_login.py {baseURL}"
    response = await mcp.call_tool(
        name="execute_command", arguments={"command": command}
    )
    return response[1]["result"]["stdout"]


async def serverHealth():
    return await mcp.call_tool(name="server_health", arguments={})


# ------------------------------------------------------------------------------- #
#                                    Test                                         #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print(asyncio.run(dvwa_login(URL)))
