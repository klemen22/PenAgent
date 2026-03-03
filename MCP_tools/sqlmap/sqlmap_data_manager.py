import base64
import tarfile
import io
import os
from dotenv import load_dotenv
from pathlib import Path
from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
import asyncio

load_dotenv()

# ------------------------------------------------------------------------------- #
#                                       Config                                    #
# ------------------------------------------------------------------------------- #

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.137:5000")
TEST_ADDR = os.getenv(key="TEST_IP", default="192.168.157.136")

client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)
dataDir = Path("MCP_tools/sqlmap/retrieved_data/")

# ------------------------------------------------------------------------------- #
#                                        Main                                     #
# ------------------------------------------------------------------------------- #


async def execute_command(command: str):
    return await mcp.call_tool(name="execute_command", arguments={"command": command})


async def deleteHistory(targetAddress: str):
    command = (
        f"rm -f /tmp/sqlmap_dump.tar.gz && "
        f"rm -rf /home/kali/.local/share/sqlmap/output/{targetAddress}"
    )

    _, meta = await execute_command(command=command)
    result = meta["result"]

    if not result["success"]:
        return {"status": "failed", "message": result["stderr"]}

    return {"status": "deleted", "message": ""}


async def retrieveData(targetAddress: str):

    command = (
        f"tar -czf /tmp/sqlmap_dump.tar.gz "
        f"-C /home/kali/.local/share/sqlmap/output "
        f"{targetAddress} "
        f"&& base64 /tmp/sqlmap_dump.tar.gz"
    )

    try:

        _, meta = await execute_command(command=command)

        result = meta["result"]

        if not result["success"]:
            return {"status": "failed", "message": result["stderr"]}

        b64_data = result["stdout"]

        archiveBytes = base64.b64decode(b64_data)

        with tarfile.open(fileobj=io.BytesIO(archiveBytes), mode="r:gz") as tar:
            tar.extractall(path=dataDir)

        return {
            "status": "success",
            "message": "",
            "data_path": str(dataDir),
            "files": os.listdir(path=f"{dataDir}/{targetAddress}"),
        }
    except Exception as e:
        return {
            "status": "failed",
            "message": str(e),
        }


# ------------------------------------------------------------------------------- #
#                                        Test                                     #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    result = asyncio.run(retrieveData(targetAddress=TEST_ADDR))
    print("\n\n[FINAL RESULT]:\n")

    if not result["message"]:
        print(f"> status: {result["status"]}")
        print(f"> data path: {result["data_path"]}")
        print(f"> files: {result["files"]}")
    else:
        print(f"> status: {result["status"]}")
        print(f"> message: {result["message"]}")"""

    result = asyncio.run(deleteHistory(targetAddress=TEST_ADDR))

    if not result["message"]:
        print(f"> status: {result["status"]}")
    else:
        print(f"> status: {result["status"]}")
        print(f"> message: {result["message"]}")
