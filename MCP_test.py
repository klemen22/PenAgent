import os, requests
from dotenv import load_dotenv

load_dotenv()

MCP_HOST = "192.168.157.129"
MCP_PORT = 5000
API_KEY = os.getenv("MCP_API_KEY")

url = f"http://{MCP_HOST}:{MCP_PORT}/health"

headers = {}
if API_KEY:
    headers = {"Authorization": f"Bearer {API_KEY}"}

response = requests.get(url=url, headers=headers, timeout=100)
print(f"\nStatus code: {response.status_code}")
print(f"json: \n{response.text}")
