import os
import asyncio
import json
from dotenv import load_dotenv
from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
from pydantic import BaseModel, Field
from typing import List, Dict
from urllib.parse import urlparse, parse_qs

load_dotenv()

# ------------------------------------------------------------------------------- #
#                                       Config                                    #
# ------------------------------------------------------------------------------- #

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.129:5000")

client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)

savedPayload = {}

# ------------------------------------------------------------------------------- #
#                                   Attack vector                                 #
# ------------------------------------------------------------------------------- #


class AttackVector(BaseModel):
    endpoint: str = Field(default=None, description="Targeted endpoint")
    method: str = Field(default=None, description="HTTP method used.")
    params: List[str] = Field(
        default_factory=list, description="Available valid parameters."
    )
    vector_type: str = Field(
        default=None, description="Deciding if the vector is form, simple params,..."
    )
    confidence: int = Field(
        default=None, description="Assigned weight based on the interest."
    )
    cookies: Dict[str, str] = Field(
        default_factory=dict, description="Captured cookies."
    )
    origins: List[str] = Field(
        default_factory=list,
        description="Other endpoint sources for the same attack vector.",
    )


# ------------------------------------------------------------------------------- #
#                                  Helper functions                               #
# ------------------------------------------------------------------------------- #


async def execute_command(command: str):
    return await mcp.call_tool(name="execute_command", arguments={"command": command})


async def serverHealth():
    return await mcp.call_tool(name="server_health", arguments={})


# filter out directories and endpoints with certain http status codes
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
                address = "".join((endpointTarget, endpoint["path"]))
            endpointsDir.append(address)

        if endType == "file" and status in [200, 302]:
            address = "".join((endpointTarget, endpoint["path"]))
            endpointsFile.append(address)

    return {
        "target": endpointTarget,
        "directories": endpointsDir,
        "files": endpointsFile,
    }


async def runKatana(url):
    realCommand = (
        f"/home/kali/go/bin/katana "
        f"-u {url} "
        f"-d 3 "
        f"-jc "
        f"-fx "
        f"-iqp "
        f"-em php "
        f"-ob "
        f"-j "
        f"-silent"
    )

    result = await execute_command(realCommand)
    return result


# fix and parse URLs
def normalizeURL(url: str):
    parsed = urlparse(url)
    path = parsed.path
    queryParams = list(parse_qs(parsed.query).keys())
    return path, queryParams


# split and filter out cookies
def parseCookies(rawCookie: str):
    if not rawCookie or not isinstance(rawCookie, str):
        return {}

    cookies = {}
    ignored = {
        "path",
        "expires",
        "max-age",
        "samesite",
        "secure",
        "httponly",
        "domain",
    }

    parts = rawCookie.split(";")

    for part in parts:
        part = part.strip()

        if "=" not in part:
            continue

        key, value = part.split("=", 1)

        if key.lower() in ignored:
            continue

        cookies[key] = value

    return cookies


# parse initial katana output and create initial attack vectors
def parseKatana(katanaOutput) -> List[AttackVector]:
    all_lines = []
    sources = set()

    for result in katanaOutput:
        if not isinstance(result, list):
            continue

        outputObj = result[0]
        try:
            textObj = json.loads(outputObj.text)
            stdout = textObj.get("stdout", "")
        except:
            continue

        if not stdout:
            continue

        for raw_line in stdout.strip().split("\n"):
            try:
                line = json.loads(raw_line)
            except:
                continue

            request = line.get("request")
            if not request or not request.get("endpoint"):
                continue

            if request.get("source"):
                sources.add(request["source"])

            all_lines.append(line)

    vectors: List[AttackVector] = []

    for line in all_lines:
        request = line.get("request")
        response = line.get("response")

        if not request or not response:
            continue

        endpoint = request.get("endpoint")
        if not endpoint or endpoint in sources:
            continue

        path, queryParams = normalizeURL(endpoint)
        headers = response.get("headers", {})
        rawCookies = headers.get("Set-Cookie")
        parsedCookies = parseCookies(rawCookies) if rawCookies else {}
        endpointRedirect = request.get("source")

        if response.get("forms"):
            for f in response["forms"]:
                if isinstance(f, dict) and f.get("parameters"):
                    params = f.get("parameters", [])
                    vectors.append(
                        AttackVector(
                            endpoint=endpoint,
                            method=f.get("method", "POST"),
                            params=sorted(set(params)),
                            vector_type="forms",
                            confidence=calculateConfidence("forms", len(params)),
                            cookies=parsedCookies,
                            origins=[endpointRedirect] if endpointRedirect else [],
                        )
                    )

        elif "?" in endpoint:
            vectors.append(
                AttackVector(
                    endpoint=endpoint,
                    method=request.get("method"),
                    params=queryParams,
                    vector_type="url_params",
                    confidence=calculateConfidence("url_params", len(queryParams)),
                    cookies=parsedCookies,
                    origins=[endpointRedirect] if endpointRedirect else [],
                )
            )

        elif "application/json" in headers.get("Content-Type", ""):
            vectors.append(
                AttackVector(
                    endpoint=endpoint,
                    method=request.get("method"),
                    params=queryParams,
                    vector_type="xhr_api",
                    confidence=calculateConfidence("xhr_api", len(queryParams)),
                    cookies=parsedCookies,
                    origins=[endpointRedirect] if endpointRedirect else [],
                )
            )

    return vectors


# remove duplicated attack vectores and combine params
def deduplicateOutput(katanaVectorList) -> List[AttackVector]:
    merged = {}

    for vec in katanaVectorList:
        key = (vec.endpoint, vec.method, vec.vector_type)

        if key not in merged:
            merged[key] = AttackVector(
                endpoint=vec.endpoint,
                method=vec.method,
                params=set(vec.params),
                vector_type=vec.vector_type,
                confidence=vec.confidence,
                cookies=dict(vec.cookies) if vec.cookies else {},
                origins=set(vec.origins) if vec.origins else set(),
            )
        else:
            existing = merged[key]

            # merge params
            existing.params = list(set(existing.params).union(set(vec.params)))

            # merge confidence
            existing.confidence = max(existing.confidence, vec.confidence)

            # merge origins
            existing.origins = list(set(existing.origins).union(set(vec.origins)))

            if vec.cookies:
                existing.cookies.update(vec.cookies)

    for v in merged.values():
        v.params = sorted(v.params)
        v.origins = sorted(v.origins)

        # give extra bonus to confidence if the same endpoint is reachable from different origins
        # TLDR: more origins == larger attack surface
        exposure = min(len(v.origins), 3)
        v.confidence += exposure

    return list(merged.values())


async def main(payload):

    filteredEndOutput = filterEndpoints(goBusterData=payload)
    endpoints = filteredEndOutput["files"]

    print(f"\n\nEndpoint files:\n\n{endpoints}")

    allVectors: List[AttackVector] = []

    fixedVectors: List[AttackVector] = []

    for endpoint in endpoints:
        print(f"\n\nEndpoint:{endpoint}")

        katanaResult = await runKatana(url=endpoint)
        vectors = parseKatana(katanaResult)

        allVectors.extend(vectors)

    fixedVectors = deduplicateOutput(allVectors)

    # diabolical print
    print(
        f"\n\nFINAL RESULT:\n\n{json.dumps([v.model_dump() for v in fixedVectors], indent=4)}"
    )

    return fixedVectors


# simple confidence calculator to give priority to better attack vectors
def calculateConfidence(vectorType: str, paramCount: str):
    score = 0

    if vectorType == "forms":
        score += 8
    elif vectorType == "url_params":
        score += 6

    if paramCount > 2:
        score += 1

    return score


# ------------------------------------------------------------------------------- #
#                                    Main test                                    #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("\n" + "-" * 20)
    print("Katana test\n")

    with open("MCP_tools\\gobuster\\gobuster_final_output_example.json", "r") as f:
        payload = json.load(f)

    asyncio.run(main(payload=payload))
