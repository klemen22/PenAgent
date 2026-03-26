import os
import asyncio
import json
from dotenv import load_dotenv
from MCP_tools.mcp_server import KaliToolsClient, setup_mcp_server
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from urllib.parse import urlparse, parse_qs, urlunparse

load_dotenv()

# ------------------------------------------------------------------------------- #
#                                       Config                                    #
# ------------------------------------------------------------------------------- #

KALI_API = os.getenv(key="KALI_API", default="http://192.168.157.137:5000")

client = KaliToolsClient(server_url=KALI_API)
mcp = setup_mcp_server(kali_client=client)

savedPayload = {}

# ------------------------------------------------------------------------------- #
#                                   Attack vector                                 #
# ------------------------------------------------------------------------------- #


class AttackVector(BaseModel):
    endpoint: str = Field(default=None, description="Targeted endpoints.")
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
from urllib.parse import urlparse, parse_qs, urlunparse


def normalizeURL(url: str):
    parsed = urlparse(url)

    clean_url = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",  # params
            "",  # query
            "",  # fragment
        )
    )

    queryParams = list(parse_qs(parsed.query).keys())

    return clean_url, queryParams


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

    # build attack vector
    vectors: List[AttackVector] = []

    for line in all_lines:
        request = line.get("request")
        response = line.get("response")

        if not request or not response:
            continue

        endpoint = request.get("endpoint")
        if not endpoint or endpoint in sources:
            continue

        # normalize url
        path, queryParams = normalizeURL(endpoint)

        originParams = []

        originURL = request.get("source")
        if originURL:
            _, originParams = normalizeURL(originURL)

        # merge params
        queryParams = list(set(queryParams) | set(originParams))

        # remove empty params
        queryParams = [p for p in queryParams if p]

        # deduplicate params
        queryParams = sorted(set(queryParams))

        headers = response.get("headers", {})
        rawCookies = headers.get("Set-Cookie")
        parsedCookies = parseCookies(rawCookies) if rawCookies else {}
        originURL = request.get("source")

        if response.get("forms"):
            for f in response["forms"]:
                if isinstance(f, dict) and f.get("parameters"):

                    formParams = f.get("parameters", [])
                    formParams = [p for p in formParams if p]
                    formParams = sorted(set(formParams))

                    vectors.append(
                        AttackVector(
                            endpoint=path,
                            method=f.get("method") or request.get("method") or "GET",
                            params=formParams,
                            vector_type="forms",
                            confidence=calculateConfidence("forms", len(formParams)),
                            cookies=parsedCookies,
                            origins=[originURL] if originURL else [],
                        )
                    )

        elif queryParams:
            vectors.append(
                AttackVector(
                    endpoint=path,
                    method=request.get("method"),
                    params=queryParams,
                    vector_type="url_params",
                    confidence=calculateConfidence("url_params", len(queryParams)),
                    cookies=parsedCookies,
                    origins=[originURL] if originURL else [],
                )
            )

        elif "application/json" in headers.get("Content-Type", ""):
            vectors.append(
                AttackVector(
                    endpoint=path,
                    method=request.get("method"),
                    params=queryParams,
                    vector_type="xhr_api",
                    confidence=calculateConfidence("xhr_api", len(queryParams)),
                    cookies=parsedCookies,
                    origins=[originURL] if originURL else [],
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


async def main(payload: Dict[str, Any]):
    endpoints_input = payload.get("endpoints", [])

    endpoints = []
    for ep in endpoints_input:
        base = ep.get("base_url", "").rstrip("/")
        path = ep.get("path", "")
        if not path.startswith("/"):
            path = "/" + path
        endpoints.append(f"{base}{path}")
    for e in endpoints:
        print(f"  - {e}")

    allVectors: List[AttackVector] = []
    for endpoint in endpoints:
        print(f"\n\n[CRAWLER] current endpoint: {endpoint}")

        try:
            katanaResult = await runKatana(url=endpoint)
            vectors = parseKatana(katanaResult)
            allVectors.extend(vectors)
        except Exception as e:
            print(f"[CRAWLER ERROR] for endpoint {endpoint}: {e}")

    fixedVectors = deduplicateOutput(allVectors)
    finalVectors = finalVectorFilter(vectors=fixedVectors)

    print(
        f"\n\n[CRAWLER] final result ({len(finalVectors)} vectors):\n{json.dumps(finalVectors, indent=4)}"
    )

    return finalVectors


# simple confidence calculator to give priority to better attack vectors
def calculateConfidence(vectorType: str, paramCount: int, cookies=None):

    score = 0

    if vectorType == "forms":
        score += 10
    elif vectorType == "url_params":
        score += 7
    elif vectorType == "xhr_api":
        score += 4

    score += min(paramCount, 5)

    if cookies:
        score += 2

    return score


def finalVectorFilter(vectors: List[AttackVector]) -> List[Dict[str, Any]]:

    finalVectors = []

    for vector in vectors:
        if len(vector.params) > 0:
            finalVectors.append(vector.model_dump(mode="json"))

    return finalVectors


# ------------------------------------------------------------------------------- #
#                                    Main test                                    #
# ------------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("\n" + "-" * 20)
    print("Katana test\n")

    with open("MCP_tools\\gobuster\\gobuster_final_output_example2.json", "r") as f:
        payload = json.load(f)

    asyncio.run(main(payload=payload))
