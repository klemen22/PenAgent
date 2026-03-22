import logging
import os
from datetime import datetime

token_totals = {
    "prompt_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def logMetadata(agent_name: str, metadata):
    try:

        prompt_tokens = metadata.get("prompt_eval_count", "")
        output_tokens = metadata.get("eval_count", "")
        total = prompt_tokens + output_tokens

        tokenCount = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
        }

        # update global counter
        token_totals["prompt_tokens"] += prompt_tokens
        token_totals["output_tokens"] += output_tokens
        token_totals["total_tokens"] += total

        logging.getLogger(agent_name).info(tokenCount)

    except Exception as e:
        print(f"[ERROR]: failed saving metadata: {str(e)}")


def setupMetadataLogger(agent_name: str):
    count = getLogCount(agent_name=agent_name)
    file = f"metadata/{agent_name}/{agent_name}_metadata_{count + 1}_{datetime.now().strftime("%Y-%m-%d")}.log"

    logger = logging.getLogger(agent_name)
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(file, mode="w")
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def getLogCount(agent_name: str):
    path = f"metadata/{agent_name}"
    logCount = 0

    if not os.path.exists(path=path):
        os.makedirs(path, exist_ok=True)
        return logCount

    for log in os.listdir(path=path):
        if os.path.isfile(os.path.join(path, log)):
            logCount += 1

    return logCount


def logTotalTokens(agent_name: str):

    summary = {
        "TOTAL_PROMPT_TOKENS": token_totals["prompt_tokens"],
        "TOTAL_OUTPUT_TOKENS": token_totals["output_tokens"],
        "TOTAL_TOKENS": token_totals["total_tokens"],
    }

    logging.getLogger(agent_name).info("========== TOKEN SUMMARY ==========")
    logging.getLogger(agent_name).info(summary)
