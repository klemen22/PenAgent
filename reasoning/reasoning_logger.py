import logging
import os
from datetime import datetime


def logReasoning(agentName: str, reasoning):
    try:
        logging.getLogger(f"{agentName}_reasoning").debug(msg=reasoning)
    except Exception as e:
        print(f"[ERROR]: failed saving reasoning for agent {agentName}: {str(e)}")


def setupReasoningLogger(agentName: str):

    logID = createLogID(agentName=agentName)
    file = f"reasoning/{agentName}/{logID}"

    logger = logging.getLogger(f"{agentName}_reasoning")
    logger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(file, mode="w")
    format = logging.Formatter("%(asctime)s | %(message)s")

    fileHandler.setFormatter(format)
    logger.addHandler(fileHandler)

    return logger


def createLogID(agentName: str):

    path = f"reasoning/{agentName}"
    logCount = 0

    if not os.path.exists(path):
        os.makedirs(path)

    else:
        for log in os.listdir(path=path):
            if os.path.isfile(os.path.join(path, log)):
                logCount += 1

    return (
        f"{agentName}_reasoning_{logCount+1}_{datetime.now().strftime("%Y-%m-%d")}.log"
    )
