def sqlmapOutputParser(toolOutput):

    stdout = ""
    keywords = [
        "parameter",
        "injectable",
        "dbms",
        "warning",
        "critical",
        "all tested parameters",
        "appears",
        "does not",
    ]

    filteredOutput = []

    if isinstance(toolOutput, tuple):
        result = toolOutput[1].get("result", {})
        stdout = result.get("stdout", "")

    elif isinstance(toolOutput, dict):
        stdout = toolOutput.get("stdout", "")

    if stdout != "":
        lines = stdout.splitlines()

        for line in lines:
            for keyword in keywords:
                if keyword.lower() in line.lower():
                    filteredOutput.append(line)

        return "\n".join(filteredOutput)
    else:
        return ""
