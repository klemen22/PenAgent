import curses
import os

ENV_FILE = ".env"

REQUIRED_VARS = {
    "LM_API": "API for connecting to LLM",
    "TOKEN_WINDOW_SIZE": "Custom LLM token window size",
    "LLM_MODEL": "Exact name of selected LLM",
}


def readEnv():
    env = {}

    if not os.path.exists(ENV_FILE):
        return env

    with open(ENV_FILE, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                env[key] = value

    return env


def writeEnv(env: dict):
    with open(ENV_FILE, "w") as f:
        for key, value in env.items():
            f.write(f"{key} = {value}\n")


def setupHeader(stdscr):
    stdscr.addstr(1, 2, "Settings", curses.A_BOLD)
    stdscr.addstr(2, 2, "=" * 20)


def setupInputField(stdscr, y, label, default=""):
    curses.echo()

    stdscr.addstr(y, 4, f"{label}: ")
    stdscr.addstr(y, 20, " " * 40)

    if default:
        stdscr.addstr(y, 20, default)

    stdscr.move(y, 20)

    value = stdscr.getstr(y, 20, 40).decode("utf-8")

    curses.noecho()

    if value.strip() == "":
        return default

    return value


def setupSettings(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)

    stdscr.clear()

    env = readEnv()

    setupHeader(stdscr=stdscr)

    stdscr.addstr(4, 2, "Configure environment:", curses.color_pair(1))

    y = 6

    for key, description in REQUIRED_VARS.items():
        default = env.get(key, "")

        value = setupInputField(stdscr, y, description, default)

        env[key] = value
        y += 2

    writeEnv(env)

    stdscr.addstr(y + 1, 2, "Environment saved!", curses.color_pair(2))
    stdscr.addstr(y + 3, 2, "Press any key to exit...")

    stdscr.refresh()
    stdscr.getch()


def main():
    curses.wrapper(setupSettings)


if __name__ == "__main__":
    main()
