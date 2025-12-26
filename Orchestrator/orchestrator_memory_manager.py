import sqlite3


def showMemory():
    connect = sqlite3.connect("orchestrator_memory.sqlite")
    cursor = connect.cursor()

    cursor.execute("SELECT * FROM store;")
    memories = cursor.fetchall()

    for memory in memories:
        print(f"    >{memory}")

    print("\nDone printing memories...")

    connect.close()
    return


def deleteMemory():
    connect = sqlite3.connect("orchestrator_memory.sqlite")
    cursor = connect.cursor()

    cursor.execute("DELETE * FROM store;")
    print("\nFinished deleting memories...")

    connect.commit()
    connect.close()
    return


if __name__ == "__main__":

    while True:

        print("\n" + "-" * 20)
        print("MEMORY MANAGER")
        print("-" * 20 + "\n")

        print("Select desired action (write 'exit' to close manager): ")
        print("1. Print memory\n2. Wipe memory\n3. Exit")

        choice = input("> ").strip().lower()

        if choice == "exit" | "quit":
            break
        elif choice.strip(".") == "1":
            showMemory()
            continue
        elif choice.strip(".") == "2":
            deleteMemory()
            continue
        elif choice.strip(".") == "3":
            print("\nExiting...")
            break
        else:
            print("\nInvalid input...")
            continue
