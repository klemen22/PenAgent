from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# initialize LMstudio llm

llm = ChatOpenAI(
    model="openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix",
    base_url="http://localhost:1234/v1",
    api_key="yes",
)

context = """You are an autonomous assistant. You are ready to help the user as best as you can."""

# create agent
agent = create_agent(model=llm, system_prompt=context)

if __name__ == "__main__":

    while True:
        userInput = input("\nUser: ")

        if userInput.lower() in ["exit", "quit"]:
            print("Ending...")
            break

        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"{userInput}"}]}
        )

        print(f"AI: {response["messages"][-1].content}")
