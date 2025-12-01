# filter out all unknown / illegal tokens via langchain middleware
from langchain.agents.middleware import (
    after_model,
    AgentState,
    before_model,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage

illegal_tokens = ["<|constrain|>", "<|thought|>", "<|commentary|>", "<|output|>"]


@after_model
def filterModel(state: AgentState, handler):
    if "messages" in state:
        for msg in state["messages"]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                content = msg.content
                for token in illegal_tokens:
                    content = content.replace(token, "")
                msg.content = content
    return None
