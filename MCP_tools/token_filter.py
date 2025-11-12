# filter out all unknown / illegal tokens via langchain middleware
from langchain.agents.middleware import wrap_model_call
from langchain.messages import AIMessage
import json

illegalTokens = ["<|constrain|>", "<|thought|>", "<|commentary|>", "<|output|>"]


@wrap_model_call
def filterModel(request, handler):
    response = handler(request)

    for msg in response.result:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            clean_text = msg.content
            for token in illegalTokens:
                clean_text = clean_text.replace(token, "")
            msg.content = clean_text

    return response
