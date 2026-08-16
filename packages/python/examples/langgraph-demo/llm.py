"""Shared model client (Fireworks via the OpenAI-compatible endpoint)."""

import os

from langchain_openai import ChatOpenAI


def chat(system_prompt, user_text, temperature=0.7):
    model = ChatOpenAI(
        model="accounts/fireworks/models/kimi-k2p6",
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
        temperature=temperature,
        max_tokens=2000,
    )
    result = model.invoke([("system", system_prompt), ("user", user_text)])
    return result.content if isinstance(result.content, str) else str(result.content)
