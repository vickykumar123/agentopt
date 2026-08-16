"""Entry point: input format '<user_id>|<message>'."""

from graph import build


def run(input_text):
    raw = input_text or "1|hello"
    user_id, _, message = raw.partition("|")
    state = build().invoke({"user_id": int(user_id or "1"), "message": message or "hello"})
    return f"[{state['intent']}] {state['answer']}"
