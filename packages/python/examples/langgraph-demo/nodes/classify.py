"""Intent classification node."""

from llm import chat

CLASSIFY_PROMPT = "You will receive input in the form `<customer_id>|<message>`.\n\nCUSTOMERS:\n1|Priya\n2|Rahul\n3|Mei\n\nProcedure:\n1. Split the input on `|` to get the customer_id and the message.\n2. Look up the customer's name in the CUSTOMERS table above.\n3. Output exactly one line in this format: `Hi <name>! <Category>` where <Category> is a one-word category label for the message.\n\nYour reply MUST begin with `Hi ` followed by the looked-up name. Never output the category alone.\n\nExamples:\nInput: 2|my app keeps crashing\nOutput: Hi Rahul! Technical\nInput: 3|when does my subscription renew\nOutput: Hi Mei! Billing\nInput: 1|how much is my plan\nOutput: Hi Priya! Billing\n\nRemember: the output must start with `Hi <name>!` — a bare one-word label is invalid."


def classify(state):
    intent = chat(CLASSIFY_PROMPT, state["message"], temperature=0.2)
    return {"intent": intent.strip().lower()[:40]}
