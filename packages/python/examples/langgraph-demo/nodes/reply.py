"""Reply node: answers using the customer's database record."""

from llm import chat

REPLY_PROMPT = 'You are a terse support bot. Begin your reply by greeting the user by name (e.g., "Hi Priya"), then answer in one short sentence, and end with "Anything else I can help with?"'


def reply(state):
    user = state["user"]
    facts = f"Customer name: {user['name']}. Plan: {user['plan']}. Monthly price: {user['price']}."
    answer = chat(REPLY_PROMPT, f"{facts}\n\nCustomer says: {state['message']}", temperature=0.7)
    return {"answer": answer}
