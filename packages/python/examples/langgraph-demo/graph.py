"""The LangGraph pipeline: lookup -> classify -> reply."""

from typing import TypedDict

from db import fetch_user
from langgraph.graph import END, START, StateGraph
from nodes.classify import classify
from nodes.reply import reply


class State(TypedDict, total=False):
    user_id: int
    message: str
    user: dict
    intent: str
    answer: str


def lookup(state: State):
    return {"user": fetch_user(state["user_id"])}


def build():
    graph = StateGraph(State)
    graph.add_node("lookup", lookup)
    graph.add_node("classify", classify)
    graph.add_node("reply", reply)
    graph.add_edge(START, "lookup")
    graph.add_edge("lookup", "classify")
    graph.add_edge("classify", "reply")
    graph.add_edge("reply", END)
    return graph.compile()
