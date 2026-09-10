"""LangGraph hello world：两节点线性图，验证编排框架在本机可用。"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class HelloState(TypedDict):
    message: str
    steps: list[str]


def _prepare(state: HelloState) -> dict:
    return {"steps": state["steps"] + ["prepare"]}


def _finish(state: HelloState) -> dict:
    return {
        "message": f"hello from langgraph ({len(state['steps']) + 1} steps)",
        "steps": state["steps"] + ["finish"],
    }


def build_hello_graph():
    graph = StateGraph(HelloState)
    graph.add_node("prepare", _prepare)
    graph.add_node("finish", _finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "finish")
    graph.add_edge("finish", END)
    return graph.compile()


def run_hello() -> dict:
    return build_hello_graph().invoke({"message": "", "steps": []})
