"""意图路由（graph/route.py）测试。"""

from __future__ import annotations

from paper_agent.graph.route import classify_intent


def _llm_reply(payload: str):
    def _fake(messages, **k):
        return payload, {"prompt": 1, "completion": 1}

    return _fake


def test_llm_classification() -> None:
    route = classify_intent(
        "帮我找几篇联邦学习推荐的论文",
        chat_fn=_llm_reply('{"intent": "search", "argument": "federated learning recommendation"}'),
    )
    assert route["intent"] == "search"
    assert route["argument"] == "federated learning recommendation"
    assert route["via"] == "llm"


def test_llm_classification_qa() -> None:
    route = classify_intent(
        "S-MBRec 的损失函数是什么？",
        chat_fn=_llm_reply('{"intent": "qa", "argument": "S-MBRec 的损失函数是什么？"}'),
    )
    assert route["intent"] == "qa"
    assert route["via"] == "llm"


def test_heuristic_fallback_on_llm_failure() -> None:
    def _broken(messages, **k):
        raise RuntimeError("未配置 LLM API Key")

    assert classify_intent("帮我找几篇 GNN 论文", chat_fn=_broken)["intent"] == "search"
    assert classify_intent("BCE 损失是怎么算的？", chat_fn=_broken)["intent"] == "qa"
    assert classify_intent("看看库里的状态", chat_fn=_broken)["intent"] == "status"
    assert classify_intent("退出", chat_fn=_broken)["intent"] == "exit"
    assert classify_intent("你能做什么", chat_fn=_broken)["intent"] == "help"


def test_heuristic_does_not_swallow_question_mark() -> None:
    def _broken(messages, **k):
        raise RuntimeError("no key")

    # 含「找」字但明显是提问的句子，不应被误判为检索
    route = classify_intent("论文里为什么找不到 baseline 对比？", chat_fn=_broken)
    assert route["intent"] == "qa"
