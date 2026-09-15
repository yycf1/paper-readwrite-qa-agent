"""意图理解一体化（route.py M5）测试：新格式解析、旧格式兼容、参数容错、域守卫。"""

from __future__ import annotations

from paper_agent.graph.route import classify_intent, optimize_query, summarize_query_understanding


def _llm_reply(payload: str):
    def _fake(messages, **k):
        return payload, {"prompt": 1, "completion": 1}

    return _fake


def test_search_with_structured_params() -> None:
    reply = (
        '{"intent": "search", "argument": "transformer 综述", '
        '"topics": ["transformer survey", "vision transformer review"], '
        '"year_from": 2023, "max_results": 20}'
    )
    route = classify_intent("帮我找 2023 年以后 20 篇 transformer 综述", chat_fn=_llm_reply(reply))
    assert route["intent"] == "search"
    assert route["via"] == "llm"
    assert route["argument"] == "transformer 综述"
    assert route["search"]["topics"] == ["transformer survey", "vision transformer review"]
    assert route["search"]["year_from"] == 2023
    assert route["search"]["max_results"] == 20


def test_old_format_still_compatible() -> None:
    """旧 prompt 格式（intent+argument，无新字段）必须继续可用。"""
    route = classify_intent(
        "帮我找几篇联邦学习推荐的论文",
        chat_fn=_llm_reply('{"intent": "search", "argument": "federated learning recommendation"}'),
    )
    assert route["intent"] == "search"
    assert route["argument"] == "federated learning recommendation"
    s = route["search"]
    assert s["topics"] == [] and s["year_from"] is None and s["max_results"] is None


def test_param_sanitization() -> None:
    """LLM 输出越界/类型错误的参数必须被丢弃，而不是透传。"""
    reply = (
        '{"intent": "search", "argument": "x", "topics": ["ok", " ", 3.5], '
        '"year_from": 1800, "max_results": 999}'
    )
    route = classify_intent("x", chat_fn=_llm_reply(reply))
    s = route["search"]
    assert s["topics"] == ["ok"]  # 空串与非字符串被过滤
    assert s["year_from"] is None and s["max_results"] is None


def test_off_topic_detected() -> None:
    route = classify_intent(
        "帮我写一份周报", chat_fn=_llm_reply('{"intent": "off_topic", "argument": "写周报"}')
    )
    assert route["intent"] == "off_topic"


def test_explore_detected() -> None:
    route = classify_intent(
        "我不知道该看什么论文", chat_fn=_llm_reply('{"intent": "explore", "argument": "推荐方向"}')
    )
    assert route["intent"] == "explore"


def test_clarify_carries_question() -> None:
    reply = (
        '{"intent": "clarify", "argument": "transformer", '
        '"clarify_question": "你是想检索新的 transformer 论文，还是问库里已入库论文的内容？"}'
    )
    route = classify_intent("transformer", chat_fn=_llm_reply(reply))
    assert route["intent"] == "clarify"
    assert "检索" in route["clarify_question"]


def test_domain_question_not_off_topic() -> None:
    """科研相关概念问题必须接待为 qa，不得误判 off_topic。"""
    route = classify_intent(
        "CNN 和深度学习是什么关系",
        chat_fn=_llm_reply('{"intent": "qa", "argument": "CNN 和深度学习是什么关系"}'),
    )
    assert route["intent"] == "qa"


def test_heuristic_fallback_paths() -> None:
    def _broken(messages, **k):
        raise RuntimeError("未配置 LLM API Key")

    assert classify_intent("帮我找几篇 GNN 论文", chat_fn=_broken)["intent"] == "search"
    assert classify_intent("BCE 损失是怎么算的？", chat_fn=_broken)["intent"] == "qa"
    assert classify_intent("看看库里的状态", chat_fn=_broken)["intent"] == "status"
    assert classify_intent("退出", chat_fn=_broken)["intent"] == "exit"
    assert classify_intent("你能做什么", chat_fn=_broken)["intent"] == "help"
    assert classify_intent("不知道看什么论文", chat_fn=_broken)["intent"] == "explore"
    # 启发式兜底不判 off_topic（保守接待）
    assert classify_intent("帮我写周报", chat_fn=_broken)["intent"] == "qa"


def test_heuristic_does_not_swallow_question_mark() -> None:
    def _broken(messages, **k):
        raise RuntimeError("no key")

    # 含「找」字但明显是提问的句子，不应被误判为检索
    route = classify_intent("论文里为什么找不到 baseline 对比？", chat_fn=_broken)
    assert route["intent"] == "qa"


def test_fast_path_skips_llm_for_short_commands() -> None:
    """短而明确的指令走快路径（via=heuristic），不调 LLM；带条件的检索不受影响。"""

    def _must_not_call(messages, **k):
        raise AssertionError("快路径不应调用 LLM")

    assert classify_intent("看看状态", chat_fn=_must_not_call)["intent"] == "status"
    assert classify_intent("库里有多少篇论文", chat_fn=_must_not_call)["intent"] == "status"
    assert classify_intent("不知道该看什么论文", chat_fn=_must_not_call)["intent"] == "explore"
    assert classify_intent("退出", chat_fn=_must_not_call)["intent"] == "exit"
    assert classify_intent("你能做什么", chat_fn=_must_not_call)["intent"] == "help"


def test_long_question_with_keyword_word_still_goes_to_llm() -> None:
    """长句含「状态」等词不被快路径抢跑，仍走 LLM 语义判断。"""
    reply = '{"intent": "qa", "argument": "论文里的状态更新机制是什么"}'
    route = classify_intent(
        "论文里的状态更新机制是什么", chat_fn=_llm_reply(reply)
    )
    assert route["intent"] == "qa" and route["via"] == "llm"


def test_search_never_fast_paths() -> None:
    """检索必须走 LLM：年份/数量参数只能由它抽取，快路径抢跑会丢参数。"""

    def _must_not_call(messages, **k):
        raise AssertionError("检索不应走快路径")

    reply = (
        '{"intent": "search", "argument": "transformer 综述", '
        '"topics": ["transformer survey"], "year_from": 2023, "max_results": 20}'
    )
    route = classify_intent("帮我找 2023 年以后 20 篇 transformer 综述", chat_fn=_llm_reply(reply))
    assert route["intent"] == "search" and route["via"] == "llm"
    assert route["search"]["year_from"] == 2023


def test_summarize_understanding() -> None:
    route = {
        "intent": "search", "via": "llm", "argument": "t",
        "search": {"topics": ["gnn survey"], "year_from": 2023, "max_results": None},
    }
    text = summarize_query_understanding(route)
    assert "gnn survey" in text and "2023" in text
    assert summarize_query_understanding({"intent": "off_topic", "via": "llm"}).startswith("域外")


# ---------------------------------------------------------------------------
# optimize_query（pa search 查询优化）
# ---------------------------------------------------------------------------


def test_optimize_query_translates_and_expands() -> None:
    reply = (
        '{"topics": ["medical image segmentation deep learning", "graph neural network"], '
        '"year_from": 2022, "max_results": null}'
    )
    plan = optimize_query("基于深度学习的医学影像分割和GNN，2022年起", chat_fn=_llm_reply(reply))
    assert plan["optimized"] is True
    assert plan["topics"] == ["medical image segmentation deep learning", "graph neural network"]
    assert plan["year_from"] == 2022


def test_optimize_query_falls_back_on_failure() -> None:
    def _broken(messages, **k):
        raise RuntimeError("no key")

    plan = optimize_query("graph neural network", chat_fn=_broken)
    assert plan["optimized"] is False
    assert plan["topics"] == ["graph neural network"]


def test_optimize_query_falls_back_on_empty_topics() -> None:
    plan = optimize_query("x", chat_fn=_llm_reply('{"topics": [], "year_from": null}'))
    assert plan["optimized"] is False
    assert plan["topics"] == ["x"]
