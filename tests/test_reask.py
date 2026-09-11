"""补测：提取层对空回复/坏 JSON 的重问恢复（离线注入）。"""

from paper_agent.extraction.experiment_flow import extract_experiment


def test_reask_recovers_from_empty_reply():
    calls = []

    def flaky_chat(messages, **kw):
        calls.append(messages)
        if len(calls) == 1:
            return "", {"prompt": 10, "completion": 0, "model": "fake"}  # 空回复
        # 第二次带重问指令，返回合法 JSON
        assert messages[-1]["role"] == "user" and "JSON" in messages[-1]["content"]
        return (
            '{"research_goal": "恢复成功", "datasets": [], "metrics": []}',
            {"prompt": 20, "completion": 5, "model": "fake"},
        )

    result = extract_experiment("## Abstract\nGNN paper.", chat_fn=flaky_chat)
    assert result["research_goal"] == "恢复成功"
    assert len(calls) == 2
    assert result["token_usage"]["prompt"] == 30  # 两次调用累计


def test_reask_gives_up_with_clear_error():
    def always_bad(messages, **kw):
        return "完全不是 JSON", {"prompt": 1, "completion": 1, "model": "fake"}

    try:
        extract_experiment("text", chat_fn=always_bad)
        raise AssertionError("应当抛 ValueError")
    except ValueError as exc:
        assert "重问后仍无法解析" in str(exc)
