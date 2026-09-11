"""llm.chat 限流重试测试：monkeypatch 假客户端，不联网。"""

import pytest
from openai import RateLimitError


class _FakeResp:
    def __init__(self, status_code=429):
        self.status_code = status_code
        self.request = type("Req", (), {})()  # openai SDK 构造异常时会读 response.request
        self.headers = {}  # 以及 response.headers.get("x-request-id")


class _FakeUsage:
    prompt_tokens = 11
    completion_tokens = 7


class _FakeMessage:
    content = "ok"


class _FakeChoice:
    message = _FakeMessage()


class _Completions:
    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RateLimitError("429", response=_FakeResp(), body=None)
        resp = type("R", (), {})()
        resp.choices = [_FakeChoice()]
        resp.usage = _FakeUsage()
        return resp


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, completions, **kw):
        self.chat = _Chat(completions)


def test_chat_retries_on_rate_limit(monkeypatch):
    from paper_agent import llm

    completions = _Completions(fail_times=2)  # 前 2 次 429，第 3 次成功
    monkeypatch.setattr(llm, "OpenAI", lambda **kw: _FakeClient(completions))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        llm, "resolve_endpoint",
        lambda raw: type("E", (), {"base_url": "http://x", "model": "m", "api_key": "k"})(),
    )
    monkeypatch.setattr(llm, "load_config", lambda: {"llm": {}})

    text, usage = llm.chat([{"role": "user", "content": "hi"}])
    assert text == "ok"
    assert usage == {"prompt": 11, "completion": 7, "model": "m"}
    assert completions.calls == 3


def test_chat_gives_up_after_max_attempts(monkeypatch):
    from paper_agent import llm

    completions = _Completions(fail_times=99)
    monkeypatch.setattr(llm, "OpenAI", lambda **kw: _FakeClient(completions))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        llm, "resolve_endpoint",
        lambda raw: type("E", (), {"base_url": "http://x", "model": "m", "api_key": "k"})(),
    )
    monkeypatch.setattr(llm, "load_config", lambda: {"llm": {}})

    with pytest.raises(RuntimeError, match="限流"):
        llm.chat([{"role": "user", "content": "hi"}])
    assert completions.calls == llm._MAX_ATTEMPTS


def test_chat_missing_key():
    from paper_agent import llm

    monkey_cfg = {"llm": {}}
    monkeypatch_set = {"api_key": ""}
    llm_module = llm
    llm_module_load = lambda: {"llm": {}}
    # 无 Key 直接报错（resolve_endpoint 返回空 api_key）
    llm_module.load_config = llm_module_load
    llm_module.resolve_endpoint = lambda raw: type(
        "E", (), {"base_url": "http://x", "model": "m", "api_key": monkeypatch_set["api_key"]}
    )()
    with pytest.raises(RuntimeError, match="Key"):
        llm.chat([{"role": "user", "content": "hi"}], cfg=monkey_cfg)
