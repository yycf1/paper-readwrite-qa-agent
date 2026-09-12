"""LLM 客户端：OpenAI 兼容封装，读 config.yaml 的 llm 节与 .env 的 Key。

extraction 与 rag/qa 共用；返回 (文本, token用量)，方便记账。
免费档限流（如智谱 429/1305）是常态，chat 内置指数退避重试。
"""

from __future__ import annotations

import json
import re
import time

from openai import OpenAI, RateLimitError

from paper_agent.config import load_config, resolve_endpoint

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = (10, 20, 40)


def chat(
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    cfg: dict | None = None,
    thinking: bool | None = None,
) -> tuple[str, dict]:
    """一次对话补全，限流时自动退避重试。返回 (回复文本, {"prompt": n, "completion": n, "model": name})。

    thinking：GLM 等混合推理模型的思考开关。None=平台默认（适合提取/报告等复杂任务）；
    False=禁用思考——意图分类、查询优化等简单任务必须禁用，否则思考 tokens 会
    吃光 max_tokens 导致 content 为空（M5 实测踩坑）。
    """
    cfg = cfg or load_config()
    ep = resolve_endpoint(cfg["llm"])
    if not ep.api_key:
        raise RuntimeError("未配置 LLM API Key：请复制 .env.example 为 .env 并填入 Key")
    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=180, max_retries=2)
    extra: dict = {}
    if thinking is not None:
        extra["extra_body"] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    last_exc: RateLimitError | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=ep.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra,
            )
            usage = resp.usage
            return (
                resp.choices[0].message.content or "",
                {
                    "prompt": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion": getattr(usage, "completion_tokens", 0) or 0,
                    "model": ep.model,
                },
            )
        except RateLimitError as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                wait = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
                time.sleep(wait)
    raise RuntimeError(
        f"LLM 连续 {_MAX_ATTEMPTS} 次被限流（免费档调用频率限制），已退避重试无效：{last_exc}"
    ) from last_exc


def parse_json_reply(text: str) -> dict:
    """解析 LLM 回复中的 JSON 对象：剥代码围栏，兜底截取首个 { 到最后一个 }。"""
    fenced = _JSON_FENCE_RE.match(text.strip())
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"回复中不含 JSON 对象：{text[:200]!r}")
        return json.loads(text[start : end + 1])
