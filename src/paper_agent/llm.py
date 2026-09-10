"""LLM 客户端：OpenAI 兼容封装，读 config.yaml 的 llm 节与 .env 的 Key。

extraction 与（M3 的）rag/qa 共用；返回 (文本, token用量)，方便记账。
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

from paper_agent.config import load_config, resolve_endpoint

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def chat(
    messages: list[dict],
    *,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    cfg: dict | None = None,
) -> tuple[str, dict]:
    """一次对话补全。返回 (回复文本, {"prompt": n, "completion": n, "model": name})。"""
    cfg = cfg or load_config()
    ep = resolve_endpoint(cfg["llm"])
    if not ep.api_key:
        raise RuntimeError("未配置 LLM API Key：请复制 .env.example 为 .env 并填入 Key")
    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=180, max_retries=2)
    resp = client.chat.completions.create(
        model=ep.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
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
