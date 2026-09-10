"""Embedding 客户端：OpenAI 兼容封装，读 config.yaml 的 embedding 节。

与 llm.chat 同构：返回纯向量列表，便于上层把 embed_fn 注入测试。
"""

from __future__ import annotations

import time

from openai import OpenAI

from paper_agent.config import load_config, resolve_endpoint

BATCH_SIZE = 32
BATCH_SLEEP = 0.2  # 批间小睡，对免费档友好


def embed(
    texts: list[str],
    *,
    cfg: dict | None = None,
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """批量向量化。顺序与输入一致；空输入返回空列表。"""
    if not texts:
        return []
    cfg = cfg or load_config()
    ep = resolve_endpoint(cfg["embedding"])
    if not ep.api_key:
        raise RuntimeError("未配置 Embedding API Key：请复制 .env.example 为 .env 并填入 Key")
    client = OpenAI(base_url=ep.base_url, api_key=ep.api_key, timeout=120, max_retries=2)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(model=ep.model, input=batch)
        vectors.extend(item.embedding for item in resp.data)
        if start + batch_size < len(texts):
            time.sleep(BATCH_SLEEP)
    return vectors
