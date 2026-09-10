"""RAG 评测：检索命中率与（可选的）答案关键词命中率。

评测集见 eval/evalset.json：每条 case 带期望出处（章节名子串，与向量库
metadata.section 匹配）与期望关键词组（组内 OR、组间 AND）。判定函数
保持纯逻辑，便于离线测试；检索/生成由 CLI 命令注入。
"""

from __future__ import annotations

import json
from pathlib import Path


def load_evalset(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError(f"评测集 {path} 缺少非空 cases 数组")
    for i, case in enumerate(data["cases"]):
        for key in ("id", "question"):
            if not case.get(key):
                raise ValueError(f"评测集第 {i} 条缺少 {key}")
    return data


def judge_retrieval(case: dict, hits: list) -> bool | None:
    """期望章节（子串、忽略大小写）是否出现在 top-k 命中里。无期望时返回 None。"""
    sections = case.get("expect_sections")
    if not sections:
        return None
    joined = " | ".join(h.section.lower() for h in hits)
    return any(s.lower() in joined for s in sections)


def judge_answer(case: dict, answer: str) -> bool | None:
    """关键词组判定：组间 AND、组内 OR（忽略大小写）。无期望时返回 None。"""
    groups = case.get("expect_keywords")
    if not groups:
        return None
    low = answer.lower()
    return all(any(alt.lower() in low for alt in group) for group in groups)
