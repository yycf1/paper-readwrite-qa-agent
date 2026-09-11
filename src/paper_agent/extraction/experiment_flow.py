"""实验知识提取：解析后的论文文本 → experiment.json。

设计要点（对应 PLAN.md §6）：
- 逐字段 confidence + unverified_claims 防编造；
- 代码/数据集链接用正则从原文确定性提取，与 LLM 结果取并集；
- 本模块是二期复现功能的唯一输入契约，字段宁多勿缺。
"""

from __future__ import annotations

import re
from datetime import datetime, UTC

from paper_agent.llm import chat, parse_json_reply

MAX_INPUT_CHARS = 36_000  # 超长论文截断（DeepSeek-V3 上下文足够，主要控成本）

_CODE_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|gitee\.com)/[\w.\-/]+",
    re.IGNORECASE,
)
_DATA_URL_RE = re.compile(
    r"https?://[\w.\-/]*?(?:huggingface\.co/datasets|zenodo\.org|figshare\.com|"
    r"kaggle\.com/datasets|paperswithcode\.com/dataset)[\w.\-/]*",
    re.IGNORECASE,
)
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")

_FIELDS = [
    "research_goal", "method", "datasets", "hyperparameters", "experiment_steps",
    "metrics", "main_results", "conclusions", "repro_assets", "reproducibility_notes",
]

_PROMPT = """你是论文实验流程结构化助手。只依据我提供的论文文本提取信息，禁止编造。

任务：输出一个 JSON 对象（不要输出任何其它内容），字段如下：
{{
  "research_goal": "一句话研究目标",
  "method": {{"name": "方法名称", "architecture": "模型/架构描述", "key_components": ["关键组件"]}},
  "datasets": [{{"name": "数据集名", "source": "来源/获取方式", "split": "划分方式（未说明则留空）"}}],
  "hyperparameters": {{"参数名": "取值（文中未说明的不要写）"}},
  "experiment_steps": [{{"step": 1, "description": "步骤概述", "details": "关键细节"}}],
  "metrics": [{{"name": "指标名", "value": "论文报告值", "baseline": "对比基线及取值（未说明则留空）"}}],
  "main_results": "主要结果描述",
  "conclusions": "主要结论",
  "repro_assets": {{"code_url": ["官方或第三方代码仓库"], "dataset_urls": ["数据集下载地址"], "project_page": ["项目主页"]}},
  "reproducibility_notes": "可复现性评估：代码/数据是否公开、作者是否给出足够细节、缺失哪些信息",
  "confidence": {{"字段名": "high|medium|low"}}   // 对每个非空字段给置信度；依据不足写 low
}}

要求：
1. 文中未明确说明的信息一律不填，并在 "unverified_claims" 数组里列出你为补全结构所做的推断（没有推断则给空数组）；
2. 链接必须来自原文，不要自己构造；
3. confidence 里未提及的字段可不写。

论文文本（可能被截断）：
---
{paper_text}
---"""


def _strip_trailing_punct(urls: list[str]) -> list[str]:
    """URL 尾部标点是句子成分，不是链接的一部分。"""
    return [u.rstrip(".,;:)」】") for u in urls]


def extract_repro_urls(text: str) -> dict[str, list[str]]:
    """正则确定性提取复现资产链接（LLM 结果的兜底与校验）。"""
    seen = lambda urls: list(dict.fromkeys(_strip_trailing_punct(urls)))
    return {
        "code_url": seen(_CODE_URL_RE.findall(text)),
        "dataset_urls": seen(_DATA_URL_RE.findall(text)),
    }


def merge_repro_assets(llm_assets: dict, regex_assets: dict[str, list[str]]) -> dict:
    """LLM 与正则结果取并集（去重、保序，LLM 的在前）。"""
    merged: dict[str, list[str]] = {}
    for key in ("code_url", "dataset_urls", "project_page"):
        llm_urls = [u for u in (llm_assets.get(key) or []) if isinstance(u, str) and u]
        merged[key] = list(dict.fromkeys(llm_urls + regex_assets.get(key, [])))
    return merged


def normalize_experiment(data: dict, *, paper_text: str, usage: dict) -> dict:
    """校验并补全 LLM 输出：必填字段缺失给空值；复现资产并入正则结果。"""
    if not isinstance(data, dict):
        raise ValueError("LLM 输出不是 JSON 对象")
    out = {
        key: (value if value is not None else _EMPTY_DEFAULTS[key])
        for key, value in ((k, data.get(k)) for k in _FIELDS)
    }
    out["confidence"] = data.get("confidence") or {}
    out["unverified_claims"] = [
        c for c in (data.get("unverified_claims") or []) if isinstance(c, str) and c
    ]
    llm_assets = out["repro_assets"] if isinstance(out["repro_assets"], dict) else {}
    out["repro_assets"] = merge_repro_assets(llm_assets, extract_repro_urls(paper_text))
    out["token_usage"] = usage
    out["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return out


_EMPTY_DEFAULTS: dict[str, object] = {
    "research_goal": "",
    "method": {"name": "", "architecture": "", "key_components": []},
    "datasets": [],
    "hyperparameters": {},
    "experiment_steps": [],
    "metrics": [],
    "main_results": "",
    "conclusions": "",
    "repro_assets": {"code_url": [], "dataset_urls": [], "project_page": []},
    "reproducibility_notes": "",
}


_REASK_INSTRUCTION = (
    "上面的回复无法解析为 JSON。请严格重新回答：只输出一个 JSON 对象，"
    "不要任何解释、Markdown 围栏或其它文字。"
)


def extract_experiment(markdown: str, *, chat_fn=chat) -> dict:
    """从解析文本提取 experiment.json；解析失败/空回复时带上下文重问一次。

    chat_fn 可注入便于离线测试。
    """
    paper_text = markdown[:MAX_INPUT_CHARS]
    prompt = _PROMPT.format(paper_text=paper_text)
    reply, usage = chat_fn(
        [{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.1,
    )
    try:
        data = parse_json_reply(reply)
    except ValueError as exc:
        reply2, usage2 = chat_fn(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": (reply or "(空回复)")[:2000]},
                {"role": "user", "content": _REASK_INSTRUCTION},
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        usage = {
            "prompt": usage["prompt"] + usage2["prompt"],
            "completion": usage["completion"] + usage2["completion"],
            "model": usage2["model"],
        }
        try:
            data = parse_json_reply(reply2)
        except ValueError:
            raise ValueError(f"重问后仍无法解析 JSON（首次错误：{exc}）") from exc
    return normalize_experiment(data, paper_text=paper_text, usage=usage)
