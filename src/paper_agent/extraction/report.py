"""精读报告生成：解析后的论文文本 → report.md（人读优先，面向「读懂」而非「评审」）。"""

from __future__ import annotations

from paper_agent.llm import chat

MAX_INPUT_CHARS = 36_000

_PROMPT = """你是论文精读助手。依据我提供的论文文本写一份中文精读报告，用 Markdown 输出。

结构（严格按以下小节，标题用 `## `）：
1. `## 一句话总结`——不超过 60 字；
2. `## 研究问题与动机`——要解决什么问题、为什么现有方法不够；
3. `## 方法脉络`——按「输入→核心思路→关键设计→输出」讲清楚，面向读懂而非评审；
4. `## 实验设计`——数据集、基线、指标、关键设置；
5. `## 主要结论与局限`——作者的主张与你自己观察到的局限；
6. `## 可复现性备注`——代码/数据是否公开（原文出现的链接原样保留）、细节充分度、复现的主要风险点。

要求：
- 只依据原文，不编造；原文没写的直接说「文中未说明」；
- 保留原文中的具体数字（指标值、数据规模等）；
- 总长度 600～1000 字，克制、信息密度优先。

论文文本（可能被截断）：
---
{paper_text}
---"""


def generate_report(markdown: str, *, chat_fn=chat) -> tuple[str, dict]:
    """生成精读报告。返回 (markdown, token用量)。chat_fn 可注入便于离线测试。"""
    reply, usage = chat_fn(
        [{"role": "user", "content": _PROMPT.format(paper_text=markdown[:MAX_INPUT_CHARS])}],
        max_tokens=2048,
        temperature=0.3,
    )
    return reply.strip() + "\n", usage
