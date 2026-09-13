"""问答链：向量检索 top-k → 带引用回答，范围外明确拒绝。

设计对应 PLAN §9 M3：
- 出处标注 [编号]（论文标题 + 章节 + 页码），编号与检索块一一对应；
- 提供的片段不足以回答时，明确回答「论文库中的内容未提及」，禁止编造；
- 多轮 REPL 靠 history（近几轮问答）维持语境，检索始终按当前问题执行。
"""

from __future__ import annotations

from paper_agent.llm import chat
from paper_agent.rag.embedder import embed
from paper_agent.rag.vectorstore import Hit, VectorStore

MAX_HISTORY_TURNS = 3

_PROMPT = """你是论文阅读助手。只依据下方论文片段回答问题，并遵守：
1. 关键结论处用 [编号] 标注出处（编号对应片段序号）；
2. 若片段不足以回答问题，回答「论文库中的内容未提及」，不要用已有一知识编造；
3. 回答简洁，用中文。

论文片段：
{context}

问题：{question}"""


class Retriever:
    """向量检索入口：问题 → 向量 → top-k 块。"""

    def __init__(self, store: VectorStore, embed_fn=embed):
        self.store = store
        self.embed_fn = embed_fn

    def retrieve(self, query: str, *, k: int = 5, paper_id: str | None = None) -> list[Hit]:
        vector = self.embed_fn([query])[0]
        return self.store.query(vector, k=k, paper_id=paper_id)


def build_context(hits: list[Hit]) -> str:
    """检索命中 → 带编号的上下文块。"""
    blocks = []
    for i, h in enumerate(hits, start=1):
        origin = f"论文《{h.title}》· {h.section}"
        if h.page:
            origin += f" · 第{h.page}页"
        blocks.append(f"[{i}] {origin}\n{h.text}")
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    *,
    retriever: Retriever,
    history: list[tuple[str, str]] | None = None,
    k: int = 8,
    paper_id: str | None = None,
    chat_fn=chat,
    hits: list[Hit] | None = None,
) -> tuple[str, list[Hit]]:
    """回答一个问题，返回 (答案文本, 使用的检索命中)。

    hits 可外部传入（评测时检索与判定共用一次结果）；history 为近几轮
    (问题, 回答)，用于 REPL 里的指代消解。k 默认 8：块均约 600 token，
    8 块 ≈ 5k token 上下文，实测关键事实多排在 6~8 位，5 块容易漏。
    """
    if hits is None:
        hits = retriever.retrieve(question, k=k, paper_id=paper_id)
    if not hits:
        return "向量库还没有内容：请先运行 `pa index` 建立索引。", []
    messages: list[dict] = []
    for prev_q, prev_a in (history or [])[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": prev_q})
        messages.append({"role": "assistant", "content": prev_a})
    messages.append(
        {"role": "user", "content": _PROMPT.format(context=build_context(hits), question=question)}
    )
    answer, _usage = chat_fn(messages, max_tokens=2048, temperature=0.2)
    if not answer.strip():
        # GLM 混合推理模型：思考 tokens 可能吃光 max_tokens 导致空回复，
        # 禁用思考按同参数重问一次（正常路径零额外开销）
        answer, _usage = chat_fn(messages, max_tokens=2048, temperature=0.2, thinking=False)
    return answer.strip(), hits
