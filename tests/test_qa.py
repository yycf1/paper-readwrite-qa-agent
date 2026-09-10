"""问答链（rag/qa.py）单测：注入 fake store / embed / chat，全离线。"""

from __future__ import annotations

from paper_agent.rag.qa import Retriever, answer_question, build_context
from paper_agent.rag.vectorstore import Hit


class FakeStore:
    def __init__(self, hits: list[Hit]):
        self.hits = hits
        self.calls: list[dict] = []

    def query(self, embedding, *, k=5, paper_id=None):
        self.calls.append({"k": k, "paper_id": paper_id})
        return self.hits[:k]


class FakeChat:
    def __init__(self, reply: str):
        self.reply = reply
        self.messages: list[list[dict]] = []

    def __call__(self, messages, *, max_tokens=0, temperature=0.0):
        self.messages.append(messages)
        return self.reply, {"prompt": 1, "completion": 1, "model": "fake"}


def _hit(i: int) -> Hit:
    return Hit(
        paper_id="x:1",
        title="测试论文",
        section="方法",
        page=i,
        text=f"第{i}段内容",
        distance=0.1 * i,
    )


def test_build_context_numbers_and_origin() -> None:
    ctx = build_context([_hit(1), _hit(2)])
    assert "[1] 论文《测试论文》· 方法 · 第1页" in ctx
    assert "[2] 论文《测试论文》· 方法 · 第2页" in ctx
    assert "第1段内容" in ctx


def test_build_context_omits_page_zero() -> None:
    h = _hit(1)
    h.page = 0
    assert "第0页" not in build_context([h])


def test_retriever_passes_k_and_paper_id() -> None:
    store = FakeStore([_hit(1)])
    retriever = Retriever(store, embed_fn=lambda texts: [[0.0, 1.0] for _ in texts])
    hits = retriever.retrieve("什么是X?", k=3, paper_id="x:1")
    assert len(hits) == 1
    assert store.calls[0] == {"k": 3, "paper_id": "x:1"}


def test_answer_question_includes_context_and_history() -> None:
    store = FakeStore([_hit(1)])
    retriever = Retriever(store, embed_fn=lambda texts: [[0.0] for _ in texts])
    chat = FakeChat("答案是 [1]")
    history = [(f"旧问题{i}", f"旧回答{i}") for i in range(5)]
    answer, hits = answer_question(
        "新问题", retriever=retriever, history=history, chat_fn=chat
    )
    assert answer == "答案是 [1]"
    assert len(hits) == 1
    sent = chat.messages[0]
    # 近 3 轮历史（6 条消息）+ 本轮
    assert len(sent) == 2 * 3 + 1
    assert sent[-1]["role"] == "user"
    assert "新问题" in sent[-1]["content"]
    assert "[1] 论文《测试论文》" in sent[-1]["content"]
    # 超出窗口的旧问答不进入
    assert "旧问题0" not in sent[-1]["content"]


def test_answer_question_uses_precomputed_hits() -> None:
    store = FakeStore([_hit(1)])
    retriever = Retriever(store, embed_fn=lambda texts: (_ for _ in ()).throw(AssertionError("不应再检索")))
    chat = FakeChat("ok")
    pre = [_hit(9)]
    answer_question("问题", retriever=retriever, chat_fn=chat, hits=pre)
    assert store.calls == []  # 未触发检索


def test_answer_question_empty_hits_hint() -> None:
    retriever = Retriever(FakeStore([]), embed_fn=lambda texts: [[0.0] for _ in texts])
    answer, hits = answer_question("问题", retriever=retriever, chat_fn=FakeChat("x"))
    assert hits == []
    assert "pa index" in answer
