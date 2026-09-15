"""ChatStore（会话存储）测试：会话 CRUD、消息追加与历史读取。"""

from __future__ import annotations

from paper_agent.chatstore import ChatStore


def test_session_lifecycle(tmp_path):
    store = ChatStore(tmp_path / "lib.db")
    try:
        sid = store.create_session("联邦学习调研")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == sid
        assert sessions[0]["title"] == "联邦学习调研"
        assert sessions[0]["message_count"] == 0

        assert store.delete_session(sid) is True
        assert store.list_sessions() == []
        assert store.delete_session(sid) is False  # 幂等删除
    finally:
        store.close()


def test_messages_roundtrip_and_order(tmp_path):
    store = ChatStore(tmp_path / "lib.db")
    try:
        sid = store.create_session("会话A")
        store.append_message(sid, role="user", content="帮我找 GNN 论文")
        store.append_message(
            sid, role="assistant", content="已入库 3 篇", intent="search",
            understanding="检索（LLM）", notes=["arxiv 不可用"],
        )
        msgs = store.get_messages(sid)
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[0]["content"] == "帮我找 GNN 论文"
        assert msgs[1]["intent"] == "search"
        assert msgs[1]["notes"] == ["arxiv 不可用"]
        # 会话排在最前（updated_at 最新）
        assert store.list_sessions()[0]["id"] == sid
    finally:
        store.close()


def test_sessions_sorted_by_recency(tmp_path):
    store = ChatStore(tmp_path / "lib.db")
    try:
        old = store.create_session("旧会话")
        new = store.create_session("新会话")
        store.append_message(old, role="user", content="唤醒旧会话")  # old 变为最近活跃
        ids = [s["id"] for s in store.list_sessions()]
        assert ids == [old, new]
    finally:
        store.close()
