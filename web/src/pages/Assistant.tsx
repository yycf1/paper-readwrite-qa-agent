import { useCallback, useEffect, useRef, useState } from "react";
import { api, type ChatSession, type ChatReply } from "../api";

interface Turn {
  role: "user" | "bot";
  text: string;
  understanding?: string;
  notes?: string[];
  newPapers?: { source_id: string; title: string }[];
}

const WELCOME: Turn[] = [
  {
    role: "bot",
    text:
      "你好，我是 paper-agent 助手。可以：\n① 检索新论文——「帮我找 2023 年以后的 transformer 综述」\n② 问答库内文献——「图像分割那篇综述的主要结论？」（带出处）\n③ 查看库状态——「库里有多少论文」\n④ 推荐方向——「不知道该看什么论文」\n\n对话会自动保存，左侧可新建或切换会话。",
  },
];

function turnsFromHistory(messages: { role: string; content: string; understanding: string | null; notes: string[] }[]): Turn[] {
  return messages.map((m) => ({
    role: m.role === "user" ? "user" : "bot",
    text: m.content,
    understanding: m.understanding ?? undefined,
    notes: m.notes ?? undefined,
  }));
}

export default function Assistant({
  onOpenDetail,
  onLibChanged,
}: {
  onOpenDetail: (id: string) => void;
  onLibChanged: () => void;
}) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>(WELCOME);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const refreshSessions = useCallback(
    () => api.sessions().then((d) => setSessions(d.sessions)).catch(() => {}),
    [],
  );

  const openSession = useCallback(async (sid: string | null) => {
    setActiveId(sid);
    const u = new URL(window.location.href);
    if (sid) u.searchParams.set("s", sid);
    else u.searchParams.delete("s");
    window.history.replaceState(null, "", u);
    if (sid === null) {
      setTurns(WELCOME);
      return;
    }
    try {
      const data = await api.sessionMessages(sid);
      const t = turnsFromHistory(data.messages);
      setTurns(t.length > 0 ? t : WELCOME);
    } catch {
      setTurns(WELCOME);
    }
  }, []);

  useEffect(() => {
    refreshSessions();
    const fromUrl = new URLSearchParams(window.location.search).get("s");
    if (fromUrl) openSession(fromUrl);
  }, [refreshSessions, openSession]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [turns, busy]);

  const newChat = () => openSession(null);

  const removeSession = async (sid: string) => {
    try {
      await api.deleteSession(sid);
    } catch {
      /* 列表刷新时暴露错误 */
    }
    if (sid === activeId) await openSession(null);
    refreshSessions();
  };

  const send = async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || busy) return;
    setInput("");
    setTurns((t) => [...t, { role: "user", text: msg }]);
    setBusy(true);
    try {
      const reply: ChatReply = await api.chat(msg, activeId);
      setTurns((t) => [
        ...t,
        {
          role: "bot",
          text: reply.reply,
          understanding: reply.understanding,
          notes: reply.notes,
          newPapers: reply.new_papers,
        },
      ]);
      if (reply.session_id !== activeId) {
        await openSession(reply.session_id);
        refreshSessions();
      } else {
        refreshSessions();
      }
      if (reply.intent === "search" && reply.new_papers.length > 0) onLibChanged();
    } catch (e) {
      setTurns((t) => [...t, { role: "bot", text: `请求失败：${e}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="assistant">
      <aside className="session-pane">
        <button className="primary new-chat" onClick={newChat} disabled={busy}>
          ＋ 新对话
        </button>
        <div className="session-list">
          {sessions.length === 0 && <div className="hint" style={{ padding: "8px 12px" }}>还没有历史会话</div>}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeId ? "on" : ""}`}
              onClick={() => openSession(s.id)}
            >
              <div className="s-title">{s.title}</div>
              <div className="s-meta">
                {s.message_count} 条 · {s.updated_at.slice(5, 16).replace("T", " ")}
              </div>
              <span
                className="s-del"
                title="删除会话"
                onClick={(e) => {
                  e.stopPropagation();
                  removeSession(s.id);
                }}
              >
                ✕
              </span>
            </div>
          ))}
        </div>
      </aside>
      <div className="chat">
        <div className="chat-log" ref={logRef}>
          {turns.map((t, i) => (
            <div key={i} className={`msg ${t.role}`}>
              {t.understanding && <div className="under">（{t.understanding}）</div>}
              {t.text}
              {t.notes && t.notes.length > 0 && t.notes.map((n, j) => <div key={j} className="note">⚠ {n}</div>)}
              {t.newPapers && t.newPapers.length > 0 && (
                <div className="papers-mini">
                  {t.newPapers.map((p) => (
                    <div key={p.source_id} className="paper-mini" onClick={() => onOpenDetail(p.source_id)}>
                      📄 {p.title}
                    </div>
                  ))}
                  <div className="hint">点击可在「文献库」页打开详情</div>
                </div>
              )}
            </div>
          ))}
          {busy && (
            <div className="msg bot">
              <span className="spin" />
              思考中…（检索类请求通常需要十几秒到一分钟）
            </div>
          )}
        </div>
        <div className="quick">
          {["库里现在有多少论文？", "不知道该看什么论文", "帮我找 2024 年以后的 LLM agent 综述"].map((s) => (
            <span key={s} className="chip" onClick={() => send(s)}>
              {s}
            </span>
          ))}
        </div>
        <div className="chat-input">
          <input
            type="text"
            placeholder="用自然语言说需求…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            autoFocus
          />
          <button className="primary" onClick={() => send()} disabled={busy || !input.trim()}>
            发送
          </button>
        </div>
      </div>
    </div>
  );
}
