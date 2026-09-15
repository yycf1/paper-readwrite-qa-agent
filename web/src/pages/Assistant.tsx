import { useEffect, useRef, useState } from "react";
import { api, type ChatReply } from "../api";

interface Turn {
  role: "user" | "bot";
  text: string;
  understanding?: string;
  notes?: string[];
  newPapers?: { source_id: string; title: string }[];
  sources?: { paper_id: string; title: string; section: string; page: number }[];
}

const WELCOME: Turn[] = [
  {
    role: "bot",
    text:
      "你好，我是 paper-agent 助手。可以：\n① 检索新论文——「帮我找 2023 年以后的 transformer 综述」\n② 问答库内文献——「图像分割那篇综述的主要结论？」（带出处）\n③ 查看库状态——「库里有多少论文」\n④ 推荐方向——「不知道该看什么论文」",
  },
];

export default function Assistant({
  onOpenDetail,
  onLibChanged,
}: {
  onOpenDetail: (id: string) => void;
  onLibChanged: () => void;
}) {
  const [turns, setTurns] = useState<Turn[]>(WELCOME);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [turns, busy]);

  const send = async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg || busy) return;
    setInput("");
    setTurns((t) => [...t, { role: "user", text: msg }]);
    setBusy(true);
    try {
      const reply: ChatReply = await api.chat(msg);
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
      if (reply.intent === "search" && reply.new_papers.length > 0) onLibChanged();
      if (reply.intent === "exit") setTurns((t) => [...t, ...WELCOME]);
    } catch (e) {
      setTurns((t) => [...t, { role: "bot", text: `请求失败：${e}` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat">
      <div className="chat-log" ref={logRef}>
        {turns.map((t, i) => (
          <div key={i} className={`msg ${t.role}`}>
            {t.understanding && <div className="under">（{t.understanding}）</div>}
            {t.text}
            {t.sources && t.sources.length > 0 && (
              <div className="sources">
                出处：
                {t.sources.map((s, j) => (
                  <span key={j} className="src" onClick={() => onOpenDetail(s.paper_id)}>
                    [{j + 1}] 《{s.title}》· {s.section}
                    {s.page ? ` · 第${s.page}页` : ""}
                  </span>
                ))}
              </div>
            )}
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
            思考中…
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
          placeholder="用自然语言说需求（q 退出只对命令行有效，这里直接关页面就好）…"
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
  );
}
