import { useEffect, useState } from "react";
import { api, type StatusSummary } from "./api";
import Library from "./pages/Library";
import Assistant from "./pages/Assistant";

type Tab = "library" | "assistant";

/** URL 深链：?paper=<id> 直开详情，?tab=chat 直开助手（可分享、可回退）。 */
function readLocation(): { tab: Tab; paper: string | null } {
  const p = new URLSearchParams(window.location.search);
  return {
    tab: p.get("tab") === "chat" ? "assistant" : "library",
    paper: p.get("paper"),
  };
}

export default function App() {
  const [{ tab, paper }, setLocation] = useState(readLocation);
  const [summary, setSummary] = useState<StatusSummary | null>(null);

  const refresh = () => api.status().then(setSummary).catch(() => {});
  useEffect(() => {
    refresh();
  }, []);

  const openDetail = (id: string | null) => {
    const u = new URL(window.location.href);
    if (id) u.searchParams.set("paper", id);
    else u.searchParams.delete("paper");
    window.history.replaceState(null, "", u);
    setLocation({ tab: "library", paper: id });
  };
  const switchTab = (t: Tab) => {
    const u = new URL(window.location.href);
    if (t === "assistant") u.searchParams.set("tab", "chat");
    else u.searchParams.delete("tab");
    window.history.replaceState(null, "", u);
    setLocation({ tab: t, paper });
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">📄</span> paper-agent
          <span className="ver">{summary ? `v${summary.version}` : ""}</span>
        </div>
        <nav className="tabs">
          <button className={tab === "library" ? "on" : ""} onClick={() => switchTab("library")}>
            文献库
          </button>
          <button className={tab === "assistant" ? "on" : ""} onClick={() => switchTab("assistant")}>
            助手
          </button>
        </nav>
        <div className="summary">
          {summary && (
            <>
              <span>共 {summary.total} 篇</span>
              <span className="dot">·</span>
              <span>{summary.vector_chunks} 知识块</span>
            </>
          )}
        </div>
      </header>
      <main>
        {tab === "library" ? (
          <Library summary={summary} onOpenDetail={openDetail} detailId={paper} onChanged={refresh} />
        ) : (
          <Assistant onOpenDetail={openDetail} onLibChanged={refresh} />
        )}
      </main>
    </div>
  );
}
