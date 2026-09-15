import { useCallback, useEffect, useState } from "react";
import { api, type Paper, type StatusSummary } from "../api";
import PaperDetail from "../components/PaperDetail";

const STATUS_ORDER = ["discovered", "downloaded", "parsed", "analyzed", "indexed", "download_failed"];

export default function Library({
  summary,
  detailId,
  onOpenDetail,
  onChanged,
}: {
  summary: StatusSummary | null;
  detailId: string | null;
  onOpenDetail: (id: string | null) => void;
  onChanged: () => void;
}) {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState("");
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchYear, setSearchYear] = useState("");
  const [notes, setNotes] = useState<string[]>([]);
  const [err, setErr] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setErr("");
    try {
      const data = await api.papers({ status: status || undefined, q: q || undefined, limit: 200 });
      setPapers(data.papers);
      setTotal(data.total);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }, [status, q]);

  useEffect(() => {
    load();
  }, [load]);

  const doSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setErr("");
    setNotes([]);
    try {
      const year = searchYear ? Number(searchYear) : null;
      const out = await api.search(searchQuery.trim(), year);
      setNotes(out.notes);
      onChanged();
      load();
      if (out.new_papers.length === 0 && out.total_hits === 0) {
        setNotes((n) => [...n, "没有检索到结果：试试更换关键词、放宽年份，或减少限定条件。"]);
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="library">
      <div className="list-pane">
        <div className="toolbar">
          <input
            type="text"
            placeholder="检索新论文（支持中文，自动优化为英文检索词）…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && doSearch()}
          />
          <input
            type="number"
            placeholder="年份 ≥"
            style={{ width: 90 }}
            value={searchYear}
            onChange={(e) => setSearchYear(e.target.value)}
          />
          <button className="primary" disabled={searching || !searchQuery.trim()} onClick={doSearch}>
            {searching ? "检索中…" : "检索入库"}
          </button>
        </div>
        <div className="toolbar" style={{ borderTop: "none" }}>
          <input
            type="text"
            placeholder="在库内筛选：标题 / 作者 / DOI…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            style={{ maxWidth: 320 }}
          />
          <div className="chips">
            <span className={`chip ${status === "" ? "on" : ""}`} onClick={() => setStatus("")}>
              全部 {summary?.total ?? ""}
            </span>
            {STATUS_ORDER.filter((s) => summary?.counts[s]).map((s) => (
              <span key={s} className={`chip ${status === s ? "on" : ""}`} onClick={() => setStatus(s)}>
                {s === "download_failed" ? "失败" : label(s)} {summary?.counts[s]}
              </span>
            ))}
          </div>
        </div>
        {(notes.length > 0 || err) && (
          <div style={{ padding: "6px 16px" }}>
            {err && <div className="err">⚠ {err}</div>}
            {notes.map((n, i) => (
              <div key={i} className="hint">
                · {n}
              </div>
            ))}
          </div>
        )}
        <div className="table-wrap">
          {loading ? (
            <div className="loading">
              <span className="spin" />
              加载中…
            </div>
          ) : papers.length === 0 ? (
            <div className="loading">没有符合条件的论文。用上方检索框找一批新论文进来。</div>
          ) : (
            <table className="papers">
              <thead>
                <tr>
                  <th style={{ width: 90 }}>状态</th>
                  <th style={{ width: 56 }}>年份</th>
                  <th style={{ width: 56 }}>引用</th>
                  <th style={{ width: 64 }}>全文</th>
                  <th>标题（{total} 篇）</th>
                </tr>
              </thead>
              <tbody>
                {papers.map((p) => (
                  <tr
                    key={p.source_id}
                    className={`row ${detailId === p.source_id ? "sel" : ""}`}
                    onClick={() => onOpenDetail(p.source_id)}
                  >
                    <td>
                      <span className={`tag status-${p.status}`}>
                        {p.status === "download_failed" ? "失败" : label(p.status)}
                      </span>
                    </td>
                    <td>{p.year ?? "-"}</td>
                    <td>{p.citations ?? "-"}</td>
                    <td>{p.has_pdf ? "✓" : "摘要"}</td>
                    <td className="title-cell">
                      <div className="paper-title">{p.title}</div>
                      {p.error && <div className="paper-err" title={p.error}>⚠ {p.error}</div>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
      {detailId ? (
        <PaperDetail
          key={detailId}
          id={detailId}
          onClose={() => onOpenDetail(null)}
          onChanged={() => {
            onChanged();
            load();
          }}
        />
      ) : (
        <div className="detail-pane" style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="hint">← 点击左侧论文查看详情、精读报告与实验结构</div>
        </div>
      )}
    </div>
  );
}

function label(s: string): string {
  const m: Record<string, string> = {
    discovered: "已发现",
    downloaded: "已下载",
    parsed: "已解析",
    analyzed: "已分析",
    indexed: "已入库",
  };
  return m[s] ?? s;
}
