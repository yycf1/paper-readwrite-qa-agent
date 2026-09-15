import React, { useEffect, useState } from "react";
import { api, pollJob, type ExperimentSchema, type Job, type PaperDetail } from "../api";

type Tab = "report" | "experiment" | "fulltext" | "abstract";

export default function PaperDetail({
  id,
  onClose,
  onChanged,
}: {
  id: string;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [tab, setTab] = useState<Tab>("report");
  const [err, setErr] = useState("");
  const [job, setJob] = useState<Job | null>(null);

  const load = () =>
    api.paper(id).then(setPaper).catch((e) => setErr(String(e)));

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const process = async () => {
    setErr("");
    setJob({ id: "", kind: "process_paper", status: "running", result: {}, error: "" });
    try {
      const j = await api.process(id);
      await pollJob(j.id, setJob);
      await load();
      onChanged();
    } catch (e) {
      setErr(String(e));
      setJob(null);
    }
  };

  if (err) return <div className="detail-pane"><div className="err">⚠ {err}</div></div>;
  if (!paper) return <div className="detail-pane"><div className="loading"><span className="spin" />加载中…</div></div>;

  const exp = paper.experiment;
  const tabs = ([
    { k: "report", label: "精读报告", show: !!paper.report?.trim() },
    { k: "experiment", label: "实验结构", show: !!exp },
    { k: "fulltext", label: "全文", show: !!paper.full_text },
    { k: "abstract", label: "摘要", show: !!paper.abstract },
  ] as { k: Tab; label: string; show: boolean }[]).filter((t) => t.show);
  const active = tabs.find((t) => t.k === tab) ? tab : tabs[0]?.k;
  const processing = job?.status === "running";
  const steps = ((job?.result?.steps as { stage: string; ok: boolean; detail: string }[]) ?? []);

  const doDelete = async () => {
    if (!window.confirm(`确定删除《${paper.title.slice(0, 40)}…》？\n将同时删除其原文、解析文本、知识与索引产物，不可恢复。`)) return;
    setErr("");
    try {
      await api.deletePaper(id);
      onChanged();
      onClose();
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <div className="detail-pane">
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
        <h2>{paper.title}</h2>
        <div style={{ display: "flex", gap: 8, whiteSpace: "nowrap", height: 36 }}>
          <button className="danger" onClick={doDelete}>删除</button>
          <button className="primary" style={{ height: "100%" }} onClick={process} disabled={processing}>
            {processing ? "处理中…" : "一键处理"}
          </button>
        </div>
      </div>
      <div className="meta-row">
        <span className={`tag status-${paper.status}`}>{paper.status}</span>
        <span>{paper.source}</span>
        {paper.tag && <span className="tag">分组：{paper.tag}</span>}
        {paper.year && <span>{paper.year} 年</span>}
        {paper.citations != null && <span>被引 {paper.citations}</span>}
        {paper.doi && <span>DOI: {paper.doi}</span>}
        {paper.authors.length > 0 && <span>{paper.authors.slice(0, 4).join(", ")}{paper.authors.length > 4 ? " 等" : ""}</span>}
        <span style={{ cursor: "pointer", color: "var(--accent-2)" }} onClick={onClose}>
          关闭 ✕
        </span>
      </div>

      {processing && (
        <div className="section-block">
          <h3>处理进度</h3>
          <div className="steps">
            {steps.length === 0 && <div className="hint"><span className="spin" />排队启动中…</div>}
            {steps.map((s, i) => (
              <div key={i} className="step">
                <span className={s.ok ? "ok" : "no"}>{s.ok ? "✓" : "✗"}</span>
                <span>{stageLabel(s.stage)} — {s.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {job?.status === "error" && <div className="err">⚠ 任务失败：{job.error}</div>}
      {paper.error && <div className="err" style={{ marginBottom: 10 }}>上次错误：{paper.error}</div>}

      {tabs.length === 0 ? (
        <div className="hint" style={{ marginTop: 30 }}>
          还没有产物。点击右上角「一键处理」：下载 → 解析 → 知识提取 → 索引，完成后这里会出现精读报告与实验结构。
        </div>
      ) : (
        <>
          <div className="chips" style={{ marginBottom: 10 }}>
            {tabs.map((t) => (
              <span key={t.k} className={`chip ${active === t.k ? "on" : ""}`} onClick={() => setTab(t.k)}>
                {t.label}
              </span>
            ))}
          </div>
          {active === "report" && <div className="section-block"><div className="body">{paper.report}</div></div>}
          {active === "abstract" && <div className="section-block"><div className="body">{paper.abstract}</div></div>}
          {active === "fulltext" && (
            <div className="section-block"><div className="body" style={{ maxHeight: 560 }}>{paper.full_text}</div></div>
          )}
          {active === "experiment" && exp && <ExperimentView exp={exp} />}
        </>
      )}
    </div>
  );
}

function stageLabel(s: string): string {
  const m: Record<string, string> = { download: "下载", parse: "解析", analyze: "知识提取", index: "向量索引" };
  return m[s] ?? s;
}

function ExperimentView({ exp }: { exp: ExperimentSchema }) {
  const conf = exp.confidence ?? {};
  return (
    <div className="section-block">
      <div className="kv" style={{ marginBottom: 14 }}>
        <div className="k">研究目标</div><div>{renderValue(exp.research_goal)}</div>
        <div className="k">方法</div><div>{renderValue(exp.method)}</div>
        <div className="k">数据集</div><div>{renderValue(exp.datasets)}</div>
        <div className="k">评价指标</div><div>{renderValue(exp.metrics)}</div>
        <div className="k">结论</div><div>{renderValue(exp.conclusions)}</div>
      </div>
      <div className="section-block">
        <h3>实验步骤</h3>
        <div className="body"><ol className="plain" style={{ margin: 0, paddingLeft: 20 }}>{toList(exp.experiment_steps).map((s, i) => <li key={i}>{renderValue(s, true)}</li>)}</ol></div>
      </div>
      {isRecord(exp.hyperparameters) && Object.keys(exp.hyperparameters).length > 0 && (
        <div className="section-block">
          <h3>超参数</h3>
          <div className="body"><div className="kv">{Object.entries(exp.hyperparameters).map(([k, v]) => (<><div className="k" key={k}>{k}</div><div key={k + "v"}>{renderValue(v)}</div></>))}</div></div>
        </div>
      )}
      <div className="section-block">
        <h3>主要结果</h3>
        <div className="body"><ul className="plain">{toList(exp.main_results).map((s, i) => <li key={i}>{renderValue(s, true)}</li>)}</ul></div>
      </div>
      <div className="hint">
        字段置信度：
        {Object.entries(conf).map(([k, v]) => `${k}=${v}`).join("，") || "未标注"}
      </div>
    </div>
  );
}

/** 递归渲染 schema 值：对象 → 有序键值；数组 → 列表；字符串原样。 */
function renderValue(v: unknown, inline = false): React.ReactNode {
  if (v == null || v === "") return "—";
  if (typeof v === "string" || typeof v === "number") return String(v);
  if (Array.isArray(v)) {
    if (v.length === 0) return "—";
    const items = v.map((x, i) => <li key={i}>{renderValue(x, true)}</li>);
    return inline ? <ul className="plain">{items}</ul> : <ul className="plain">{items}</ul>;
  }
  if (isRecord(v)) {
    const parts = Object.entries(v)
      .filter(([, x]) => x != null && x !== "" && !(Array.isArray(x) && x.length === 0))
      .map(([k, x]) => (
        <span key={k}>
          <span className="muted">{k}: </span>
          {renderValue(x, true)}
        </span>
      ));
    return parts.length ? <>{parts.map((p, i) => <React.Fragment key={i}>{p}{i < parts.length - 1 ? "；" : ""}</React.Fragment>)}</> : "—";
  }
  return String(v);
}

function toList(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [v];
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
