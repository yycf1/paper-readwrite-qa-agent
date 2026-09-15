/** 后端 /api 的类型定义与请求封装。 */

export interface Paper {
  source_id: string;
  source: string;
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  citations: number | null;
  has_pdf: boolean;
  landing_page: string | null;
  status: string;
  error: string | null;
  added_at: string;
  abstract?: string;
}

export interface PaperDetail extends Paper {
  abstract: string;
  experiment: ExperimentSchema | null;
  report: string | null;
  full_text: string | null;
}

/** extraction/normalize_experiment 的 schema（M2），字段值宽松处理。 */
export interface ExperimentSchema {
  research_goal: string;
  method: string;
  datasets: string[];
  hyperparameters: Record<string, string>;
  experiment_steps: string[];
  metrics: string[];
  main_results: string[];
  conclusions: string;
  confidence: Record<string, string>;
  missing_fields?: string[];
  token_usage?: { prompt: number; completion: number };
  [k: string]: unknown;
}

export interface StatusSummary {
  total: number;
  counts: Record<string, number>;
  vector_chunks: number;
  version: string;
}

export interface SearchReply {
  new_papers: Paper[];
  total_hits: number;
  notes: string[];
  source_hits: Record<string, number>;
  topics: string[];
  optimized: boolean;
}

export interface AskReply {
  answer: string;
  sources: { paper_id: string; title: string; section: string; page: number }[];
}

export interface ChatReply {
  intent: string;
  reply: string;
  understanding: string;
  notes: string[];
  new_papers: Paper[];
}

export interface Job {
  id: string;
  kind: string;
  status: "running" | "done" | "error";
  result: Record<string, unknown>;
  error: string;
}

const BASE = ""; // 开发模式经 Vite 代理转发到 pa serve；生产由 FastAPI 同端口托管

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* 非 JSON 响应 */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => req<StatusSummary>("/status"),
  papers: (params: { status?: string; q?: string; limit?: number }) => {
    const usp = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && usp.set(k, String(v)));
    return req<{ total: number; papers: Paper[] }>(`/papers?${usp}`);
  },
  paper: (id: string) => req<PaperDetail>(`/papers/${encodeURIComponent(id)}`),
  search: (query: string, year_from?: number | null, max_results = 10) =>
    req<SearchReply>("/search", {
      method: "POST",
      body: JSON.stringify({ query, year_from: year_from ?? undefined, max_results }),
    }),
  ask: (question: string, paper_id?: string) =>
    req<AskReply>("/ask", { method: "POST", body: JSON.stringify({ question, paper_id }) }),
  chat: (message: string) =>
    req<ChatReply>("/chat", { method: "POST", body: JSON.stringify({ message }) }),
  process: (id: string) =>
    req<Job>(`/papers/${encodeURIComponent(id)}/process`, { method: "POST" }),
  run: (query: string) =>
    req<Job>("/jobs/run", { method: "POST", body: JSON.stringify({ query }) }),
  job: (id: string) => req<Job>(`/jobs/${id}`),
};

/** 轮询任务直至 done/error（或超时）。 */
export async function pollJob(id: string, onTick?: (j: Job) => void, timeoutMs = 600_000) {
  const start = Date.now();
  for (;;) {
    const job = await api.job(id);
    onTick?.(job);
    if (job.status !== "running") return job;
    if (Date.now() - start > timeoutMs) throw new Error("任务超时");
    await new Promise((r) => setTimeout(r, 1500));
  }
}
