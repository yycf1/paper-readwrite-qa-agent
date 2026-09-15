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
  abstract_only?: boolean;
  status: string;
  error: string | null;
  added_at: string;
  tag: string;
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
  candidates: (Paper & { abstract: string; duplicate_of: string | null })[];
  total_hits: number;
  notes: string[];
  source_hits: Record<string, number>;
  topics: string[];
  optimized: boolean;
}

export interface IngestReply {
  ingested: number;
  duplicates: number;
  papers: Paper[];
}

export interface TagsReply {
  tags: Record<string, number>;
  sources: Record<string, number>;
}

export interface AskReply {
  answer: string;
  sources: { paper_id: string; title: string; section: string; page: number }[];
}

export interface ChatReply {
  session_id: string;
  intent: string;
  reply: string;
  understanding: string;
  notes: string[];
  new_papers: Paper[];
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  intent: string | null;
  understanding: string | null;
  notes: string[];
  created_at: string;
}

export interface Job {
  id: string;
  kind: string;
  status: "running" | "done" | "error";
  result: Record<string, unknown>;
  error: string;
}

const BASE = ""; // 开发模式经 Vite 代理转发到 pa serve；生产由 FastAPI 同端口托管

async function req<T>(path: string, init?: RequestInit, timeoutMs = 120_000): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    const headers: Record<string, string> = {};
    if (!(init?.body instanceof FormData)) headers["Content-Type"] = "application/json";
    res = await fetch(`${BASE}/api${path}`, { headers, ...init, signal: ctrl.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("请求超时（LLM 平台响应慢或数据源限流），请稍后重试");
    }
    throw new Error("网络错误：服务是否已启动（pa serve）？");
  } finally {
    clearTimeout(timer);
  }
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
  papers: (params: { status?: string; tag?: string; q?: string; limit?: number }) => {
    const usp = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v && usp.set(k, String(v)));
    return req<{ total: number; papers: Paper[] }>(`/papers?${usp}`);
  },
  paper: (id: string) => req<PaperDetail>(`/papers/${encodeURIComponent(id)}`),
  deletePaper: (id: string) =>
    req<{ ok: boolean }>(`/papers/${encodeURIComponent(id)}`, { method: "DELETE" }),
  tags: () => req<TagsReply>("/tags"),
  search: (
    query: string,
    year_from?: number | null,
    sources: string[] = ["openalex", "europepmc", "arxiv"],
    max_results = 10,
  ) =>
    req<SearchReply>("/search", {
      method: "POST",
      body: JSON.stringify({
        query,
        year_from: year_from ?? undefined,
        sources: sources.join(","),
        max_results,
      }),
    }),
  ingest: (
    papers: { source: string; source_id: string; title: string; abstract?: string }[],
    tag: string,
  ) =>
    req<IngestReply>("/papers/ingest", {
      method: "POST",
      body: JSON.stringify({ papers, tag }),
    }),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<{ is_new: boolean; paper: Paper }>("/papers/upload", { method: "POST", body: fd }, 300_000);
  },
  ask: (question: string, paper_id?: string) =>
    req<AskReply>("/ask", { method: "POST", body: JSON.stringify({ question, paper_id }) }),
  chat: (message: string, session_id?: string | null) =>
    req<ChatReply>("/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: session_id ?? undefined }),
    }),
  sessions: () => req<{ sessions: ChatSession[] }>("/sessions"),
  createSession: (title = "新对话") =>
    req<ChatSession>("/sessions", { method: "POST", body: JSON.stringify({ title }) }),
  sessionMessages: (sid: string) =>
    req<{ session_id: string; messages: ChatMessage[] }>(`/sessions/${sid}/messages`),
  deleteSession: (sid: string) =>
    req<{ ok: boolean }>(`/sessions/${sid}`, { method: "DELETE" }),
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
