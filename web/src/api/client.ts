import { currentLanguage } from "../i18n";
import type {
  AssessmentResult,
  AuditEntry,
  BenchmarkResearch,
  BenchmarkSummary,
  ChatResponse,
  CopilotCatalog,
  CopilotChatResponse,
  CopilotExplanation,
  CopilotStatus,
  KnowledgeBase,
  KbRule,
  LlmDebug,
  PatientStatus,
  RecordsTable,
  SystemHealth,
  SystemInfo,
} from "./types";

export class ApiError extends Error {
  status: number;
  payload: unknown;
  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

type RequestOptions = RequestInit & { query?: Record<string, string | number | undefined | null> };

function withQuery(path: string, query?: RequestOptions["query"]): string {
  const params = new URLSearchParams();
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== undefined && value !== null && value !== "") {
        params.set(key, String(value));
      }
    }
  }
  const suffix = params.toString();
  if (!suffix) return path;
  return `${path}${path.includes("?") ? "&" : "?"}${suffix}`;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { query, headers, ...rest } = options;
  const language = currentLanguage();
  const url = withQuery(path, { ...query, lang: language });
  const response = await fetch(url, {
    ...rest,
    headers: {
      Accept: "application/json",
      "Accept-Language": language,
      ...(headers || {}),
    },
  });
  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const record = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
    const message =
      (typeof record.error === "string" && record.error) ||
      (typeof record.detail === "string" && record.detail) ||
      (typeof payload === "string" && payload) ||
      `Request failed with status ${response.status}`;
    throw new ApiError(message, response.status, payload);
  }
  return payload as T;
}

function json(body: unknown): Pick<RequestOptions, "method" | "headers" | "body"> {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export const api = {
  core: {
    health: () => request<{ status: string; version?: string }>("/api/core/health"),
    patients: () => request<{ patients: string[] }>("/api/core/chat/patients"),
    patientStatus: (patientId: string) =>
      request<PatientStatus>(`/api/core/chat/status`, { query: { patient_id: patientId } }),
    records: (patientId: string) =>
      request<RecordsTable>(`/api/core/records/${encodeURIComponent(patientId)}`),
    assess: (body: {
      patient_id: string;
      mode?: string;
      records: Array<{ timestamp: string; metric: string; value: number }>;
    }) => request<AssessmentResult>("/api/core/assess", json({ ...body, lang: currentLanguage() })),
    chat: (body: { patient_id: string; message: string; date?: string }) =>
      request<ChatResponse>("/api/core/chat", json({ ...body, lang: currentLanguage() })),
    chatFile: (patientId: string, date: string, file: File) => {
      const form = new FormData();
      form.append("patient_id", patientId);
      if (date) form.append("date", date);
      form.append("lang", currentLanguage());
      form.append("file", file);
      return request<ChatResponse>("/api/core/chat_file", { method: "POST", body: form });
    },
    reset: (patientId: string) =>
      request<{ ok: boolean; message?: string }>("/api/core/chat/reset", {
        method: "POST",
        query: { patient_id: patientId },
      }),
    upsertRecord: (
      patientId: string,
      body: { timestamp: string; metric: string; value: number | null; unit?: string },
    ) =>
      request<{ ok: boolean }>(`/api/core/records/${encodeURIComponent(patientId)}`, {
        ...json(body),
        method: "PUT",
      }),
    deleteRecord: (patientId: string, timestamp: string, metric = "") =>
      request<{ ok: boolean }>(`/api/core/records/${encodeURIComponent(patientId)}`, {
        method: "DELETE",
        query: { timestamp, metric },
      }),
    kb: () => request<KnowledgeBase>("/api/core/kb"),
    audit: (limit = 200) => request<{ entries: AuditEntry[] }>("/api/core/kb/audit", { query: { limit } }),
    addRule: (rule: Record<string, unknown>) => request<{ ok: boolean; rule: KbRule }>("/api/core/kb/rules", json(rule)),
    updateRule: (ruleId: string, patch: Record<string, unknown>) =>
      request<{ ok: boolean; rule: KbRule }>(`/api/core/kb/rules/${encodeURIComponent(ruleId)}`, {
        ...json(patch),
        method: "PUT",
      }),
    deleteRule: (ruleId: string, actor: string) =>
      request<{ ok: boolean }>(`/api/core/kb/rules/${encodeURIComponent(ruleId)}`, {
        method: "DELETE",
        query: { actor },
      }),
    transitionRule: (ruleId: string, body: { to: string; actor: string; note?: string }) =>
      request<{ ok: boolean; rule: KbRule }>(
        `/api/core/kb/rules/${encodeURIComponent(ruleId)}/transition`,
        json(body),
      ),
    addSystem: (body: { system_key: string; label: string }) =>
      request<{ ok: boolean }>("/api/core/kb/systems", json(body)),
    addMetric: (body: { metric: string; info: Record<string, unknown> }) =>
      request<{ ok: boolean }>("/api/core/kb/metrics", json(body)),
    benchmark: () => request<BenchmarkSummary>("/api/core/benchmark"),
    benchmarkResearch: () => request<BenchmarkResearch>("/api/core/benchmark/research"),
    benchmarkExplain: (body: { metrics: Record<string, number>; models?: string[] }) =>
      request<{ results: Array<Record<string, unknown>>; clinical?: Record<string, unknown> }>(
        "/api/core/benchmark/explain",
        json({ ...body, lang: currentLanguage() }),
      ),
    evidenceMl: (score?: number) =>
      request<{ html_md: string }>("/api/core/evidence/ml", { query: { score } }),
  },
  copilot: {
    status: () => request<CopilotStatus>("/api/copilot/status"),
    catalog: () => request<CopilotCatalog>("/api/copilot/catalog"),
    patient: (patientId: string) =>
      request<Record<string, unknown>>(`/api/copilot/patient/${encodeURIComponent(patientId)}`),
    guide: (body: Record<string, unknown>) =>
      request<Record<string, unknown>>("/api/copilot/guide", json({ ...body, lang: currentLanguage() })),
    metric: (body: { metric: string; audience?: string }) =>
      request<Record<string, unknown>>("/api/copilot/metric", json({ ...body, lang: currentLanguage() })),
    assess: (body: Record<string, unknown>) =>
      request<{
        ok: boolean;
        assessment?: AssessmentResult;
        explanation?: CopilotExplanation;
        warnings?: string[];
        error?: string;
        llm_debug?: LlmDebug | null;
      }>("/api/copilot/assess", json({ ...body, lang: currentLanguage() })),
    explain: (body: { assessment: AssessmentResult; audience?: string; use_llm?: boolean; debug_llm?: boolean }) =>
      request<CopilotExplanation>("/api/copilot/explain", json({ ...body, lang: currentLanguage() })),
    chat: (body: Record<string, unknown>) =>
      request<CopilotChatResponse>("/api/copilot/chat", json({ ...body, lang: currentLanguage() })),
    disclaimer: () => request<{ disclaimer: string }>("/api/copilot/disclaimer"),
    warmup: () => request<{
      warmed_up: boolean;
      llm_available: boolean;
      provider: string;
      model: string;
    }>("/api/copilot/warmup"),
  },
  system: {
    health: () => request<SystemHealth>("/api/system/health"),
    info: () => request<SystemInfo>("/api/system/info"),
    model: () => request<Record<string, unknown>>("/api/system/model"),
  },
};

export type ApiClient = typeof api;
