import {
  Activity,
  Brain,
  Bug,
  Check,
  CheckCircle2,
  CircleHelp,
  ArrowUp,
  ClipboardCheck,
  Copy,
  Paperclip,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  WandSparkles,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../api/client";
import type { AssessmentResult, CopilotExplanation, CopilotStatus, LlmDebug, RuleReference } from "../api/types";
import { Markdown } from "../components/Markdown";
import { PatientBar } from "../components/PatientBar";
import { usePatient } from "../components/PatientContext";
import { useToast } from "../components/Toast";
import { Alert, LoadingBlock, PageHeader, RiskBadge } from "../components/Ui";
import i18n, { levelLabel, metricName, metricUnit, specialtyLabel, systemLabel } from "../i18n";

type ChatChannel = "consult" | "journal";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  channel: ChatChannel;
  method?: string;
  provider?: string;
  model?: string;
  stage?: string;
  llmLatency?: number;
  llmReason?: string;
  llmValidation?: string[];
}

function todayInput(): string {
  const now = new Date();
  const offset = now.getTimezoneOffset();
  return new Date(now.getTime() - offset * 60_000).toISOString().slice(0, 10);
}

interface GuidePayload {
  next_question?: string;
  expected_metric?: string;
  metric_name_vi?: string;
  validation_hint?: string;
  unit?: string;
  normal_range?: string;
  progress?: string;
  done?: boolean;
  warnings?: string[];
  disclaimer?: string;
  [key: string]: unknown;
}

function parseMetrics(values: Record<string, string>): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(values)) {
    const trimmed = value.trim();
    if (!trimmed) continue;
    const number = Number(trimmed);
    if (Number.isFinite(number)) result[key] = number;
  }
  return result;
}

function CompactAssessment({ assessment }: { assessment: AssessmentResult }) {
  const { t } = useTranslation();
  const systems = assessment.affected_systems || [];
  const recommendations = assessment.recommendations || [];
  return (
    <div className="card">
      <div className="card__header">
        <div>
          <h2 className="card__title">{t("assessment.resultTitle")}</h2>
          <p className="card__subtitle">{t("risk.scoreNote")}</p>
        </div>
        <RiskBadge level={assessment.risk_level} />
      </div>
      <div className="stat-grid">
        <div className="stat">
          <div className="stat__label">{t("risk.score")}</div>
          <div className="stat__value">{Number(assessment.risk_score ?? 0).toFixed(3)}</div>
          <div className="stat__hint">{t("risk.level")}: {levelLabel(assessment.risk_level)}</div>
        </div>
        <div className="stat">
          <div className="stat__label">{t("risk.affected")}</div>
          <div className="stat__value">{systems.length}</div>
          <div className="stat__hint">{systems.slice(0, 2).map(systemLabel).join(", ") || t("common.none")}</div>
        </div>
        <div className="stat">
          <div className="stat__label">{t("risk.coverage")}</div>
          <div className="stat__value">
            {assessment.data_sufficiency?.coverage !== undefined
              ? `${Math.round(Number(assessment.data_sufficiency.coverage) * 100)}%`
              : "—"}
          </div>
          <div className="stat__hint">{assessment.data_sufficiency?.note || strengthText(assessment)}</div>
        </div>
      </div>
      {recommendations.length ? (
        <div className="chip-row" style={{ marginTop: 12 }}>
          {recommendations.map((item) => (
            <span className="chip" key={item}><Stethoscope size={12} /> {specialtyLabel(item)}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function strengthText(assessment: AssessmentResult): string {
  const coverage = Number(assessment.data_sufficiency?.coverage ?? 0);
  if (coverage >= 0.8) return i18n.t("risk.highCoverage");
  if (coverage >= 0.5) return i18n.t("risk.partialCoverage");
  return i18n.t("risk.lowCoverage");
}

function ExplanationView({ explanation }: { explanation: CopilotExplanation }) {
  const { t } = useTranslation();
  const summaryLlm = explanation.summary_llm || explanation.summary;
  const summaryDet = explanation.summary_deterministic || explanation.summary;
  const isLlm = explanation.method === "llm_assisted" && !!explanation.summary_llm;

  return (
    <section className="card card--accent">
      <div className="card__header">
        <div>
          <div className="eyebrow" style={{ marginBottom: 4 }}><Sparkles size={12} /> {t("copilot.explanation")}</div>
          <h2 className="card__title">{t("copilot.summary")}</h2>
        </div>
        <div className="stack-sm" style={{ alignItems: "flex-end" }}>
          {explanation.risk_level ? <RiskBadge level={explanation.risk_level} /> : null}
          <span className={`badge ${isLlm ? "badge--llm" : "badge--det"}`} title={isLlm ? t("copilot.llmRole") : t("copilot.deterministicRole")}>
            {isLlm ? `${t("copilot.llmLabel")} · ${explanation.method_llm || t("copilot.llm")}` : t("copilot.deterministicLabel")}
          </span>
        </div>
      </div>

      {/* Dual summary display: deterministic + LLM stacked with clear separation */}
      {summaryLlm || summaryDet ? (
        <div className="summary-dual" style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="summary-version" style={{ padding: 14, borderRadius: 8, border: `1px solid ${isLlm ? "var(--info)" : "var(--success)"}`, background: isLlm ? "color-mix(in srgb, var(--info) 10%, transparent)" : "color-mix(in srgb, var(--success) 10%, transparent)" }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
              <span style={{ fontWeight: 800, fontSize: 13, color: isLlm ? "var(--info)" : "var(--success)" }}>
                {isLlm ? "🤖" : "🖥️"}
              </span>
              <span style={{ fontWeight: 800, fontSize: 13 }}>{isLlm ? t("copilot.llmLabel") : t("copilot.deterministicLabel")}</span>
              <span className="small muted">{isLlm ? t("copilot.llmRole") : t("copilot.deterministicRole")}</span>
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.8 }}>{summaryLlm || summaryDet}</div>
          </div>
          {summaryDet && summaryDet !== summaryLlm && (
            <div className="summary-version" style={{ padding: 14, borderRadius: 8, border: "1px dashed var(--border)", background: "var(--bg-soft)" }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 10 }}>
                <span style={{ fontWeight: 800, fontSize: 13, color: "var(--success)" }}>🖥️</span>
                <span style={{ fontWeight: 800, fontSize: 13 }}>{t("copilot.deterministicLabel")}</span>
                <span className="small muted">{t("copilot.deterministicRole")}</span>
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.8 }}>{summaryDet}</div>
            </div>
          )}
        </div>
      ) : null}

      {explanation.urgent?.length ? (
        <div style={{ marginTop: 16 }}>
          <Alert variant="danger">
            <strong>{t("copilot.urgent")}</strong>
            <ul className="finding-list" style={{ marginTop: 8 }}>
              {explanation.urgent.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </Alert>
        </div>
      ) : null}

      {explanation.key_findings?.length ? (
        <div style={{ marginTop: 18 }}>
          <div className="field__label" style={{ marginBottom: 8 }}>{t("copilot.findings")}</div>
          <ul className="finding-list">
            {explanation.key_findings.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      ) : null}

      {explanation.recommendations?.length ? (
        <div style={{ marginTop: 18 }}>
          <div className="field__label" style={{ marginBottom: 8 }}>{t("copilot.recommendations")}</div>
          <ul className="rec-list">
            {explanation.recommendations.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
      ) : null}

      {explanation.data_sufficiency_note ? (
        <div className="alert" style={{ marginTop: 18 }}>
          <CircleHelp size={15} />
          <div>{explanation.data_sufficiency_note}</div>
        </div>
      ) : null}

      {/* Rule references with source links */}
      {explanation.rule_references?.length ? (
        <div style={{ marginTop: 18 }}>
          <div className="field__label" style={{ marginBottom: 8 }}>{t("copilot.ruleReferences")}</div>
          <ul className="finding-list" style={{ marginTop: 8 }}>
            {explanation.rule_references.map((ref) => (
              <li key={ref.rule_id} style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 6 }}>
                <span style={{ fontWeight: 700, color: "var(--text)" }}>{ref.rule_name}</span>{" "}
                <span className="small muted" style={{ fontSize: 11 }}>[{(ref.severity ?? 0).toFixed(1)}]</span>
                {ref.source_url ? (
                  <>
                    <br />
                    <a
                      href={ref.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="small muted"
                      style={{ color: "var(--info)", textDecoration: "underline", wordBreak: "break-all" }}
                    >
                      {ref.source_url}
                    </a>
                  </>
                ) : null}
                {ref.source_excerpt ? (
                  <>
                    <br />
                    <span className="small" style={{ color: "var(--text-muted)", fontStyle: "italic" }}>"{ref.source_excerpt}"</span>
                  </>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div className="disclaimer-bar" style={{ marginTop: 14 }}>
        <ShieldCheck size={16} />
        <span>{explanation.disclaimer || t("copilot.disclaimer")}</span>
      </div>
    </section>
  );
}

function LlmDebugPanel({ debug, onClear }: { debug: LlmDebug; onClear: () => void }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const raw = typeof debug.raw === "string" ? debug.raw : "";
  const parsed = debug.parsed === null || debug.parsed === undefined ? "" : JSON.stringify(debug.parsed, null, 2);
  const validation = Array.isArray(debug.validation) ? debug.validation.filter(Boolean) : [];
  const systemPrompt =
    typeof debug.system_prompt === "string" ? debug.system_prompt : "";
  const userPrompt = typeof debug.user_prompt === "string" ? debug.user_prompt : "";
  const copyRaw = async () => {
    try {
      await navigator.clipboard.writeText(raw || parsed);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard may be unavailable in an embedded browser; the raw text stays selectable.
    }
  };
  return (
    <section className="card debug-panel">
      <div className="card__header">
        <div>
          <div className="eyebrow" style={{ marginBottom: 4 }}><Bug size={12} /> {t("copilot.debugTitle")}</div>
          <h2 className="card__title">{t("copilot.debugRawTitle")}</h2>
          <p className="card__subtitle">{t("copilot.debugHint")}</p>
        </div>
        <div className="row">
          <span className={`badge ${debug.accepted ? "badge--low" : "badge--high"}`}>
            {debug.accepted ? t("copilot.debugAccepted") : t("copilot.debugRejected")}
          </span>
          <button type="button" className="btn btn--ghost btn--sm" onClick={() => void copyRaw()} disabled={!raw && !parsed}>
            {copied ? <Check size={13} /> : <Copy size={13} />}
            {copied ? t("copilot.debugCopied") : t("copilot.debugCopy")}
          </button>
          <button type="button" className="btn btn--ghost btn--sm" onClick={onClear}>{t("common.close")}</button>
        </div>
      </div>
      <Alert variant="danger">
        <div><strong>{t("copilot.debugWarningTitle")}</strong><div>{t("copilot.debugWarning")}</div></div>
      </Alert>
      <div className="debug-meta">
        <span className="badge">{debug.provider || t("common.unknown")} · {debug.model || t("common.unknown")}</span>
        <span className="badge">{t("copilot.debugAttempted")}: {debug.attempted ? t("common.yes") : t("common.no")}</span>
        <span className="badge">{t("copilot.debugLatency")}: {Number(debug.latency_ms || 0).toFixed(0)} ms</span>
        <span className="badge">{t("copilot.debugReason")}: {debug.reason || "—"}</span>
      </div>
      {validation.length ? (
        <div className="small muted" style={{ marginTop: 8 }}>{t("copilot.debugValidation")}: {validation.join(" · ")}</div>
      ) : null}
      <div className="debug-block">
        <div className="field__label">{t("copilot.debugRaw")}</div>
        <pre>{raw || t("copilot.debugEmpty")}</pre>
      </div>
      {parsed ? (
        <div className="debug-block">
          <div className="field__label">{t("copilot.debugParsed")}</div>
          <pre>{parsed}</pre>
        </div>
      ) : null}
      {systemPrompt ? (
        <div className="debug-block">
          <div className="field__label">{t("copilot.debugSystemPrompt")}</div>
          <pre>{systemPrompt}</pre>
        </div>
      ) : null}
      {userPrompt ? (
        <div className="debug-block">
          <div className="field__label">{t("copilot.debugUserPrompt")}</div>
          <pre>{userPrompt}</pre>
        </div>
      ) : null}
    </section>
  );
}

export function CopilotPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const { patientId, status, refresh } = usePatient();
  const [mode, setMode] = useState("all");
  const [audience, setAudience] = useState("patient");
  const [useLlm, setUseLlm] = useState(true);
  const [debugLlm, setDebugLlm] = useState(false);
  const [llmDebug, setLlmDebug] = useState<LlmDebug | null>(null);
  const [llmProcessing, setLlmProcessing] = useState(false);
  const [values, setValues] = useState<Record<string, number>>({});
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [assessment, setAssessment] = useState<AssessmentResult | null>(null);
  const [explanation, setExplanation] = useState<CopilotExplanation | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [copilotStatus, setCopilotStatus] = useState<CopilotStatus | null>(null);
  const [chatMode, setChatMode] = useState<ChatChannel>("consult");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatDraft, setChatDraft] = useState("");
  const [chatBusy, setChatBusy] = useState(false);
  const [chatState, setChatState] = useState<Record<string, unknown>>({});
  const [lastAssistantLabel, setLastAssistantLabel] = useState<{
    method?: string;
    provider?: string;
    model?: string;
  }>({});
  const [useDate, setUseDate] = useState(false);
  const [measurementDate, setMeasurementDate] = useState(todayInput);
  const [resetBusy, setResetBusy] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const nextMessageId = useRef(1);
  const chatThreadRef = useRef<HTMLDivElement>(null);

  const loadCopilotStatus = async () => {
    try {
      setCopilotStatus(await api.copilot.status());
    } catch {
      setCopilotStatus(null);
    }
  };

  const loadPatientValues = useCallback(async (pid: string) => {
    if (!pid.trim()) return;
    try {
      const table = await api.core.records(pid);
      const dates = [...table.dates].sort();
      // Load latest values for all metrics automatically
      const latest: Record<string, number> = {};
      for (const date of dates) {
        const cells = table.grid[date];
        if (cells) {
          for (const [metric, cell] of Object.entries(cells)) {
            if (cell?.value !== null && cell?.value !== undefined) {
              latest[metric] = Number(cell.value);
            }
          }
        }
      }
      setValues(latest);
    } catch {
      setValues({});
    }
  }, []);

  useEffect(() => {
    void loadCopilotStatus();
    // Đảm bảo model local đã được nạp/prompt đã được cache trước cuộc gọi LLM đầu tiên.
    void api.copilot.warmup().catch(() => {
      // Warmup không bắt buộc; bỏ qua nếu endpoint chưa có.
    });
  }, []);

  useEffect(() => {
    setAssessment(null);
    setExplanation(null);
    setWarnings([]);
    setChatMessages([]);
    setChatDraft("");
    setChatState({});
    setValues({});
    setLlmDebug(null);
    if (patientId) void loadPatientValues(patientId);
  }, [patientId, loadPatientValues]);

  useEffect(() => {
    setChatState((current) => ({
      metrics: values,
      mode,
      timestamp: (current.timestamp as string) || new Date().toISOString().slice(0, 10),
      assessment: current.assessment ?? null,
      last_expected: current.last_expected ?? null,
    }));
  }, [values, mode]);

  // Cuộn tự động xuống tin nhắn mới nhất khi có tin nhắn mới hoặc xong bận.
  useEffect(() => {
    const el = chatThreadRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [chatMessages, chatBusy]);

  const runAssessment = async () => {
    if (!Object.keys(values).length) {
      toast.error(t("copilot.empty"));
      return;
    }
    setBusy(true);
    setWarnings([]);
    setLlmDebug(null);
    setAssessment(null);
    setExplanation(null);
    try {
      // Always show the deterministic result first. It is fast and safe, so
      // the user never waits on the local model just to see the assessment.
      const data = await api.copilot.assess({
        patient_id: patientId,
        metrics: values,
        mode,
        audience,
        use_llm: false,
        debug_llm: false,
      });
      const assessmentResult = data.assessment;
      if (!data.ok || !assessmentResult) {
        toast.error(data.error || t("common.error"));
        return;
      }
      setAssessment(assessmentResult);
      setExplanation(data.explanation || null);
      setWarnings(data.warnings || []);
      setChatState((current) => ({
        ...current,
        metrics: values,
        mode,
        timestamp: new Date().toISOString().slice(0, 10),
        assessment: assessmentResult,
        last_expected: null,
      }));

      if (useLlm) {
        // The LLM only rewrites the validated draft. If it is slow or fails
        // safety checks, the deterministic explanation above stays visible.
        setLlmProcessing(true);
        try {
          const enhanced = await api.copilot.explain({
            assessment: assessmentResult,
            audience,
            use_llm: true,
            debug_llm: debugLlm,
          });
          setExplanation(enhanced);
          setLlmDebug(enhanced.llm_debug || null);
          const debug = (enhanced.llm_debug || {}) as Record<string, unknown>;
          if (debug.method === 'llm_assisted') {
            toast.success('LLM đã xử lý (' + Number(debug.latency_ms || 0).toFixed(0) + 'ms)');
          } else {
            toast.error('LLM bị từ chối, dùng bản tất định');
          }
        } finally {
          setLlmProcessing(false);
        }
      }
    } catch (error: any) {
      // More detailed error messages
      if (error.message?.includes('NetworkError') || error.message?.includes('fetch') || error.message?.includes('Failed to fetch')) {
        toast.error('Không kết nối được server. Hãy kiểm tra API lõi (cổng 8000) và Copilot (cổng 8001) đã chạy chưa.');
      } else if (error.message?.includes('404')) {
        toast.error('Endpoint không tìm thấy. Kiểm tra BFF (cổng 3000) đã chạy chưa.');
      } else if (error.message?.includes('500')) {
        toast.error('Lỗi server. Xem log Copilot hoặc API lõi để biết chi tiết.');
      } else if (error.message?.includes('408') || error.message?.includes('timeout')) {
        toast.error('Yêu cầu quá thời gian. Model AI có thể đang tải lần đầu, vui lòng thử lại.');
      } else {
        toast.error(error instanceof Error ? error.message : t("common.error"));
      }
      console.error('Assess error:', error);
    } finally {
      setBusy(false);
    }
  };

  const assistantLabelFor = (data: Record<string, unknown>) => {
    const debug = (data.llm_debug || {}) as Record<string, unknown> | undefined;
    const method =
      (typeof data.method === "string" && data.method) ||
      (debug && typeof debug.method === "string" ? debug.method : undefined);
    const provider =
      debug && typeof debug.provider === "string" ? debug.provider : undefined;
    const model = debug && typeof debug.model === "string" ? debug.model : undefined;
    const latency =
      debug && typeof debug.latency_ms === "number" ? debug.latency_ms : undefined;
    const reason =
      debug && typeof debug.reason === "string" ? debug.reason : undefined;
    const validation =
      debug && Array.isArray(debug.validation) ? debug.validation : undefined;
    return { method, provider, model, latency, reason, validation };
  };

  const [openDetailId, setOpenDetailId] = useState<number | null>(null);

  const appendAssistant = (data: Record<string, unknown>, channel: ChatChannel) => {
    const reply =
      typeof data.reply === "string"
        ? data.reply
        : typeof data.message === "string"
          ? data.message
          : "";
    if (!reply) return;
    const { method, provider, model, latency, reason, validation } =
      assistantLabelFor(data);
    setChatMessages((current) => [
      ...current,
      {
        id: nextMessageId.current++,
        role: "assistant",
        content: reply,
        channel,
        method,
        provider,
        model,
        stage: typeof data.stage === "string" ? data.stage : undefined,
        llmLatency: latency,
        llmReason: reason,
        llmValidation: validation,
      },
    ]);
  };

  const sendChat = async (text = chatDraft) => {
    const message = text.trim();
    if (!message || chatBusy) return;
    const channel = chatMode;
    setChatDraft("");
    setChatBusy(true);
    setChatMessages((current) => [
      ...current,
      { id: nextMessageId.current++, role: "user", content: message, channel },
    ]);
    try {
      if (channel === "journal") {
        const data = await api.core.chat({
          patient_id: patientId,
          message,
          ...(useDate && measurementDate ? { date: measurementDate } : {}),
        });
        appendAssistant(data as unknown as Record<string, unknown>, "journal");
        setLlmDebug(null);
        await refresh();
        await loadPatientValues(patientId);
        setAssessment(null);
        setExplanation(null);
        setWarnings([]);
      } else {
        const data = await api.copilot.chat({
          patient_id: patientId,
          message,
          state: chatState,
          use_llm: useLlm,
          debug_llm: debugLlm,
        });
        appendAssistant(data as unknown as Record<string, unknown>, "consult");
        setLlmDebug(data.llm_debug || data.explanation?.llm_debug || null);
        // Ghi nhớ mô hình/provider của lượt trả lời gần nhất để hiển thị tiêu đề hội thoại.
        const debug = (data.llm_debug || {}) as Record<string, unknown> | undefined;
        const lastMethod =
          (typeof data.method === "string" && data.method) ||
          (debug && typeof debug.method === "string" ? debug.method : undefined);
        const lastProvider =
          debug && typeof debug.provider === "string" ? debug.provider : undefined;
        const lastModel =
          debug && typeof debug.model === "string" ? debug.model : undefined;
        setLastAssistantLabel({ method: lastMethod, provider: lastProvider, model: lastModel });
        if (data.state && typeof data.state === "object") {
          const nextState = data.state as Record<string, unknown>;
          setChatState(nextState);
          const stateMetrics = nextState.metrics;
          if (stateMetrics && typeof stateMetrics === "object") {
            setValues((current) => {
              const next = { ...current };
              for (const [metric, value] of Object.entries(stateMetrics as Record<string, unknown>)) {
                if (next[metric] === undefined) next[metric] = Number(value);
              }
              return next;
            });
          }
        }
        if (data.suggestions?.length) setSuggestions(data.suggestions);
        if (data.assessment) setAssessment(data.assessment);
        if (data.explanation) setExplanation(data.explanation);
        if (data.warnings?.length) setWarnings(data.warnings);
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("common.error"));
    } finally {
      setChatBusy(false);
    }
  };

  const uploadFile = async (file: File) => {
    if (!file) return;
    if (file.size > 10 * 1024 * 1024) {
      toast.error(t("chat.fileTooLarge"));
      return;
    }
    setChatBusy(true);
    setChatMessages((current) => [
      ...current,
      {
        id: nextMessageId.current++,
        role: "user",
        content: `${t("common.upload")}: ${file.name}`,
        channel: "journal",
      },
    ]);
    try {
      const data = await api.core.chatFile(patientId, useDate ? measurementDate : "", file);
      appendAssistant(data as unknown as Record<string, unknown>, "journal");
      setLlmDebug(null);
      await refresh();
      await loadPatientValues(patientId);
      setAssessment(null);
      setExplanation(null);
      setWarnings([]);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("common.error"));
    } finally {
      setChatBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const resetPatient = async () => {
    if (!patientId) return;
    if (!window.confirm(t("chat.resetConfirm"))) return;
    setResetBusy(true);
    try {
      await api.core.reset(patientId);
      setChatMessages([]);
      setSuggestions([]);
      setChatState({});
      setAssessment(null);
      setExplanation(null);
      setWarnings([]);
      setLlmDebug(null);
      setValues({});
      await refresh();
      toast.success(t("common.success"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("common.error"));
    } finally {
      setResetBusy(false);
    }
  };

  const consultPrompts = [t("copilot.quickWhy"), t("copilot.quickNext"), t("copilot.quickWhich"), t("copilot.quickWhen")];
  const journalPrompts = [t("chat.exampleBp"), t("chat.exampleGlucose")];
  const quickPrompts = chatMode === "consult" ? consultPrompts : journalPrompts;
  const visiblePrompts = chatMode === "consult" && suggestions.length ? suggestions.slice(0, 4) : quickPrompts;
  const copilotModel = copilotStatus?.llm?.model || copilotStatus?.ollama_model;
  const copilotProvider = copilotStatus?.llm?.provider || copilotStatus?.llm_provider;
  const cloudDisabled = copilotStatus?.llm?.cloud_allowed === false || copilotStatus?.allow_cloud === false;

  return (
    <div className="page">
      <PageHeader
        title={t("copilot.title")}
        subtitle={t("copilot.subtitleShort")}
        actions={
          <div className="row">
            <span className={`badge ${copilotStatus ? "badge--low" : "badge--medium"}`}>
              <span className={`dot ${copilotStatus ? "dot--online" : "dot--offline"}`} />
              {copilotStatus ? t("common.online") : t("common.offline")}
            </span>
            {copilotModel ? (
              <span className="badge badge--primary" title={copilotProvider || ""}>
                {copilotProvider ? `${copilotProvider} · ${copilotModel}` : copilotModel}
              </span>
            ) : null}
          </div>
        }
      />

      <div className="stack-lg">
        <PatientBar mode={mode} onModeChange={setMode} />

        {/* ========== Primary: automatic patient record + assessment ========== */}
        <div className="grid grid--2" style={{ alignItems: "start" }}>
          <section className="card">
            <div className="card__header">
              <div>
                <h2 className="card__title">{t("copilot.metricsTitle")}</h2>
                <p className="card__subtitle">
                  {status?.has_data ? t("copilot.metricsLoaded") : t("copilot.metricsHint")}
                </p>
              </div>
              <div className="card__icon"><Activity size={18} /></div>
            </div>

            <label className="row" style={{ gap: 6, fontSize: 12, marginTop: 20 }}>
              <input type="checkbox" checked={useLlm} onChange={(event) => setUseLlm(event.target.checked)} />
              {t("copilot.llm")}
            </label>
            <label
              className="row debug-toggle"
              style={{ gap: 6, fontSize: 12, marginTop: 20 }}
              title={t("copilot.debugHint")}
            >
              <input type="checkbox" checked={debugLlm} onChange={(event) => setDebugLlm(event.target.checked)} disabled={!useLlm} />
              <Bug size={13} /> {t("copilot.debug")}
            </label>

            <div style={{ marginTop: 16 }}>
              <button type="button" className="btn btn--primary" onClick={() => void runAssessment()} disabled={busy || !Object.keys(values).length}>
                {busy ? <span className="spinner" /> : <WandSparkles size={15} />}
                {busy ? (assessment ? t("copilot.explaining") : t("copilot.assessing")) : t("copilot.assess")}
              </button>
              {busy && useLlm ? (
                <p className="small muted" style={{ margin: "8px 0 0" }}>{t("copilot.llmWait")}</p>
              ) : null}
              {llmProcessing && (
                <p className="small" style={{ margin: "8px 0 0", color: "var(--info)" }}>
                  <span className="spinner" /> Đang gọi mô hình AI (qwen3:8b)...
                </p>
              )}
            </div>

            {warnings.length ? (
              <Alert variant="warning">
                <strong>{t("copilot.validation")}</strong>
                <ul className="finding-list" style={{ marginTop: 6 }}>
                  {warnings.map((warning) => <li key={warning}>{warning}</li>)}
                </ul>
              </Alert>
            ) : null}
          </section>

          <div className="stack">
            {assessment ? <CompactAssessment assessment={assessment} /> : null}
            {explanation ? <ExplanationView explanation={explanation} /> : (
              <section className="card">
                <div className="empty-state">
                  <Brain size={32} />
                  <strong>{t("copilot.empty")}</strong>
                  <p>{t("copilot.noAssessment")}</p>
                </div>
              </section>
            )}
          </div>
        </div>

        <div className="stack">
          <section className="card">
            <div className="card__header">
              <div>
                <h2 className="card__title">{t("copilot.chatTitle")}</h2>
                <p className="card__subtitle">
                  {chatMode === "consult" ? t("copilot.consultHint") : t("copilot.journalHint")}
                </p>
              </div>
              <div className="segmented" role="group" aria-label={t("copilot.chatMode")}>
                <button
                  type="button"
                  className={chatMode === "consult" ? "is-active" : ""}
                  onClick={() => {
                    setChatMode("consult");
                    setSuggestions([]);
                  }}
                  aria-pressed={chatMode === "consult"}
                >
                  {t("copilot.modeConsult")}
                </button>
                <button
                  type="button"
                  className={chatMode === "journal" ? "is-active" : ""}
                  onClick={() => {
                    setChatMode("journal");
                    setSuggestions([]);
                  }}
                  aria-pressed={chatMode === "journal"}
                >
                  {t("copilot.modeJournal")}
                </button>
              </div>
            </div>
            <div
              ref={chatThreadRef}
              className="stack-sm chat-thread"
              style={{ minHeight: 260, maxHeight: 520, overflowY: "auto" }}
            >
              {chatMessages.length === 0 ? (
                <div className="alert">
                  <Sparkles size={15} />
                  <div>{chatMode === "consult" ? t("copilot.chatEmptyConsult") : t("chat.welcomeShort")}</div>
                </div>
              ) : (
                chatMessages.map((message) => (
                  <div className={`message message--${message.role}`} key={message.id}>
                    <div className="message__meta">
                      {message.role === "user" ? t("chat.you") : t("chat.assistant")}
                      <span className="muted"> · {message.channel === "journal" ? t("copilot.modeJournal") : t("copilot.modeConsult")}</span>
                      {message.role === "assistant" && message.method ? (
                        <>
                          <span
                            className={`badge ${message.method === "llm_assisted" ? "badge--llm" : "badge--det"}`}
                            title={message.model || ""}
                          >
                            {message.method === "llm_assisted"
                              ? `${t("copilot.llm")} · ${message.model || t("common.unknown")}`
                              : t("copilot.deterministic")}
                          </span>
                          <button
                            type="button"
                            className="btn btn--ghost btn--xs"
                            onClick={() =>
                              setOpenDetailId(
                                openDetailId === message.id ? null : message.id
                              )
                            }
                            title={
                              openDetailId === message.id
                                ? t("copilot.detailHide")
                                : t("copilot.detailShow")
                            }
                          >
                            {openDetailId === message.id
                              ? <RotateCcw size={11} />
                              : <Bug size={11} />}
                          </button>
                          {openDetailId === message.id ? (
                            <div className="message-detail">
                              <div className="detail-row">
                                <span className="detail-label">{t("copilot.detailMethod")}</span>
                                <span className="detail-value">{message.method || "—"}</span>
                              </div>
                              {message.provider || message.model ? (
                                <div className="detail-row">
                                  <span className="detail-label">{t("copilot.detailModel")}</span>
                                  <span className="detail-value">
                                    {message.provider ? `${message.provider} · ` : ""}{message.model || t("common.unknown")}
                                  </span>
                                </div>
                              ) : null}
                              {message.llmLatency != null ? (
                                <div className="detail-row">
                                  <span className="detail-label">{t("copilot.detailLatency")}</span>
                                  <span className="detail-value">{Number(message.llmLatency).toFixed(0)} ms</span>
                                </div>
                              ) : null}
                              {message.llmValidation?.length ? (
                                <div className="detail-row">
                                  <span className="detail-label">{t("copilot.detailValidation")}</span>
                                  <span className="detail-value">{message.llmValidation.join(" · ")}</span>
                                </div>
                              ) : null}
                              {message.llmReason ? (
                                <div className="detail-row">
                                  <span className="detail-label">{t("copilot.detailReason")}</span>
                                  <span className="detail-value">{message.llmReason}</span>
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                    <div className="message__bubble">
                      {message.role === "assistant" ? <Markdown>{message.content}</Markdown> : message.content}
                    </div>
                  </div>
                ))
              )}
              {chatBusy ? <div className="row small muted"><span className="spinner" /> {t("common.loading")}</div> : null}
            </div>
            <div className="quick-prompts" style={{ margin: "12px 0" }}>
              {visiblePrompts.map((prompt: string) => (
                <button type="button" className="quick-prompt" key={prompt} onClick={() => void sendChat(prompt)}>
                  {prompt}
                </button>
              ))}
            </div>
            <div className="composer-wrapper">
              <textarea
                className="textarea"
                rows={1}
                value={chatDraft}
                onChange={(event) => setChatDraft(event.target.value)}
                placeholder={chatMode === "consult" ? t("copilot.chatPlaceholder") : t("chat.placeholder")}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendChat();
                  }
                }}
              />
              <button type="button" className="btn-send" onClick={() => void sendChat()} disabled={chatBusy || !chatDraft.trim()} title="Gửi">
                {chatBusy ? <span className="spinner" /> : <ArrowUp size={14} />}
              </button>
            </div>
            {chatMode === "journal" ? (
              <div className="composer-tools">
                <input
                  ref={fileInput}
                  type="file"
                  accept=".pdf,.docx,.doc,.txt,application/pdf,text/plain"
                  className="sr-only"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void uploadFile(file);
                  }}
                />
                <button type="button" className="btn btn--sm" onClick={() => fileInput.current?.click()} disabled={chatBusy} title={t("chat.fileType")}>
                  <Paperclip size={14} /> {t("chat.attach")}
                </button>
                <label className="row" style={{ gap: 6, fontSize: 11.5, color: "var(--text-soft)" }}>
                  <input type="checkbox" checked={useDate} onChange={(event) => setUseDate(event.target.checked)} />
                  {t("chat.dateToggle")}
                </label>
                {useDate ? (
                  <input
                    className="input input--sm"
                    style={{ width: 150 }}
                    type="date"
                    value={measurementDate}
                    onChange={(event) => setMeasurementDate(event.target.value)}
                    aria-label={t("chat.date")}
                  />
                ) : null}
                <button
                  type="button"
                  className="btn btn--ghost btn--sm"
                  style={{ marginLeft: "auto" }}
                  onClick={() => void resetPatient()}
                  disabled={resetBusy || !patientId}
                >
                  {resetBusy ? <span className="spinner" /> : <RotateCcw size={14} />}
                  {t("chat.reset")}
                </button>
              </div>
            ) : null}
            {debugLlm && llmDebug ? (
              <details className="card debug-panel" open={true}>
                <summary className="card__header" style={{ cursor: "pointer", padding: "10px 14px" }}>
                  <Bug size={13} /> {t("copilot.debugTitle")}
                  <span style={{ marginLeft: 8, fontSize: 11, color: "var(--text-muted)" }}>
                    {llmDebug.accepted ? t("copilot.debugAccepted") : t("copilot.debugRejected")}
                  </span>
                </summary>
                <LlmDebugPanel debug={llmDebug} onClear={() => setLlmDebug(null)} />
              </details>
            ) : null}
          </section>
        </div>
      </div>
    </div>
  );
}

