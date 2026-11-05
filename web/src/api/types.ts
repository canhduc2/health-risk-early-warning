export type RiskLevel = "THAP" | "TRUNG_BINH" | "CAO" | "INSUFFICIENT_DATA" | string;

export interface PatientStatus {
  patient_id: string;
  has_data: boolean;
  unique_dates: number;
  needed_dates: number;
  ready: boolean;
  metrics: Array<{ metric: string; days: number; dates: string[] }>;
  total_records: number;
}

export interface RecordsTable {
  patient_id: string;
  metrics: string[];
  dates: string[];
  grid: Record<string, Record<string, { value: number | null; unit: string }>>;
}

export interface EvidenceItem {
  rule_id?: string;
  rule?: string;
  system?: string;
  severity?: number;
  metric?: string;
  current?: number;
  baseline_mean?: number;
  z_score?: number;
  message?: string;
  source_url?: string;
  source_page?: string;
  source_section?: string;
  source_excerpt?: string;
  [key: string]: unknown;
}

export interface MetricDetail {
  metric: string;
  name?: string;
  unit?: string;
  current?: number;
  baseline_mean?: number | null;
  delta?: number | null;
  pct_change?: number | null;
  trend?: string | null;
  trend_name?: string | null;
  z_score?: number | null;
  flagged?: boolean;
  range_lo?: number | null;
  range_hi?: number | null;
  range_status?: string;
  deviation?: number | null;
  [key: string]: unknown;
}

export interface DataSufficiency {
  level?: string;
  coverage?: number;
  needed?: number;
  present?: number;
  missing_metrics?: string[];
  present_metrics?: string[];
  note?: string;
  [key: string]: unknown;
}

export interface MlModelResult {
  key: string;
  name: string;
  family?: string;
  ml_score?: number;
  ml_level?: string;
  total_score?: number;
  total_level?: string;
  [key: string]: unknown;
}

export interface AssessmentResult {
  patient_id?: string;
  modes?: string[] | null;
  risk_level: RiskLevel;
  risk_score?: number;
  affected_systems?: string[];
  evidence?: EvidenceItem[];
  recommendations?: string[];
  components?: Record<string, number>;
  metrics_detail?: MetricDetail[];
  data_sufficiency?: DataSufficiency;
  disclaimer?: string;
  tier1_summary?: Record<string, unknown>;
  tier1_note?: string | null;
  ml_score?: number | null;
  ml_all?: MlModelResult[];
  message?: string;
  [key: string]: unknown;
}

export interface ChatResponse {
  reply: string;
  ready?: boolean;
  collected?: PatientStatus;
  snapshot?: Record<string, number>;
  risk_level?: RiskLevel;
  silent?: boolean;
  [key: string]: unknown;
}

export interface CopilotStatus {
  ok?: boolean;
  service?: string;
  version?: string;
  copilot_version?: string;
  api_base_url?: string;
  api?: {
    online?: boolean;
    base_url?: string;
    version?: string;
  };
  llm?: {
    provider?: string;
    model?: string;
    available?: boolean;
    privacy?: string;
    cloud_allowed?: boolean;
  };
  kb?: {
    available?: boolean;
    version?: string;
    rules?: number;
    metrics?: number;
  };
  llm_provider?: string;
  allow_cloud?: boolean;
  ollama_model?: string;
  ollama_url?: string;
  catalog_metrics?: string[];
  [key: string]: unknown;
}

export interface CopilotCatalog {
  metrics?: Array<{
    key: string;
    name_vi?: string;
    unit?: string;
    normal_range?: string;
    prompt?: string;
    hint?: string;
  }>;
  modes?: Array<{ key: string; label?: string }>;
  [key: string]: unknown;
}

export interface LlmDebug {
  attempted?: boolean;
  accepted?: boolean;
  reason?: string;
  validation?: string[];
  raw?: string | null;
  parsed?: unknown;
  latency_ms?: number;
  provider?: string;
  model?: string;
  method?: string;
  [key: string]: unknown;
}

export interface CopilotExplanation {
  summary?: string;
  risk_level?: string;
  risk_level_vi?: string;
  risk_score?: number | null;
  affected_systems?: string[];
  key_findings?: string[];
  recommendations?: string[];
  data_sufficiency_note?: string;
  urgent?: string[];
  disclaimer?: string;
  method?: string;
  llm_debug?: LlmDebug | null;
  [key: string]: unknown;
}

export interface CopilotChatResponse {
  reply?: string;
  answer?: string;
  message?: string;
  suggestions?: string[];
  state?: Record<string, unknown>;
  assessment?: AssessmentResult;
  explanation?: CopilotExplanation;
  disclaimer?: string;
  ok?: boolean;
  error?: string;
  warnings?: string[];
  method?: string;
  provider?: string;
  model?: string;
  stage?: string;
  llm_debug?: LlmDebug | null;
  [key: string]: unknown;
}

export interface KbMetric {
  name?: string;
  unit?: string;
  range?: [number, number] | number[];
  [key: string]: unknown;
}

export interface KbRule {
  rule_id: string;
  name: string;
  system: string;
  condition: Record<string, unknown>;
  severity: number;
  specialty?: string;
  evidence?: string;
  source_url?: string;
  source_page?: string;
  source_section?: string;
  source_excerpt?: string;
  modes?: string[];
  status?: string;
  rule_version?: number;
  [key: string]: unknown;
}

export interface KnowledgeBase {
  meta?: Record<string, unknown>;
  metrics?: Record<string, KbMetric>;
  rules?: KbRule[];
  system_labels?: Record<string, string>;
  [key: string]: unknown;
}

export interface AuditEntry {
  timestamp?: string;
  actor?: string;
  action?: string;
  rule_id?: string;
  details?: string;
  [key: string]: unknown;
}

export interface BenchmarkSummary {
  summary?: Record<string, unknown>;
  experiments?: Array<Record<string, unknown>>;
  input_metrics?: Array<Record<string, unknown>>;
}

export interface BenchmarkResearch {
  label_sensitivity?: Record<string, unknown>;
  baseline_stability?: Record<string, unknown>;
  weight_sensitivity?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
  temporal_validation?: Record<string, unknown>;
  temporal_validation_mimic?: Record<string, unknown>;
  complete_case?: Record<string, unknown>;
  evidence_status?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export interface SystemHealth {
  ok: boolean;
  checkedAt: string;
  services: {
    core: ServiceHealth;
    copilot: ServiceHealth;
    ollama: ServiceHealth & { models?: string[] };
  };
  model: {
    name: string;
    contextWindow: number;
    available: boolean;
    installedModels: string[];
  };
}

export interface ServiceHealth {
  ok: boolean;
  status: number;
  latencyMs: number;
  url: string;
  detail?: unknown;
  models?: string[];
}

export interface SystemInfo {
  app: string;
  version: string;
  architecture: Record<string, string | number>;
  services: Record<string, string>;
  model: { name: string; contextWindow: number; quantization?: string };
  safety: { decisionSupportOnly: boolean; disclaimerRequired: boolean };
}
