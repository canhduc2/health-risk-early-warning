import { ClipboardCheck, ShieldCheck } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { api } from "../api/client";
import type { AssessmentResult, RecordsTable } from "../api/types";
import { AssessmentView } from "../components/AssessmentView";
import { MetricTrends } from "../components/MetricTrends";
import { PatientBar } from "../components/PatientBar";
import { usePatient } from "../components/PatientContext";
import { RecordGrid } from "../components/RecordGrid";
import { useToast } from "../components/Toast";
import { EmptyState, LoadingBlock, PageHeader } from "../components/Ui";

export function AssessmentPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const { patientId, status, refresh } = usePatient();
  const [mode, setMode] = useState("all");
  const [windowDays, setWindowDays] = useState<"10" | "30" | "60" | "all">("all");
  const [table, setTable] = useState<RecordsTable | null>(null);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [loadingTable, setLoadingTable] = useState(false);
  const [assessing, setAssessing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadTable = useCallback(
    async (pid: string) => {
      if (!pid.trim()) {
        setTable(null);
        return;
      }
      setLoadingTable(true);
      setError(null);
      try {
        const allRecords = await api.core.records(pid);
        // Apply window filter: keep only dates within the selected window
        if (windowDays !== "all" && allRecords) {
          const days = parseInt(windowDays, 10);
          const cutoff = new Date();
          cutoff.setDate(cutoff.getDate() - days);
          const filteredDates: string[] = [];
          const filteredGrid: RecordsTable["grid"] = {};
          for (const [date, metrics] of Object.entries(allRecords.grid)) {
            const d = new Date(date);
            if (d >= cutoff) {
              filteredDates.push(date);
              filteredGrid[date] = metrics;
            }
          }
          setTable({ ...allRecords, dates: filteredDates, grid: filteredGrid });
        } else {
          setTable(allRecords);
        }
      } catch (exc) {
        setTable(null);
        setError(exc instanceof Error ? exc.message : t("common.error"));
      } finally {
        setLoadingTable(false);
      }
    },
    [t],
  );

  useEffect(() => {
    const timer = window.setTimeout(() => void loadTable(patientId), 150);
    return () => window.clearTimeout(timer);
  }, [patientId, loadTable]);

  useEffect(() => {
    setResult(null);
  }, [patientId]);

  const reload = useCallback(async () => {
    await loadTable(patientId);
    await refresh();
  }, [loadTable, patientId, refresh]);

  const runAssessment = async () => {
    if (!table || table.dates.length === 0) {
      toast.error(t("assessment.noRecords"));
      return;
    }
    const records: Array<{ timestamp: string; metric: string; value: number }> = [];
    for (const [date, metrics] of Object.entries(table.grid)) {
      for (const [metric, cell] of Object.entries(metrics)) {
        if (cell.value !== null && cell.value !== undefined && Number.isFinite(Number(cell.value))) {
          records.push({ timestamp: date, metric, value: Number(cell.value) });
        }
      }
    }
    if (!records.length) {
      toast.error(t("assessment.noRecords"));
      return;
    }
    setAssessing(true);
    setError(null);
    try {
      const assessment = await api.core.assess({ patient_id: patientId, mode, records });
      setResult(assessment);
      toast.success(t("assessment.resultTitle"));
      await reload();
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : t("common.error");
      setError(message);
      toast.error(message);
    } finally {
      setAssessing(false);
    }
  };

  return (
    <div className="page">
      <PageHeader
        title={t("assessment.title")}
        subtitle={t("assessment.subtitleShort")}
        actions={
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => void runAssessment()}
            disabled={assessing || loadingTable || !status?.has_data}
          >
            {assessing ? <span className="spinner" /> : <ClipboardCheck size={15} />}
            {assessing ? t("assessment.running") : t("assessment.run")}
          </button>
        }
      />

      <div className="stack-lg">
        <PatientBar mode={mode} onModeChange={setMode} />

        <div className="field patient-bar__window">
          <label htmlFor="window-days">{t("assessment.window")}</label>
          <select
            id="window-days"
            className="select"
            value={windowDays}
            onChange={(event) => setWindowDays(event.target.value as "10" | "30" | "60" | "all")}
          >
            <option value="10">{t("assessment.window10")}</option>
            <option value="30">{t("assessment.window30")}</option>
            <option value="60">{t("assessment.window60")}</option>
            <option value="all">{t("assessment.windowAll")}</option>
          </select>
        </div>

        {error ? (
          <div className="alert alert--danger" role="alert">
            <ShieldCheck size={16} />
            <div>{error}</div>
          </div>
        ) : null}

        {loadingTable && !table ? (
          <div className="card">
            <LoadingBlock label={t("common.loading")} />
          </div>
        ) : (
          <RecordGrid table={table} patientId={patientId} onChanged={reload} />
        )}

        <div className="grid grid--assessment">
          <MetricTrends table={table} />

          <div className="stack">
            {result ? (
              <AssessmentView result={result} />
            ) : (
              <section className="card">
                <EmptyState
                  icon={<ClipboardCheck size={28} />}
                  title={t("assessment.emptyTitle")}
                  body={t("assessment.emptyBodyShort")}
                />
              </section>
            )}
            <div className="disclaimer-bar">
              <ShieldCheck size={16} />
              <span>{t("disclaimer.short")}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
