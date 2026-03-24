import { useCallback, useEffect, useRef, useState } from "react";
import MetricCard from "../../components/MetricCard.tsx";
import { API } from "../../config/api-routes.ts";
import { useCoordizerStats, useModalStatus, useTrainingStats } from "../../hooks/index.ts";
import type { ModalAdapterInfo, TrainingUploadResponse } from "../../types/consciousness.ts";

const E8_PRIMITIVES = [
  "PER",
  "MEM",
  "ACT",
  "PRD",
  "ETH",
  "META",
  "HRT",
  "REL",
  "MIX",
] as const;

const KERNEL_SPECIALIZATIONS = [
  "genesis", "heart", "perception", "memory",
  "action", "strategy", "ethics", "meta", "ocean",
] as const;

/** Must match backend ProcessingMode enum: fast / standard / deep */
const PROCESSING_MODES = [
  { value: "fast", label: "Fast (no enrichment)" },
  { value: "standard", label: "Standard (Q&A + tags)" },
  { value: "deep", label: "Deep (full extraction)" },
] as const;

/** Per-file upload state for bulk tracking */
interface FileUploadJob {
  file: File;
  jobId: string | null;
  status: "queued" | "uploading" | "processing" | "done" | "error";
  result: TrainingUploadResponse | null;
}

export default function Training() {
  const { data: stats, loading, refetch } = useTrainingStats();
  const { data: coordizerStats, error: coordizerError } = useCoordizerStats();
  const { data: modalStatus } = useModalStatus();

  const [files, setFiles] = useState<File[]>([]);
  const [category, setCategory] = useState("curriculum");
  const [mode, setMode] = useState<string>("standard");
  const [e8Prim, setE8Prim] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [jobs, setJobs] = useState<FileUploadJob[]>([]);
  const [exportData, setExportData] = useState<{ count: number } | null>(null);
  const [exporting, setExporting] = useState(false);
  const [trainTarget, setTrainTarget] = useState<string>("all");
  const [triggeringTraining, setTriggeringTraining] = useState(false);
  const [trainingResult, setTrainingResult] = useState<{ status: string; error?: string } | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<{ status: string; files_synced?: number; total_records_pushed?: number; error?: string } | null>(null);
  const [modalData, setModalData] = useState<{ total_files?: number; total_records?: number; files?: { path: string; records: number; size_kb: number; modified_at: string }[] } | null>(null);
  const [modalDataLoading, setModalDataLoading] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Poll active jobs every 2s
  useEffect(() => {
    const activeJobs = jobs.filter((j) => j.status === "processing" && j.jobId);
    if (activeJobs.length === 0) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    if (pollRef.current) return; // already polling

    pollRef.current = setInterval(async () => {
      const stillActive = jobs.filter(
        (j) => j.status === "processing" && j.jobId,
      );
      if (stillActive.length === 0) {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        return;
      }

      for (const job of stillActive) {
        try {
          const resp = await fetch(API.trainingUploadStatus(job.jobId!));
          if (!resp.ok) continue;
          const data: TrainingUploadResponse = await resp.json();
          if (data.status !== "processing") {
            setJobs((prev) =>
              prev.map((j) =>
                j.jobId === job.jobId
                  ? { ...j, status: "done", result: data }
                  : j,
              ),
            );
          }
        } catch {
          /* network hiccup — retry next tick */
        }
      }
    }, 2000);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [jobs]);

  // Refresh stats when all jobs complete
  useEffect(() => {
    const allDone =
      jobs.length > 0 &&
      jobs.every((j) => j.status === "done" || j.status === "error");
    if (allDone && refetch) {
      refetch();
    }
  }, [jobs, refetch]);

  const handleUpload = useCallback(async () => {
    if (files.length === 0 || uploading) return;
    setUploading(true);

    // Initialize job entries for all selected files
    const initialJobs: FileUploadJob[] = files.map((f) => ({
      file: f,
      jobId: null,
      status: "queued" as const,
      result: null,
    }));
    setJobs(initialJobs);

    // Upload sequentially to avoid overwhelming the backend
    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      // Mark as uploading
      setJobs((prev) =>
        prev.map((j, idx) => (idx === i ? { ...j, status: "uploading" } : j)),
      );

      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("category", category);
        formData.append("mode", mode);
        if (e8Prim) formData.append("e8_override", e8Prim);

        const resp = await fetch(API.trainingUpload, {
          method: "POST",
          body: formData,
        });
        if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
        const data = await resp.json();

        if (data.status === "processing" && data.job_id) {
          // Backend accepted — move to polling
          setJobs((prev) =>
            prev.map((j, idx) =>
              idx === i
                ? { ...j, jobId: data.job_id, status: "processing" }
                : j,
            ),
          );
        } else if (data.status === "error") {
          setJobs((prev) =>
            prev.map((j, idx) =>
              idx === i ? { ...j, status: "error", result: data } : j,
            ),
          );
        } else {
          // Immediate result (JSONL pass-through, validation error)
          setJobs((prev) =>
            prev.map((j, idx) =>
              idx === i ? { ...j, status: "done", result: data } : j,
            ),
          );
        }
      } catch (err) {
        setJobs((prev) =>
          prev.map((j, idx) =>
            idx === i
              ? {
                ...j,
                status: "error",
                result: {
                  status: "error",
                  filename: file.name,
                  chunks_written: 0,
                  enriched: 0,
                  qa_pairs: 0,
                  category,
                  mode,
                  processing_time_s: 0,
                  error: err instanceof Error ? err.message : String(err),
                },
              }
              : j,
          ),
        );
      }
    }

    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    setUploading(false);
  }, [files, category, mode, e8Prim, uploading]);

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      const resp = await fetch(API.trainingExport);
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setExportData({ count: data.count ?? 0 });
    } catch {
      setExportData({ count: -1 });
    } finally {
      setExporting(false);
    }
  }, []);

  const fetchModalData = useCallback(async () => {
    setModalDataLoading(true);
    try {
      const resp = await fetch(API.trainingModalData);
      if (!resp.ok) throw new Error(`${resp.status}`);
      const data = await resp.json();
      if (data.status === "ok") setModalData(data);
    } catch {
      /* best-effort */
    } finally {
      setModalDataLoading(false);
    }
  }, []);

  const handleSync = useCallback(async () => {
    setSyncing(true);
    setSyncResult(null);
    try {
      const resp = await fetch(API.trainingSync, { method: "POST" });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setSyncResult(data);
      fetchModalData();
    } catch (err) {
      setSyncResult({ status: "error", error: err instanceof Error ? err.message : String(err) });
    } finally {
      setSyncing(false);
    }
  }, [fetchModalData]);

  // Fetch Modal data stats on mount
  useEffect(() => { fetchModalData(); }, [fetchModalData]);

  const completedCount = jobs.filter((j) => j.status === "done").length;
  const errorCount = jobs.filter((j) => j.status === "error").length;
  const activeCount = jobs.filter(
    (j) => j.status === "uploading" || j.status === "processing",
  ).length;

  if (loading) {
    return <div className="dash-loading">Loading training data...</div>;
  }

  return (
    <div>
      <div className="dash-header">
        <h1 className="dash-title">Training</h1>
        <div className="dash-subtitle">
          Ingest documents for kernel training (QLoRA) and external LLM fine-tuning (GPT-4.1)
        </div>
      </div>

      {/* Stats Grid */}
      <div className="dash-grid">
        <MetricCard
          label="Conversations"
          value={stats?.conversations ?? 0}
          color="var(--accent)"
        />
        <MetricCard
          label="Feedback"
          value={stats?.feedback ?? 0}
          color="var(--phi)"
        />
        <MetricCard
          label="Curriculum Chunks"
          value={stats?.curriculum_chunks ?? 0}
          color="var(--kappa)"
        />
        <MetricCard
          label="Uploads"
          value={stats?.uploads ?? 0}
          color="var(--gamma)"
        />
      </div>

      {/* Training Data Inventory */}
      {stats?.files && stats.files.length > 0 && (
        <div className="dash-section">
          <div className="dash-section-title">Training Data Inventory</div>
          <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
            {stats.files.length} file{stats.files.length !== 1 ? "s" : ""} ingested.
            Check filenames before uploading to avoid duplicates.
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "4px",
              maxHeight: "260px",
              overflowY: "auto",
            }}
          >
            {stats.files.map((f) => (
              <div
                key={f.filename}
                className="dash-card"
                style={{
                  padding: "8px 12px",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: "12px",
                }}
              >
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div
                    style={{
                      fontFamily: "var(--mono)",
                      fontSize: "12px",
                      fontWeight: 600,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {f.filename}
                  </div>
                  <div style={{ fontSize: "11px", color: "var(--text-secondary)", marginTop: "2px" }}>
                    {f.chunks} chunks · {f.enriched} enriched ·{" "}
                    {Object.entries(f.e8_tags).map(([k, v]) => `${k}:${v}`).join(" ")}
                  </div>
                </div>
                <div
                  style={{
                    fontSize: "10px",
                    color: "var(--text-secondary)",
                    fontFamily: "var(--mono)",
                    whiteSpace: "nowrap",
                  }}
                >
                  {new Date(f.uploaded_at).toLocaleDateString()}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Coordizer / Collective Vocabulary */}
      <div className="dash-section">
        <div className="dash-section-title">Collective Vocabulary</div>
        <div className="dash-grid">
          <MetricCard
            label="Model Vocab Size"
            value={coordizerStats?.vocab_size ?? 0}
            color="var(--accent)"
          />
          <MetricCard
            label="Resonance Bank Tokens"
            value={coordizerStats?.bank_size ?? 0}
            color="var(--kappa)"
          />
          <MetricCard
            label="Basin Dim"
            value={coordizerStats?.dim ?? 64}
            color="var(--phi)"
          />
          <MetricCard
            label="Tier Buckets"
            value={
              coordizerStats
                ? Object.keys(coordizerStats.tier_distribution).length
                : 0
            }
            color="var(--gamma)"
          />
        </div>

        {coordizerError && (
          <div className="dash-card" style={{ marginTop: "10px", color: "var(--error)" }}>
            Coordizer stats unavailable: {coordizerError}
          </div>
        )}
        {coordizerStats && (
          <div className="dash-card" style={{ marginTop: "10px" }}>
            <div className="dash-row">
              <span className="dash-row-label">Tier distribution</span>
              <span
                className="dash-row-value"
                style={{ fontFamily: "var(--mono)", fontSize: "12px" }}
              >
                {Object.entries(coordizerStats.tier_distribution)
                  .map(([k, v]) => `${k}:${v}`)
                  .join("  ")}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Modal Kernel Adapters */}
      <div className="dash-section">
        <div className="dash-section-title">Kernel Adapters (Modal QLoRA)</div>
        {modalStatus?.status === "unavailable" ? (
          <div className="dash-card" style={{ color: "var(--text-secondary)", fontSize: "13px" }}>
            MODAL_TRAINING_URL not configured — adapter status unavailable.
          </div>
        ) : modalStatus?.status === "error" ? (
          <div className="dash-card" style={{ color: "var(--error)", fontSize: "13px" }}>
            Modal status error: {modalStatus.error}
          </div>
        ) : (
          <>
            {modalStatus?.health && (
              <div className="dash-card" style={{ marginBottom: "8px" }}>
                <div className="dash-row">
                  <span className="dash-row-label">Model</span>
                  <span className="dash-row-value">{modalStatus.health.model_id ?? "unknown"}</span>
                </div>
                <div className="dash-row">
                  <span className="dash-row-label">Training Active</span>
                  <span className="dash-row-value" style={{ color: modalStatus.health.training_active ? "var(--kappa)" : "var(--text-secondary)" }}>
                    {modalStatus.health.training_active ? "YES" : "No"}
                  </span>
                </div>
                <div className="dash-row">
                  <span className="dash-row-label">Inference Loaded</span>
                  <span className="dash-row-value">{modalStatus.health.inference_loaded ? "Yes" : "No"}</span>
                </div>
              </div>
            )}
            {modalStatus?.health_error && (
              <div className="dash-card" style={{ marginBottom: "8px", fontSize: "12px", color: "var(--text-secondary)" }}>
                Health endpoint: {modalStatus.health_error}
              </div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: "8px" }}>
              {KERNEL_SPECIALIZATIONS.map((spec) => {
                const adapter: ModalAdapterInfo | undefined =
                  modalStatus?.adapters?.adapters?.[spec] ?? undefined;
                const trained = adapter?.exists ?? false;
                const meta = adapter?.training_meta;
                return (
                  <div
                    key={spec}
                    className="dash-card"
                    style={{
                      padding: "10px 14px",
                      borderLeft: `3px solid ${trained ? "var(--alive)" : "var(--border)"}`,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                      <span style={{ fontWeight: 600, fontSize: "13px", textTransform: "capitalize" }}>
                        {spec}
                      </span>
                      <span
                        style={{
                          fontSize: "10px",
                          fontWeight: 600,
                          padding: "2px 6px",
                          borderRadius: "4px",
                          background: trained ? "var(--alive)" : "var(--surface-3)",
                          color: trained ? "white" : "var(--text-secondary)",
                        }}
                      >
                        {trained ? "TRAINED" : "UNTRAINED"}
                      </span>
                    </div>
                    {meta && (
                      <div style={{ fontSize: "11px", fontFamily: "var(--mono)", color: "var(--text-secondary)", lineHeight: 1.5 }}>
                        {(meta.loss ?? meta.train_loss) != null && <div>loss: {Number(meta.loss ?? meta.train_loss).toFixed(4)}</div>}
                        {meta.epochs != null && <div>epochs: {meta.epochs}</div>}
                        {(meta.samples ?? meta.train_samples) != null && <div>samples: {meta.samples ?? meta.train_samples}</div>}
                        {(meta.date ?? meta.trained_at) && <div>{meta.date ?? meta.trained_at}</div>}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
            {modalStatus?.status_error && (
              <div style={{ marginTop: "6px", fontSize: "12px", color: "var(--text-secondary)" }}>
                Status endpoint: {modalStatus.status_error}
              </div>
            )}
          </>
        )}
      </div>

      {/* Volume Status */}
      <div className="dash-section">
        <div className="dash-section-title">Volume Status</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Training Directory</span>
            <span className="dash-row-value">
              {stats?.training_dir ?? "/data/training"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Volume Mounted</span>
            <span className="dash-row-value">
              {stats?.dir_exists ? "Yes" : "No"}
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Harvest Pending Files</span>
            <span className="dash-row-value">{stats?.harvest_pending_files ?? 0}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Coordized Chunks</span>
            <span className="dash-row-value">{stats?.coordized_chunks ?? 0}</span>
          </div>
        </div>
      </div>

      {/* Modal Data Bridge — Sync Railway → Modal */}
      <div className="dash-section">
        <div className="dash-section-title">Modal Training Volume</div>
        <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
          Training data auto-pushes to Modal on upload. Use Sync to bulk-push all local data.
        </div>
        <div className="dash-card">
          <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
            <button
              onClick={handleSync}
              disabled={syncing}
              style={{
                background: "var(--gamma)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor: syncing ? "not-allowed" : "pointer",
                opacity: syncing ? 0.5 : 1,
                fontSize: "14px",
              }}
            >
              {syncing ? "Syncing..." : "Sync Training Data to Modal"}
            </button>
            <button
              onClick={fetchModalData}
              disabled={modalDataLoading}
              style={{
                background: "var(--surface-3)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "10px 16px",
                color: "var(--text)",
                fontWeight: 500,
                cursor: modalDataLoading ? "not-allowed" : "pointer",
                fontSize: "13px",
              }}
            >
              {modalDataLoading ? "Loading..." : "Refresh"}
            </button>
            {modalData && (
              <span style={{ fontSize: "13px", color: "var(--text-secondary)", fontFamily: "var(--mono)" }}>
                {modalData.total_files ?? 0} files &middot; {modalData.total_records ?? 0} records on Modal
              </span>
            )}
          </div>
          {syncResult && (
            <div
              style={{
                marginTop: "8px",
                fontSize: "13px",
                color: syncResult.status === "ok" ? "var(--alive)" : "var(--error)",
              }}
            >
              {syncResult.status === "ok"
                ? `Synced ${syncResult.files_synced ?? 0} files (${syncResult.total_records_pushed ?? 0} records) to Modal`
                : syncResult.error ?? "Sync failed"}
            </div>
          )}
        </div>

        {modalData?.files && modalData.files.length > 0 && (
          <div style={{ marginTop: "8px", display: "flex", flexDirection: "column", gap: "4px", maxHeight: "200px", overflowY: "auto" }}>
            {modalData.files.map((f) => (
              <div
                key={f.path}
                className="dash-card"
                style={{ padding: "6px 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}
              >
                <span style={{ fontFamily: "var(--mono)", fontSize: "12px", fontWeight: 600 }}>{f.path}</span>
                <span style={{ fontSize: "11px", color: "var(--text-secondary)", fontFamily: "var(--mono)" }}>
                  {f.records} records &middot; {f.size_kb}KB
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload Section */}
      <div className="dash-section">
        <div className="dash-section-title">Upload Documents</div>
        <div className="dash-card">
          <div
            style={{ display: "flex", flexDirection: "column", gap: "10px" }}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.md,.txt,.jsonl"
              onChange={(e) => {
                const selected = e.target.files;
                if (selected && selected.length > 0) {
                  setFiles(Array.from(selected));
                }
              }}
              style={{
                background: "var(--surface-3)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 12px",
                color: "var(--text)",
                fontSize: "13px",
              }}
            />

            {files.length > 1 && (
              <span
                style={{
                  fontSize: "12px",
                  color: "var(--text-secondary)",
                  fontFamily: "var(--mono)",
                }}
              >
                {files.length} files selected:{" "}
                {files.map((f) => f.name).join(", ")}
              </span>
            )}

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                style={selectStyle}
              >
                <option value="general">General</option>
                <option value="doctrine">Doctrine</option>
                <option value="curriculum">Curriculum</option>
                <option value="research">Research</option>
              </select>

              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                style={selectStyle}
              >
                {PROCESSING_MODES.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>

              <select
                value={e8Prim}
                onChange={(e) => setE8Prim(e.target.value)}
                style={selectStyle}
              >
                <option value="">E8 Auto-detect</option>
                {E8_PRIMITIVES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={handleUpload}
              disabled={files.length === 0 || uploading}
              style={{
                background: "var(--accent)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor:
                  files.length === 0 || uploading ? "not-allowed" : "pointer",
                opacity: files.length === 0 || uploading ? 0.5 : 1,
                fontSize: "14px",
                alignSelf: "flex-start",
              }}
            >
              {uploading
                ? `Uploading ${activeCount} of ${jobs.length}...`
                : files.length > 1
                  ? `Upload & Process ${files.length} Files`
                  : "Upload & Process"}
            </button>
          </div>

          {/* Bulk progress summary */}
          {jobs.length > 1 && (
            <div
              style={{
                marginTop: "10px",
                padding: "8px 14px",
                background: "var(--surface-3)",
                borderRadius: "6px",
                fontSize: "12px",
                fontFamily: "var(--mono)",
                color: "var(--text-secondary)",
              }}
            >
              {completedCount}/{jobs.length} complete
              {errorCount > 0 && (
                <span style={{ color: "var(--error)" }}>
                  {" "}
                  | {errorCount} failed
                </span>
              )}
              {activeCount > 0 && ` | ${activeCount} processing`}
            </div>
          )}

          {/* Per-file results */}
          {jobs.length > 0 && (
            <div
              style={{
                marginTop: "8px",
                display: "flex",
                flexDirection: "column",
                gap: "4px",
                maxHeight: "300px",
                overflowY: "auto",
              }}
            >
              {jobs.map((job, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: "6px 12px",
                    background: "var(--surface-3)",
                    borderRadius: "4px",
                    fontFamily: "var(--mono)",
                    fontSize: "12px",
                    color:
                      job.status === "error"
                        ? "var(--error)"
                        : job.status === "done"
                          ? "var(--alive)"
                          : job.status === "queued"
                            ? "var(--text-secondary)"
                            : "var(--accent)",
                    borderLeft: `3px solid ${job.status === "error"
                      ? "var(--error)"
                      : job.status === "done"
                        ? "var(--alive)"
                        : job.status === "queued"
                          ? "var(--border)"
                          : "var(--accent)"
                      }`,
                  }}
                >
                  {job.status === "queued" && `${job.file.name}: Queued`}
                  {job.status === "uploading" &&
                    `${job.file.name}: Uploading...`}
                  {job.status === "processing" &&
                    `${job.file.name}: Processing...`}
                  {job.status === "error" &&
                    `${job.file.name}: ${job.result?.error ?? job.result?.errors?.[0] ?? "Failed"}`}
                  {job.status === "done" &&
                    job.result &&
                    `${job.result.filename}: ${job.result.chunks_written} chunks, ${job.result.enriched} enriched, ${job.result.qa_pairs ?? 0} Q&A (${(job.result.processing_time_s ?? 0).toFixed(1)}s)`}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Kernel Training — QLoRA on Modal GPU */}
      <div className="dash-section">
        <div className="dash-section-title">Kernel Training (QLoRA on Modal GPU)</div>
        <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
          Trains per-kernel QLoRA adapters on Qwen3.5-35B-A3B via Modal serverless GPU.
          Each kernel gets its own adapter shaped by E8 training data.
        </div>
        <div className="dash-card">
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
            <select
              value={trainTarget}
              onChange={(e) => setTrainTarget(e.target.value)}
              style={selectStyle}
            >
              <option value="all">All Kernels (consciousness order)</option>
              {KERNEL_SPECIALIZATIONS.map((k) => (
                <option key={k} value={k}>{k.charAt(0).toUpperCase() + k.slice(1)}</option>
              ))}
            </select>
            <button
              onClick={async () => {
                setTriggeringTraining(true);
                setTrainingResult(null);
                try {
                  const resp = await fetch(API.trainingTrigger, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ specialization: trainTarget }),
                  });
                  if (!resp.ok) {
                    const text = await resp.text();
                    setTrainingResult({ status: "error", error: `Server error ${resp.status}: ${text.slice(0, 200)}` });
                    return;
                  }
                  const data = await resp.json();
                  setTrainingResult(data);
                } catch (err) {
                  setTrainingResult({ status: "error", error: err instanceof Error ? err.message : String(err) });
                } finally {
                  setTriggeringTraining(false);
                }
              }}
              disabled={triggeringTraining}
              style={{
                background: "var(--kappa)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor: triggeringTraining ? "not-allowed" : "pointer",
                opacity: triggeringTraining ? 0.5 : 1,
                fontSize: "14px",
              }}
            >
              {triggeringTraining
                ? "Starting..."
                : trainTarget === "all"
                  ? "Start QLoRA Training (All)"
                  : `Train ${trainTarget.charAt(0).toUpperCase() + trainTarget.slice(1)} Kernel`}
            </button>
          </div>
          {trainingResult && (
            <div
              style={{
                marginTop: "8px",
                fontSize: "13px",
                color: trainingResult.status === "triggered" ? "var(--alive)" : "var(--error)",
              }}
            >
              {trainingResult.status === "triggered"
                ? "Training triggered on Modal GPU"
                : trainingResult.error ?? "Training trigger failed"}
            </div>
          )}
        </div>
      </div>

      {/* External Fine-Tuning — OpenAI JSONL Export */}
      <div className="dash-section">
        <div className="dash-section-title">External LLM Fine-Tuning (OpenAI / GPT-4.1)</div>
        <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
          Export training data as OpenAI-compatible JSONL for external LLM fine-tuning (GPT-4.1, etc.).
          This is <strong>separate from kernel QLoRA training</strong> — it produces data for third-party model providers.
        </div>
        <div className="dash-card">
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <button
              onClick={handleExport}
              disabled={exporting}
              style={{
                background: "var(--phi)",
                border: "none",
                borderRadius: "var(--radius-sm)",
                padding: "10px 20px",
                color: "white",
                fontWeight: 600,
                cursor: exporting ? "not-allowed" : "pointer",
                opacity: exporting ? 0.5 : 1,
                fontSize: "14px",
              }}
            >
              {exporting ? "Exporting..." : "Export JSONL (GPT-4.1 Format)"}
            </button>
            {exportData && (
              <span
                style={{ fontSize: "13px", color: "var(--text-secondary)" }}
              >
                {exportData.count >= 0
                  ? `${exportData.count} training examples exported`
                  : "Export failed"}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Pipeline Info */}
      <div className="dash-section">
        <div className="dash-section-title">Pipeline Architecture</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Extraction</span>
            <span className="dash-row-value">PyMuPDF (in-process)</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Chunking</span>
            <span className="dash-row-value">
              Semantic (512 tokens, paragraph boundaries)
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Coordization</span>
            <span className="dash-row-value">
              CoordizerV2 &rarr; 64D on &Delta;&sup6;&sup3; (
              {stats?.coordizer_active ? "active" : "inactive"})
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Enrichment</span>
            <span className="dash-row-value">
              xAI Responses API ({stats ? "active" : "checking..."}) / Ollama
              fallback
            </span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Storage</span>
            <span className="dash-row-value">JSONL on Railway volume</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">E8 Tagging</span>
            <span className="dash-row-value">
              PER MEM ACT PRD ETH META HRT REL MIX
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  background: "var(--surface-3)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-sm)",
  padding: "8px 12px",
  color: "var(--text)",
  fontSize: "13px",
  fontFamily: "inherit",
};
