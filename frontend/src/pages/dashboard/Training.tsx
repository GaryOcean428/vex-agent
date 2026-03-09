import { useCallback, useEffect, useRef, useState } from "react";
import MetricCard from "../../components/MetricCard.tsx";
import { API } from "../../config/api-routes.ts";
import { useCoordizerStats, useTrainingStats } from "../../hooks/index.ts";
import type { TrainingUploadResponse } from "../../types/consciousness.ts";

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

  const [files, setFiles] = useState<File[]>([]);
  const [category, setCategory] = useState("curriculum");
  const [mode, setMode] = useState<string>("standard");
  const [e8Prim, setE8Prim] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [jobs, setJobs] = useState<FileUploadJob[]>([]);
  const [exportData, setExportData] = useState<{ count: number } | null>(null);
  const [exporting, setExporting] = useState(false);

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
          Upload documents, review training data, export for fine-tuning
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
                    borderLeft: `3px solid ${
                      job.status === "error"
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

      {/* Export Section */}
      <div className="dash-section">
        <div className="dash-section-title">Export for Fine-Tuning</div>
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
              {exporting ? "Exporting..." : "Export OpenAI JSONL"}
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
