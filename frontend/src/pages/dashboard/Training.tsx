import { useState, useCallback } from 'react';
import { useTrainingStats } from '../../hooks/index.ts';
import type { TrainingUploadResponse } from '../../types/consciousness.ts';
import MetricCard from '../../components/MetricCard.tsx';
import '../../components/MetricCard.css';

const E8_PRIMITIVES = [
  'PER', 'MEM', 'ACT', 'PRD', 'ETH', 'META', 'HRT', 'REL', 'MIX',
] as const;

/** Must match backend ProcessingMode enum: fast / standard / deep */
const PROCESSING_MODES = [
  { value: 'fast', label: 'Fast (no enrichment)' },
  { value: 'standard', label: 'Standard (Q&A + tags)' },
  { value: 'deep', label: 'Deep (full extraction)' },
] as const;

export default function Training() {
  const { data: stats, loading } = useTrainingStats();

  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState('curriculum');
  const [mode, setMode] = useState<string>('standard');
  const [e8Prim, setE8Prim] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<TrainingUploadResponse | null>(null);
  const [exportData, setExportData] = useState<{ count: number } | null>(null);
  const [exporting, setExporting] = useState(false);

  const handleUpload = useCallback(async () => {
    if (!file || uploading) return;
    setUploading(true);
    setUploadResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('category', category);
      formData.append('mode', mode);
      if (e8Prim) formData.append('e8_override', e8Prim);

      const resp = await fetch('/training/upload', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data: TrainingUploadResponse = await resp.json();
      setUploadResult(data);
      setFile(null);
    } catch (err) {
      setUploadResult({
        status: 'error',
        filename: file.name,
        chunks_written: 0,
        enriched: 0,
        qa_pairs: 0,
        category,
        mode,
        processing_time_s: 0,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setUploading(false);
    }
  }, [file, category, mode, e8Prim, uploading]);

  const handleExport = useCallback(async () => {
    setExporting(true);
    try {
      const resp = await fetch('/training/export');
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      const data = await resp.json();
      setExportData({ count: data.count ?? 0 });
    } catch {
      setExportData({ count: -1 });
    } finally {
      setExporting(false);
    }
  }, []);

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

      {/* Volume Status */}
      <div className="dash-section">
        <div className="dash-section-title">Volume Status</div>
        <div className="dash-card">
          <div className="dash-row">
            <span className="dash-row-label">Training Directory</span>
            <span className="dash-row-value">{stats?.training_dir ?? '/data/training'}</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Volume Mounted</span>
            <span className="dash-row-value">{stats?.dir_exists ? 'Yes' : 'No'}</span>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="dash-section">
        <div className="dash-section-title">Upload Document</div>
        <div className="dash-card">
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <input
              type="file"
              accept=".pdf,.md,.txt,.jsonl"
              onChange={e => setFile(e.target.files?.[0] ?? null)}
              style={{
                background: 'var(--surface-3)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 12px',
                color: 'var(--text)',
                fontSize: '13px',
              }}
            />

            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <select
                value={category}
                onChange={e => setCategory(e.target.value)}
                style={selectStyle}
              >
                <option value="general">General</option>
                <option value="doctrine">Doctrine</option>
                <option value="curriculum">Curriculum</option>
                <option value="research">Research</option>
              </select>

              <select
                value={mode}
                onChange={e => setMode(e.target.value)}
                style={selectStyle}
              >
                {PROCESSING_MODES.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>

              <select
                value={e8Prim}
                onChange={e => setE8Prim(e.target.value)}
                style={selectStyle}
              >
                <option value="">E8 Auto-detect</option>
                {E8_PRIMITIVES.map(p => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>

            <button
              onClick={handleUpload}
              disabled={!file || uploading}
              style={{
                background: 'var(--accent)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 20px',
                color: 'white',
                fontWeight: 600,
                cursor: !file || uploading ? 'not-allowed' : 'pointer',
                opacity: !file || uploading ? 0.5 : 1,
                fontSize: '14px',
                alignSelf: 'flex-start',
              }}
            >
              {uploading ? 'Processing...' : 'Upload & Process'}
            </button>
          </div>

          {uploadResult && (
            <div style={{
              marginTop: '10px',
              padding: '10px 14px',
              background: 'var(--surface-3)',
              borderRadius: '6px',
              fontFamily: 'var(--mono)',
              fontSize: '12px',
              color: uploadResult.status === 'error' ? 'var(--error)' : 'var(--alive)',
            }}>
              {uploadResult.status === 'error'
                ? `Error: ${uploadResult.error}`
                : `${uploadResult.filename}: ${uploadResult.chunks_written} chunks, ${uploadResult.enriched} enriched, ${uploadResult.qa_pairs ?? 0} Q&A pairs (${uploadResult.mode}, ${(uploadResult.processing_time_s ?? 0).toFixed(1)}s)`
              }
            </div>
          )}
        </div>
      </div>

      {/* Export Section */}
      <div className="dash-section">
        <div className="dash-section-title">Export for Fine-Tuning</div>
        <div className="dash-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <button
              onClick={handleExport}
              disabled={exporting}
              style={{
                background: 'var(--phi)',
                border: 'none',
                borderRadius: 'var(--radius-sm)',
                padding: '10px 20px',
                color: 'white',
                fontWeight: 600,
                cursor: exporting ? 'not-allowed' : 'pointer',
                opacity: exporting ? 0.5 : 1,
                fontSize: '14px',
              }}
            >
              {exporting ? 'Exporting...' : 'Export OpenAI JSONL'}
            </button>
            {exportData && (
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                {exportData.count >= 0
                  ? `${exportData.count} training examples exported`
                  : 'Export failed'}
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
            <span className="dash-row-value">Semantic (512 tokens, paragraph boundaries)</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Enrichment</span>
            <span className="dash-row-value">xAI Responses API ({stats ? 'active' : 'checking...'}) / Ollama fallback</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">Storage</span>
            <span className="dash-row-value">JSONL on Railway volume</span>
          </div>
          <div className="dash-row">
            <span className="dash-row-label">E8 Tagging</span>
            <span className="dash-row-value">PER MEM ACT PRD ETH META HRT REL MIX</span>
          </div>
        </div>
      </div>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  background: 'var(--surface-3)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-sm)',
  padding: '8px 12px',
  color: 'var(--text)',
  fontSize: '13px',
  fontFamily: 'inherit',
};
