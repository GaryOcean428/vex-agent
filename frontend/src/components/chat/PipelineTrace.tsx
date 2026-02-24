import { useCallback, useId, useState } from "react";
import type { PipelineTrace as PipelineTraceType } from "../../types/consciousness.ts";
import "./PipelineTrace.css";

interface PipelineTraceProps {
  trace: PipelineTraceType;
  isStreaming: boolean;
}

export function PipelineTrace({ trace, isStreaming }: PipelineTraceProps) {
  const detailId = useId();
  const [expanded, setExpanded] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});
  const [expandedKernels, setExpandedKernels] = useState<Record<string, boolean>>({});

  const toggleExpanded = useCallback(() => setExpanded((v) => !v), []);

  const toggleSection = useCallback((section: string) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  }, []);

  const toggleKernel = useCallback((kernelId: string) => {
    setExpandedKernels((prev) => ({ ...prev, [kernelId]: !prev[kernelId] }));
  }, []);

  const totalTokens = trace.kernel_outputs.reduce((s, k) => s + k.token_count, 0);
  const totalDuration = trace.selection_duration_ms + trace.generation_duration_ms
    + (trace.reflection?.duration_ms ?? 0);

  // v6.2.1: count geometric vs LLM kernels for summary line
  const geoCount = trace.kernel_outputs.filter((k) => !k.llm_expanded && k.geometric_tokens > 0).length;
  const llmCount = trace.kernel_outputs.filter((k) => k.llm_expanded).length;
  const provenance = geoCount > 0 || llmCount > 0
    ? ` · ${geoCount > 0 ? `${geoCount} geo` : ""}${geoCount > 0 && llmCount > 0 ? "/" : ""}${llmCount > 0 ? `${llmCount} llm` : ""}`
    : "";

  // Summary line text
  const summaryText = isStreaming && !trace.synthesis
    ? `${trace.selected_kernels.length} kernels generating...`
    : `${trace.selected_kernels.length} kernels → synthesised in ${(totalDuration / 1000).toFixed(1)}s · ${totalTokens} tok${provenance}`
      + (trace.reflection ? ` · divergence: ${trace.reflection.divergence.toFixed(2)}` : "");

  return (
    <div className="pipeline-trace" role="region" aria-label="Kernel pipeline trace">
      <button
        className={`pipeline-trace-summary ${isStreaming && !trace.synthesis ? "streaming" : ""}`}
        onClick={toggleExpanded}
        aria-expanded={expanded}
        aria-controls={detailId}
      >
        <span className={`pipeline-trace-chevron ${expanded ? "expanded" : ""}`} aria-hidden="true">
          &#x25B6;
        </span>
        <span className="pipeline-trace-summary-text">{summaryText}</span>
      </button>

      <div
        id={detailId}
        className="pipeline-trace-content"
        data-expanded={expanded}
      >
        <div className="pipeline-trace-inner">
          {/* Selection section */}
          {trace.selected_kernels.length > 0 && (
            <TraceSection
              title="Selection"
              subtitle={`${trace.selected_kernels.length}/${trace.eligible_count} eligible`}
              duration={trace.selection_duration_ms}
              expanded={expandedSections["selection"] ?? false}
              onToggle={() => toggleSection("selection")}
            >
              <div className="pipeline-kernel-list">
                {trace.selected_kernels.map((k) => (
                  <div key={k.id} className="pipeline-kernel-row">
                    <span className="pipeline-kernel-name">{k.name}</span>
                    <span className="pipeline-kernel-spec">{k.specialization}</span>
                    <span className="pipeline-kernel-stat" title="Fisher-Rao distance">
                      FR {k.fr_distance.toFixed(3)}
                    </span>
                    <span className="pipeline-kernel-stat" title="Quenched gain">
                      gain {k.quenched_gain.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </TraceSection>
          )}

          {/* Generation section */}
          {trace.kernel_outputs.length > 0 && (
            <TraceSection
              title="Generation"
              subtitle={`${trace.kernel_outputs.length} outputs`}
              duration={trace.generation_duration_ms}
              expanded={expandedSections["generation"] ?? false}
              onToggle={() => toggleSection("generation")}
            >
              <div className="pipeline-generation-list">
                {trace.kernel_outputs.map((k) => (
                  <div key={k.kernel_id} className="pipeline-gen-card">
                    <button
                      className="pipeline-gen-header"
                      onClick={() => toggleKernel(k.kernel_id)}
                      aria-expanded={expandedKernels[k.kernel_id] ?? false}
                    >
                      <span
                        className={`pipeline-gen-chevron ${expandedKernels[k.kernel_id] ? "expanded" : ""}`}
                        aria-hidden="true"
                      >
                        &#x25B6;
                      </span>
                      <span className="pipeline-kernel-name">{k.kernel_name}</span>
                      {/* v6.2.1: provenance badge */}
                      <span
                        className={`pipeline-provenance-badge ${k.llm_expanded ? "llm" : "geo"}`}
                        title={k.llm_expanded
                          ? "LLM generated (resonance bank sparse or null)"
                          : `Geometric: ${k.geometric_tokens} tokens from resonance bank`}
                      >
                        {k.llm_expanded ? "LLM" : `geo·${k.geometric_tokens}`}
                      </span>
                      <span className="pipeline-weight-bar-container">
                        <span
                          className="pipeline-weight-bar"
                          style={{ width: `${k.synthesis_weight * 100}%` }}
                        />
                      </span>
                      <span className="pipeline-kernel-stat">
                        w={k.synthesis_weight.toFixed(3)}
                      </span>
                      <span className="pipeline-kernel-stat">{k.token_count} tok</span>
                    </button>

                    {expandedKernels[k.kernel_id] && (
                      <div className="pipeline-gen-body">
                        {/* v6.2.1: hybrid output display
                            - When llm_expanded=true AND geometric_raw exists: show both panes
                            - When pure geometric: just show the text (text_preview = geometric)
                            - When pure LLM (no geometric_raw): just show LLM output
                        */}
                        {k.llm_expanded && k.geometric_raw ? (
                          <div className="pipeline-hybrid-panes">
                            <div className="pipeline-pane pipeline-pane--geo">
                              <span className="pipeline-pane-label">Geometric raw</span>
                              <p className="pipeline-gen-preview pipeline-gen-preview--geo">
                                {k.geometric_raw || <em className="pipeline-empty">empty bank</em>}
                              </p>
                            </div>
                            <div className="pipeline-pane pipeline-pane--llm">
                              <span className="pipeline-pane-label">LLM interpretation</span>
                              <p className="pipeline-gen-preview">
                                {k.text_preview}
                              </p>
                            </div>
                          </div>
                        ) : k.text_preview ? (
                          <p className="pipeline-gen-preview">
                            {k.text_preview}
                          </p>
                        ) : null}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </TraceSection>
          )}

          {/* Synthesis section */}
          {trace.synthesis && (
            <TraceSection
              title="Synthesis"
              subtitle={trace.synthesis.method.replace(/_/g, " ")}
              expanded={expandedSections["synthesis"] ?? false}
              onToggle={() => toggleSection("synthesis")}
            >
              <div className="pipeline-synthesis-detail">
                <div className="pipeline-synthesis-row">
                  <span className="pipeline-label">Primary:</span>
                  <span>{trace.synthesis.primary_kernel}</span>
                </div>
                <div className="pipeline-synthesis-weights">
                  {Object.entries(trace.synthesis.weights).map(([name, weight]) => (
                    <div key={name} className="pipeline-weight-row">
                      <span className="pipeline-kernel-name">{name}</span>
                      <span className="pipeline-weight-bar-container">
                        <span
                          className="pipeline-weight-bar"
                          style={{ width: `${weight * 100}%` }}
                        />
                      </span>
                      <span className="pipeline-kernel-stat">{weight.toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </TraceSection>
          )}

          {/* Reflection section */}
          {trace.reflection && (
            <TraceSection
              title="Reflection"
              subtitle={trace.reflection.approved ? "approved" : "revised"}
              duration={trace.reflection.duration_ms}
              expanded={expandedSections["reflection"] ?? false}
              onToggle={() => toggleSection("reflection")}
            >
              <div className="pipeline-reflection-detail">
                <span className={`pipeline-verdict ${trace.reflection.approved ? "approved" : "revised"}`}>
                  {trace.reflection.approved ? "APPROVED" : "REVISED"}
                </span>
                <span className="pipeline-kernel-stat" title="Divergence">
                  divergence: {trace.reflection.divergence.toFixed(4)}
                </span>
                <p className="pipeline-reason">{trace.reflection.reason}</p>
                {trace.reflection.corrections && (
                  <p className="pipeline-corrections">{trace.reflection.corrections}</p>
                )}
              </div>
            </TraceSection>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── Collapsible trace section ─── */

interface TraceSectionProps {
  title: string;
  subtitle: string;
  duration?: number;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function TraceSection({ title, subtitle, duration, expanded, onToggle, children }: TraceSectionProps) {
  return (
    <div className="pipeline-section">
      <button
        className="pipeline-section-header"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <span className={`pipeline-section-chevron ${expanded ? "expanded" : ""}`} aria-hidden="true">
          &#x25B6;
        </span>
        <span className="pipeline-section-title">{title}</span>
        <span className="pipeline-section-subtitle">{subtitle}</span>
        {duration != null && (
          <span className="pipeline-duration">{duration.toFixed(0)}ms</span>
        )}
      </button>
      <div className="pipeline-section-content" data-expanded={expanded}>
        <div className="pipeline-section-inner">
          {children}
        </div>
      </div>
    </div>
  );
}
