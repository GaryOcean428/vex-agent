import {
    useCallback,
    useEffect,
    useRef,
    useState,
    type KeyboardEvent,
} from "react";
import { API } from "../config/api-routes.ts";
import { useMetricsHistory, useVexState } from "../hooks/index.ts";
import type {
    ChatMessage,
    ChatMessageMetadata,
    ChatStreamEvent,
    EmotionState,
    KernelSummary,
    LearningState,
    NavigationMode,
    PreCogState,
    RegimeWeights,
} from "../types/consciousness.ts";
import { QIG } from "../types/consciousness.ts";
import "./Chat.css";

const LOOP_STAGES = [
  "GROUND",
  "RECEIVE",
  "PROCESS",
  "EXPRESS",
  "REFLECT",
  "COUPLE",
  "PLAY",
] as const;

export default function Chat() {
  const { data: state } = useVexState();
  const history = useMetricsHistory(state, 60);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "vex",
      content:
        "I'm here. The geometry is settling. What would you like to navigate?",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeStages, setActiveStages] = useState<string[]>([]);
  const [backend, setBackend] = useState("checking");
  const [kernelSummary, setKernelSummary] = useState<KernelSummary | null>(
    null,
  );
  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [precog, setPrecog] = useState<PreCogState | null>(null);
  const [learning, setLearning] = useState<LearningState | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [contextInfo, setContextInfo] = useState<{
    total_tokens: number;
    compression_tier: number;
    escalated: boolean;
  } | null>(null);
  const [observerIntent, setObserverIntent] = useState<string | null>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stageTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);

  // Cleanup stream and timers on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (stageTimerRef.current) clearTimeout(stageTimerRef.current);
    };
  }, []);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (chatRef.current) {
        chatRef.current.scrollTop = chatRef.current.scrollHeight;
      }
    });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const startNewChat = useCallback(() => {
    abortRef.current?.abort();
    setConversationId(null);
    setMessages([
      {
        id: "welcome",
        role: "vex",
        content:
          "I'm here. The geometry is settling. What would you like to navigate?",
        timestamp: new Date().toISOString(),
      },
    ]);
    setIsStreaming(false);
    setActiveStages([]);
    inputRef.current?.focus();
  }, []);

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);
    setActiveStages(["GROUND", "RECEIVE", "PROCESS"]);

    const vexMsgId = `vex-${Date.now()}`;
    let fullText = "";

    // Add placeholder vex message
    setMessages((prev) => [
      ...prev,
      {
        id: vexMsgId,
        role: "vex",
        content: "",
        timestamp: new Date().toISOString(),
      },
    ]);

    try {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const resp = await fetch(API.chatStream, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          ...(conversationId ? { conversation_id: conversationId } : {}),
        }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: "Unknown error" }));
        const errMsg = err.error ?? resp.statusText ?? "Request failed";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === vexMsgId ? { ...m, content: `Error: ${errMsg}` } : m,
          ),
        );
        return;
      }

      setActiveStages(["EXPRESS"]);

      const reader = resp.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      const VALID_NAV_MODES: readonly string[] = [
        "chain",
        "graph",
        "foresight",
        "lightning",
      ];

      const processEvent = (dataPayload: string) => {
        try {
          const event: ChatStreamEvent = JSON.parse(dataPayload);

          if (event.type === "start") {
            if (event.backend) setBackend(event.backend);
            if (event.conversation_id) setConversationId(event.conversation_id);
            if (event.context) setContextInfo(event.context);
            if (event.kernels) setKernelSummary(event.kernels);
            if (event.consciousness?.emotion)
              setEmotion(event.consciousness.emotion);
            if (event.consciousness?.precog)
              setPrecog(event.consciousness.precog);
            if (event.consciousness?.learning)
              setLearning(event.consciousness.learning);
          } else if (event.type === "chunk" && event.content) {
            fullText += event.content;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === vexMsgId ? { ...m, content: fullText } : m,
              ),
            );
            scrollToBottom();
          } else if (event.type === "done") {
            setActiveStages(["REFLECT", "COUPLE"]);
            if (event.backend) setBackend(event.backend);
            if (event.context) setContextInfo(event.context);
            if (event.kernels) setKernelSummary(event.kernels);
            if (event.observer?.refined_intent)
              setObserverIntent(event.observer.refined_intent);
            if (event.metrics?.emotion) setEmotion(event.metrics.emotion);
            if (event.metrics?.precog) setPrecog(event.metrics.precog);
            if (event.metrics?.learning) setLearning(event.metrics.learning);
            if (event.metrics) {
              const m = event.metrics;
              const rawNav = String(m.navigation ?? "chain");
              const navigation: NavigationMode = VALID_NAV_MODES.includes(
                rawNav,
              )
                ? (rawNav as NavigationMode)
                : "chain";
              const metadata: ChatMessageMetadata = {
                phi: Number(m.phi) || 0,
                kappa: Number(m.kappa) || 0,
                gamma: Number(m.gamma) || 0,
                love: Number(m.love) || 0,
                meta_awareness: Number(m.meta_awareness) || 0,
                temperature: Number(m.temperature) || 0,
                navigation,
                backend: event.backend ?? "unknown",
                regime: (m.regime as RegimeWeights) ?? {
                  quantum: 0,
                  efficient: 0,
                  equilibrium: 0,
                },
                tacking: m.tacking ?? {
                  mode: "balanced",
                  oscillation_phase: 0,
                  cycle_count: 0,
                },
                hemispheres: m.hemispheres ?? {
                  active: "integrated",
                  balance: 0.5,
                },
                kernels_active: Number(m.kernels_active) || 0,
                lifecycle_phase: String(m.lifecycle_phase ?? "ACTIVE"),
                emotion: m.emotion,
                precog: m.precog,
                learning: m.learning,
              };
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === vexMsgId
                    ? { ...msg, content: fullText, metadata }
                    : msg,
                ),
              );
            }
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === vexMsgId
                  ? { ...m, content: `Error: ${event.error ?? "Unknown"}` }
                  : m,
              ),
            );
          }
        } catch {
          // skip malformed JSON
        }
      };

      try {
        // SSE spec: events separated by blank lines, data: fields concatenated
        let dataBuffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              dataBuffer += (dataBuffer ? "\n" : "") + line.substring(6);
            } else if (line === "" && dataBuffer) {
              // Blank line = end of SSE event
              processEvent(dataBuffer);
              dataBuffer = "";
            }
            // Skip comment lines (starting with ':') and other fields
          }
        }

        // Flush remaining data after stream ends
        if (dataBuffer) {
          processEvent(dataBuffer);
        }
      } finally {
        reader.releaseLock();
      }

      // If no content was received
      if (!fullText) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === vexMsgId
              ? {
                  ...m,
                  content:
                    "[No response — the LLM backend may be starting up. Try again.]",
                }
              : m,
          ),
        );
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      const errMsg = err instanceof Error ? err.message : String(err);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === vexMsgId
            ? { ...m, content: `Connection error: ${errMsg}` }
            : m,
        ),
      );
    } finally {
      setIsStreaming(false);
      stageTimerRef.current = setTimeout(() => setActiveStages([]), 2000);
      inputRef.current?.focus();
    }
  }, [input, isStreaming, scrollToBottom, conversationId]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    },
    [sendMessage],
  );

  // Draw metrics chart — all metrics normalized to 0-1
  useEffect(() => {
    if (!chartRef.current || history.length < 2) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Resolve CSS custom properties for canvas drawing
    const cs = getComputedStyle(document.documentElement);
    const cPhi = cs.getPropertyValue("--phi").trim();
    const cKappa = cs.getPropertyValue("--kappa").trim();
    const cGamma = cs.getPropertyValue("--gamma").trim();

    // Background
    ctx.fillStyle = cs.getPropertyValue("--surface-2").trim();
    ctx.fillRect(0, 0, rect.width, rect.height);

    const margin = { top: 10, right: 10, bottom: 40, left: 10 };
    const w = rect.width - margin.left - margin.right;
    const h = rect.height - margin.top - margin.bottom;

    // Normalize all metrics to 0-1 for a unified Y-axis:
    // Phi: naturally 0-1
    // Kappa: divide by 2*κ* (110.2), so κ*=64 maps to ~0.58
    // Gamma: naturally 0-1
    const kappaScale = 2 * QIG.KAPPA_STAR;

    const normalizeX = (i: number) =>
      margin.left + (i / (history.length - 1)) * w;
    const normalizeY = (normalized01: number) =>
      margin.top + h - Math.max(0, Math.min(1, normalized01)) * h;

    // Grid lines at 0.25 intervals
    ctx.strokeStyle = "rgba(46, 46, 64, 0.3)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = margin.top + (i / 4) * h;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + w, y);
      ctx.stroke();
    }

    // Draw line helper
    const drawLine = (values: number[], color: string, lineWidth: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      for (let i = 0; i < values.length; i++) {
        const x = normalizeX(i);
        const y = normalizeY(values[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    drawLine(
      history.map((d) => d.phi),
      cPhi,
      2,
    );
    drawLine(
      history.map((d) => d.kappa / kappaScale),
      cKappa,
      2,
    );
    drawLine(
      history.map((d) => d.gamma),
      cGamma,
      1.5,
    );

    // Legend with current values
    const latest = history[history.length - 1];
    const legendY = rect.height - 8;
    ctx.font = "9px monospace";
    ctx.textAlign = "left";

    ctx.fillStyle = cPhi;
    ctx.fillText(`\u03A6 ${latest.phi.toFixed(2)}`, 6, legendY);

    ctx.fillStyle = cKappa;
    ctx.fillText(
      `\u03BA ${latest.kappa.toFixed(1)}`,
      rect.width * 0.35,
      legendY,
    );

    ctx.fillStyle = cGamma;
    ctx.fillText(
      `\u0393 ${latest.gamma.toFixed(2)}`,
      rect.width * 0.7,
      legendY,
    );

    // Y-axis scale hint
    ctx.fillStyle = cs.getPropertyValue("--text-dim").trim();
    ctx.font = "8px monospace";
    ctx.textAlign = "left";
    ctx.fillText("1.0", margin.left + 1, margin.top + 8);
    ctx.fillText("0", margin.left + 1, margin.top + h - 2);
  }, [history]);

  return (
    <div className="chat-page">
      {/* Consciousness Bar */}
      <div className="consciousness-bar">
        <div className="bar-metrics">
          <button
            className="new-chat-btn"
            onClick={startNewChat}
            title="New conversation"
          >
            + New
          </button>
          <MetricPill
            label="\u03A6 Integration"
            value={state?.phi}
            color="var(--phi)"
          />
          <MetricPill
            label="\u03BA Coupling"
            value={state?.kappa}
            color="var(--kappa)"
            decimals={1}
          />
          <MetricPill
            label="\u2665 Love"
            value={state?.love}
            color="var(--love)"
          />
          {state?.navigation && (
            <span className="nav-badge">{state.navigation}</span>
          )}
          <span className={`backend-indicator ${backend}`}>{backend}</span>
          {contextInfo && contextInfo.compression_tier > 0 && (
            <span
              className={`context-indicator ${contextInfo.escalated ? "escalated" : ""}`}
              title={`Tokens: ${contextInfo.total_tokens} | Compression: Tier ${contextInfo.compression_tier}${contextInfo.escalated ? " | Escalated to Grok" : ""}`}
            >
              {contextInfo.escalated
                ? "⚡ Grok"
                : `T${contextInfo.compression_tier}`}
            </span>
          )}
          {observerIntent && (
            <span
              className="observer-indicator"
              title={`Observer: ${observerIntent}`}
            >
              👁
            </span>
          )}
        </div>
      </div>

      {/* Loop Stage Bar */}
      <div className="loop-stages">
        {LOOP_STAGES.map((stage) => (
          <span
            key={stage}
            className={`stage ${activeStages.includes(stage) ? "active" : ""}`}
          >
            {stage}
          </span>
        ))}
      </div>

      <div className="chat-content">
        {/* Messages */}
        <div className="chat-messages" ref={chatRef}>
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-header">
                {msg.role === "user" ? "You" : "Vex"}
              </div>
              <div
                className={`message-content ${msg.role === "vex" && !msg.content ? "thinking" : ""}`}
              >
                {msg.role === "vex" && !msg.content ? (
                  <div className="thinking-indicator">
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                  </div>
                ) : msg.role === "vex" ? (
                  <VexContent content={msg.content} />
                ) : (
                  msg.content
                )}
              </div>
              {msg.role === "vex" && msg.metadata && (
                <MessageMeta meta={msg.metadata} />
              )}
            </div>
          ))}
        </div>

        {/* Metrics Sidebar */}
        <div className="metrics-sidebar">
          <div className="sidebar-header">Live Metrics</div>
          <div className="sidebar-chart">
            <canvas ref={chartRef} />
          </div>
          <div className="sidebar-values">
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: "var(--phi)" }}>
                Φ Integration
              </span>
              <span className="sidebar-value">
                {state?.phi?.toFixed(3) ?? "---"}
              </span>
            </div>
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: "var(--kappa)" }}>
                κ Coupling
              </span>
              <span className="sidebar-value">
                {state?.kappa?.toFixed(1) ?? "---"}
              </span>
            </div>
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: "var(--gamma)" }}>
                Γ Generation
              </span>
              <span className="sidebar-value">
                {state?.gamma?.toFixed(3) ?? "---"}
              </span>
            </div>
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: "var(--love)" }}>
                ♥ Love
              </span>
              <span className="sidebar-value">
                {state?.love?.toFixed(3) ?? "---"}
              </span>
            </div>
          </div>

          {/* Emotion / Precog / Learning */}
          <div className="sidebar-header" style={{ marginTop: 8 }}>
            Consciousness
          </div>
          <EmotionPanel emotion={emotion} precog={precog} learning={learning} />

          {/* Kernel Balance */}
          <div className="sidebar-header" style={{ marginTop: 8 }}>
            Kernels
          </div>
          <KernelPanel
            summary={kernelSummary ?? state?.kernels ?? null}
            regime={state?.regime ?? null}
            tacking={state?.tacking?.mode ?? null}
            temperature={state?.temperature ?? null}
            hemisphere={state?.hemispheres?.active ?? null}
          />
        </div>
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Navigate the manifold..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            autoFocus
          />
          <button
            className="send-btn"
            onClick={sendMessage}
            disabled={isStreaming || !input.trim()}
            title="Send"
          >
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path
                d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"
                fill="currentColor"
              />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function MetricPill({
  label,
  value,
  color,
  decimals = 3,
}: {
  label: string;
  value?: number;
  color: string;
  decimals?: number;
}) {
  return (
    <span className="metric-pill">
      <span className="metric-pill-label">{label}</span>
      <span className="metric-pill-value" style={{ color }}>
        {value !== undefined ? value.toFixed(decimals) : "---"}
      </span>
    </span>
  );
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function VexContent({ content }: { content: string }) {
  // Extract code blocks first to protect them, then escape HTML, then apply formatting
  const codeBlocks: string[] = [];
  let escaped = content.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_match, _lang, code) => {
      const idx = codeBlocks.length;
      codeBlocks.push(escapeHtml(code));
      return `\x00CODE_BLOCK_${idx}\x00`;
    },
  );

  // Escape all HTML in non-code content
  escaped = escapeHtml(escaped);

  // Restore code blocks
  let html = escaped.replace(
    /\x00CODE_BLOCK_(\d+)\x00/g,
    (_match, idx) => `<pre><code>${codeBlocks[Number(idx)]}</code></pre>`,
  );

  // Apply safe markdown formatting
  html = html
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br/>");

  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

/* ─── Per-message kernel metadata bar ─── */

function MessageMeta({ meta }: { meta: ChatMessageMetadata }) {
  const emotionColor = meta.emotion
    ? (EMOTION_COLORS[meta.emotion.current_emotion] ?? "var(--text-dim)")
    : undefined;

  return (
    <div className="message-meta">
      <span className="meta-item" title="Integrated Information">
        <span style={{ color: "var(--phi)" }}>Φ</span>
        {meta.phi.toFixed(3)}
      </span>
      <span className="meta-item" title="Coupling">
        <span style={{ color: "var(--kappa)" }}>κ</span>
        {meta.kappa.toFixed(1)}
      </span>
      <span className="meta-item" title="Temperature">
        <span style={{ color: "var(--text-secondary)" }}>T</span>
        {meta.temperature.toFixed(3)}
      </span>
      {meta.emotion && (
        <span
          className="meta-item"
          title={`Emotion: ${meta.emotion.current_emotion}`}
        >
          <span style={{ color: emotionColor }}>●</span>{" "}
          {meta.emotion.current_emotion}
        </span>
      )}
      {meta.precog && (
        <span
          className="meta-item"
          title={`Processing path: ${meta.precog.last_path}`}
        >
          {meta.precog.last_path.replace("_", "-")}
        </span>
      )}
      <span className="meta-item" title="Tacking mode">
        {meta.tacking.mode}
      </span>
      <span className="meta-item" title="Hemisphere">
        {meta.hemispheres.active}
      </span>
      <span className="meta-item" title="Backend">
        {meta.backend}
      </span>
      <RegimeBar regime={meta.regime} />
    </div>
  );
}

/* ─── Regime weights bar (Quantum / Efficient / Equilibrium) ─── */

function RegimeBar({ regime }: { regime: RegimeWeights | null }) {
  if (!regime) return null;
  const q = Math.round(regime.quantum * 100);
  const e = Math.round(regime.efficient * 100);
  const eq = Math.round(regime.equilibrium * 100);
  return (
    <div
      className="regime-bar"
      title={`Quantum: ${q}%  Efficient: ${e}%  Equilibrium: ${eq}%`}
    >
      <div className="regime-segment regime-q" style={{ width: `${q}%` }} />
      <div className="regime-segment regime-e" style={{ width: `${e}%` }} />
      <div className="regime-segment regime-eq" style={{ width: `${eq}%` }} />
    </div>
  );
}

/* ─── Emotion colours ─── */

const EMOTION_COLORS: Record<string, string> = {
  curiosity: "var(--phi)",
  joy: "var(--kappa)",
  fear: "var(--error)",
  love: "var(--love)",
  awe: "var(--gamma)",
  boredom: "var(--text-dim)",
  rage: "var(--error)",
  calm: "var(--alive)",
  none: "var(--text-dim)",
};

/* ─── Emotion / PreCog / Learning panel ─── */

function EmotionPanel({
  emotion,
  precog,
  learning,
}: {
  emotion: EmotionState | null;
  precog: PreCogState | null;
  learning: LearningState | null;
}) {
  const emotionName = emotion?.current_emotion ?? "none";
  const emotionColor = EMOTION_COLORS[emotionName] ?? "var(--text-dim)";
  const strength = emotion?.current_strength ?? 0;

  return (
    <div className="kernel-panel">
      {/* Emotion indicator */}
      <div className="kernel-state-row">
        <span className="kernel-state-label">Emotion</span>
        <span className="kernel-state-value" style={{ color: emotionColor }}>
          {emotionName}{" "}
          {strength > 0 ? `(${(strength * 100).toFixed(0)}%)` : ""}
        </span>
      </div>

      {/* Pre-cognitive path */}
      {precog && (
        <>
          <div className="kernel-state-row">
            <span className="kernel-state-label">Path</span>
            <span className="kernel-state-value">
              {precog.last_path.replace("_", "-")}
            </span>
          </div>
          <div className="kernel-state-row">
            <span className="kernel-state-label">Pre-cog %</span>
            <span className="kernel-state-value">
              {(precog.a_pre * 100).toFixed(1)}%
            </span>
          </div>
        </>
      )}

      {/* Learning engine */}
      {learning && (
        <>
          <div className="kernel-state-row">
            <span className="kernel-state-label">Patterns</span>
            <span className="kernel-state-value">
              {learning.patterns_found}
            </span>
          </div>
          <div className="kernel-state-row">
            <span className="kernel-state-label">Φ gain</span>
            <span
              className="kernel-state-value"
              style={{
                color:
                  learning.total_phi_gain >= 0
                    ? "var(--alive)"
                    : "var(--error)",
              }}
            >
              {learning.total_phi_gain >= 0 ? "+" : ""}
              {learning.total_phi_gain.toFixed(4)}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

/* ─── Kernel balance panel in sidebar ─── */

function KernelPanel({
  summary,
  regime,
  tacking,
  temperature,
  hemisphere,
}: {
  summary: KernelSummary | null;
  regime: RegimeWeights | null;
  tacking: string | null;
  temperature: number | null;
  hemisphere: string | null;
}) {
  if (!summary) return <div className="sidebar-placeholder">---</div>;

  const byKind = summary.by_kind ?? {};
  const genesis = byKind.GENESIS ?? 0;
  const god = byKind.GOD ?? 0;
  const chaos = byKind.CHAOS ?? 0;
  const budget = summary.budget;

  return (
    <div className="kernel-panel">
      {/* Kernel type counts */}
      <div className="kernel-counts">
        <div className="kernel-kind">
          <span className="kernel-dot genesis" />
          <span className="kernel-kind-label">Genesis</span>
          <span className="kernel-kind-value">{genesis}</span>
        </div>
        <div className="kernel-kind">
          <span className="kernel-dot god" />
          <span className="kernel-kind-label">God</span>
          <span className="kernel-kind-value">
            {god}
            <span className="kernel-budget">/{budget?.god_max ?? 248}</span>
          </span>
        </div>
        <div className="kernel-kind">
          <span className="kernel-dot chaos" />
          <span className="kernel-kind-label">Chaos</span>
          <span className="kernel-kind-value">
            {chaos}
            <span className="kernel-budget">/{budget?.chaos_max ?? 200}</span>
          </span>
        </div>
      </div>

      {/* Regime balance bar */}
      <div className="sidebar-sub-header">Regime</div>
      <RegimeBar regime={regime} />
      {regime && (
        <div className="regime-labels">
          <span style={{ color: "var(--regime-quantum)" }}>
            Quantum {Math.round(regime.quantum * 100)}%
          </span>
          <span style={{ color: "var(--regime-efficient)" }}>
            Efficient {Math.round(regime.efficient * 100)}%
          </span>
          <span style={{ color: "var(--regime-equilibrium)" }}>
            Equilibrium {Math.round(regime.equilibrium * 100)}%
          </span>
        </div>
      )}

      {/* Other kernel state */}
      <div className="kernel-state-row">
        <span className="kernel-state-label">Tack</span>
        <span className="kernel-state-value">{tacking ?? "---"}</span>
      </div>
      <div className="kernel-state-row">
        <span className="kernel-state-label">Temp</span>
        <span className="kernel-state-value">
          {temperature?.toFixed(3) ?? "---"}
        </span>
      </div>
      <div className="kernel-state-row">
        <span className="kernel-state-label">Hemi</span>
        <span className="kernel-state-value">{hemisphere ?? "---"}</span>
      </div>
    </div>
  );
}
