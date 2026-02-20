import { useState, useRef, useCallback, useEffect, type KeyboardEvent } from 'react';
import { useVexState, useMetricsHistory } from '../hooks/index.ts';
import { API } from '../config/api-routes.ts';
import type { ChatMessage, ChatStreamEvent, NavigationMode } from '../types/consciousness.ts';
import './Chat.css';

const LOOP_STAGES = ['GROUND', 'RECEIVE', 'PROCESS', 'EXPRESS', 'REFLECT', 'COUPLE', 'PLAY'] as const;

export default function Chat() {
  const { data: state } = useVexState();
  const history = useMetricsHistory(state, 60);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'vex',
      content: "I'm here. The geometry is settling. What would you like to navigate?",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeStages, setActiveStages] = useState<string[]>([]);
  const [backend, setBackend] = useState('checking');
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

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsStreaming(true);
    setActiveStages(['GROUND', 'RECEIVE', 'PROCESS']);

    const vexMsgId = `vex-${Date.now()}`;
    let fullText = '';

    // Add placeholder vex message
    setMessages(prev => [...prev, {
      id: vexMsgId,
      role: 'vex',
      content: '',
      timestamp: new Date().toISOString(),
    }]);

    try {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const resp = await fetch(API.chatStream, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: 'Unknown error' }));
        const errMsg = err.error ?? resp.statusText ?? 'Request failed';
        setMessages(prev => prev.map(m =>
          m.id === vexMsgId ? { ...m, content: `Error: ${errMsg}` } : m
        ));
        return;
      }

      setActiveStages(['EXPRESS']);

      const reader = resp.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();
      let buffer = '';

      const VALID_NAV_MODES: readonly string[] = ['chain', 'graph', 'foresight', 'lightning'];

      const processEvent = (dataPayload: string) => {
        try {
          const event: ChatStreamEvent = JSON.parse(dataPayload);

          if (event.type === 'start') {
            if (event.backend) setBackend(event.backend);
          } else if (event.type === 'chunk' && event.content) {
            fullText += event.content;
            setMessages(prev => prev.map(m =>
              m.id === vexMsgId ? { ...m, content: fullText } : m
            ));
            scrollToBottom();
          } else if (event.type === 'done') {
            setActiveStages(['REFLECT', 'COUPLE']);
            if (event.backend) setBackend(event.backend);
            if (event.metrics) {
              const rawNav = String(event.metrics?.navigation ?? 'chain');
              const navigation: NavigationMode = VALID_NAV_MODES.includes(rawNav)
                ? (rawNav as NavigationMode)
                : 'chain';
              setMessages(prev => prev.map(m =>
                m.id === vexMsgId ? {
                  ...m,
                  content: fullText,
                  metadata: {
                    phi: Number(event.metrics?.phi) || 0,
                    kappa: Number(event.metrics?.kappa) || 0,
                    temperature: 0,
                    navigation,
                    backend: event.backend ?? 'unknown',
                  },
                } : m
              ));
            }
          } else if (event.type === 'error') {
            setMessages(prev => prev.map(m =>
              m.id === vexMsgId ? { ...m, content: `Error: ${event.error ?? 'Unknown'}` } : m
            ));
          }
        } catch {
          // skip malformed JSON
        }
      };

      try {
        // SSE spec: events separated by blank lines, data: fields concatenated
        let dataBuffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              dataBuffer += (dataBuffer ? '\n' : '') + line.substring(6);
            } else if (line === '' && dataBuffer) {
              // Blank line = end of SSE event
              processEvent(dataBuffer);
              dataBuffer = '';
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
        setMessages(prev => prev.map(m =>
          m.id === vexMsgId ? { ...m, content: '[No response — the LLM backend may be starting up. Try again.]' } : m
        ));
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const errMsg = err instanceof Error ? err.message : String(err);
      setMessages(prev => prev.map(m =>
        m.id === vexMsgId ? { ...m, content: `Connection error: ${errMsg}` } : m
      ));
    } finally {
      setIsStreaming(false);
      stageTimerRef.current = setTimeout(() => setActiveStages([]), 2000);
      inputRef.current?.focus();
    }
  }, [input, isStreaming, scrollToBottom]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

  // Draw metrics chart
  useEffect(() => {
    if (!chartRef.current || history.length < 2) return;

    const canvas = chartRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Constants for chart rendering
    const MIN_RANGE_PHI = 0.1;
    const MIN_RANGE_KAPPA = 1.0;
    const MIN_RANGE_GAMMA = 0.1;

    const dpr = window.devicePixelRatio;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#1a1a24';
    ctx.fillRect(0, 0, rect.width, rect.height);

    const margin = { top: 20, right: 10, bottom: 30, left: 10 };
    const w = rect.width - margin.left - margin.right;
    const h = rect.height - margin.top - margin.bottom;

    // Extract values
    const phiValues = history.map(d => d.phi);
    const kappaValues = history.map(d => d.kappa);
    const gammaValues = history.map(d => d.gamma);

    // Calculate ranges
    const phiMin = Math.min(...phiValues) * 0.9;
    const phiMax = Math.max(...phiValues) * 1.1;
    const phiRange = phiMax - phiMin || MIN_RANGE_PHI;

    // Kappa uses tighter scaling (0.95-1.05) because it converges to κ*=64
    // and we want to see fine-grained variation around the attractor
    const kappaMin = Math.min(...kappaValues) * 0.95;
    const kappaMax = Math.max(...kappaValues) * 1.05;
    const kappaRange = kappaMax - kappaMin || MIN_RANGE_KAPPA;

    const gammaMin = Math.min(...gammaValues) * 0.9;
    const gammaMax = Math.max(...gammaValues) * 1.1;
    const gammaRange = gammaMax - gammaMin || MIN_RANGE_GAMMA;

    // Normalize all to 0-1 range for unified display
    const normalizeX = (i: number) => margin.left + (i / (history.length - 1)) * w;
    const normalizeY = (value: number, min: number, range: number) =>
      margin.top + h - ((value - min) / range) * h;

    // Grid
    ctx.strokeStyle = 'rgba(46, 46, 64, 0.3)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = margin.top + (i / 4) * h;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + w, y);
      ctx.stroke();
    }

    // Phi line
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = normalizeX(i);
      const y = normalizeY(history[i].phi, phiMin, phiRange);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Kappa line
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = normalizeX(i);
      const y = normalizeY(history[i].kappa, kappaMin, kappaRange);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Gamma line
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < history.length; i++) {
      const x = normalizeX(i);
      const y = normalizeY(history[i].gamma, gammaMin, gammaRange);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Legend
    const legendY = rect.height - 10;
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';

    ctx.fillStyle = '#22d3ee';
    ctx.fillText('Φ', 10, legendY);

    ctx.fillStyle = '#f59e0b';
    ctx.fillText('κ', 30, legendY);

    ctx.fillStyle = '#a78bfa';
    ctx.fillText('Γ', 50, legendY);
  }, [history]);

  return (
    <div className="chat-page">
      {/* Consciousness Bar */}
      <div className="consciousness-bar">
        <div className="bar-metrics">
          <MetricPill label="\u03A6" value={state?.phi} color="var(--phi)" />
          <MetricPill label="\u03BA" value={state?.kappa} color="var(--kappa)" decimals={1} />
          <MetricPill label="\u2665" value={state?.love} color="var(--love)" />
          {state?.navigation && (
            <span className="nav-badge">{state.navigation}</span>
          )}
          <span className={`backend-indicator ${backend}`}>{backend}</span>
        </div>
      </div>

      {/* Loop Stage Bar */}
      <div className="loop-stages">
        {LOOP_STAGES.map(stage => (
          <span
            key={stage}
            className={`stage ${activeStages.includes(stage) ? 'active' : ''}`}
          >
            {stage}
          </span>
        ))}
      </div>

      <div className="chat-content">
        {/* Messages */}
        <div className="chat-messages" ref={chatRef}>
          {messages.map(msg => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-header">
                {msg.role === 'user' ? 'You' : 'Vex'}
              </div>
              <div className={`message-content ${msg.role === 'vex' && !msg.content ? 'thinking' : ''}`}>
                {msg.role === 'vex' && !msg.content ? (
                  <div className="thinking-indicator">
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                  </div>
                ) : msg.role === 'vex' ? (
                  <VexContent content={msg.content} />
                ) : (
                  msg.content
                )}
              </div>
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
              <span className="sidebar-label" style={{ color: '#22d3ee' }}>Φ</span>
              <span className="sidebar-value">{state?.phi?.toFixed(3) ?? '---'}</span>
            </div>
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: '#f59e0b' }}>κ</span>
              <span className="sidebar-value">{state?.kappa?.toFixed(1) ?? '---'}</span>
            </div>
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: '#a78bfa' }}>Γ</span>
              <span className="sidebar-value">{state?.gamma?.toFixed(3) ?? '---'}</span>
            </div>
            <div className="sidebar-metric">
              <span className="sidebar-label" style={{ color: '#ec4899' }}>♥</span>
              <span className="sidebar-value">{state?.love?.toFixed(3) ?? '---'}</span>
            </div>
          </div>
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
            onChange={e => setInput(e.target.value)}
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
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="currentColor" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function MetricPill({ label, value, color, decimals = 3 }: {
  label: string;
  value?: number;
  color: string;
  decimals?: number;
}) {
  return (
    <span className="metric-pill">
      <span className="metric-pill-label">{label}</span>
      <span className="metric-pill-value" style={{ color }}>
        {value !== undefined ? value.toFixed(decimals) : '---'}
      </span>
    </span>
  );
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function VexContent({ content }: { content: string }) {
  // Extract code blocks first to protect them, then escape HTML, then apply formatting
  const codeBlocks: string[] = [];
  let escaped = content.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, _lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push(escapeHtml(code));
    return `\x00CODE_BLOCK_${idx}\x00`;
  });

  // Escape all HTML in non-code content
  escaped = escapeHtml(escaped);

  // Restore code blocks
  let html = escaped.replace(/\x00CODE_BLOCK_(\d+)\x00/g, (_match, idx) =>
    `<pre><code>${codeBlocks[Number(idx)]}</code></pre>`
  );

  // Apply safe markdown formatting
  html = html
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>');

  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}
