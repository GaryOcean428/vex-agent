import { useState, useRef, useCallback, useEffect } from 'react';
import { useVexState } from '../hooks/index.ts';
import type { ChatMessage, ChatStreamEvent } from '../types/consciousness.ts';
import './Chat.css';

const LOOP_STAGES = ['GROUND', 'RECEIVE', 'PROCESS', 'EXPRESS', 'REFLECT', 'COUPLE', 'PLAY'] as const;

export default function Chat() {
  const { data: state } = useVexState();
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

      const resp = await fetch('/chat/stream', {
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

      const processLine = (line: string) => {
        if (line === '' || line.startsWith(':')) return;
        if (!line.startsWith('data: ')) return;

        try {
          const event: ChatStreamEvent = JSON.parse(line.substring(6));

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
              setMessages(prev => prev.map(m =>
                m.id === vexMsgId ? {
                  ...m,
                  content: fullText,
                  metadata: {
                    phi: Number(event.metrics?.phi) || 0,
                    kappa: Number(event.metrics?.kappa) || 0,
                    temperature: 0,
                    navigation: (String(event.metrics?.navigation ?? 'chain')) as 'chain',
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
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n');
          buffer = parts.pop() ?? '';

          for (const line of parts) {
            processLine(line);
          }
        }

        // Process any remaining buffer after stream ends
        if (buffer.trim()) {
          processLine(buffer);
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

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }, [sendMessage]);

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
