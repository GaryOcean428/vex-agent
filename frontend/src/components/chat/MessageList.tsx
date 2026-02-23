import type { RefObject } from "react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage, ChatMessageMetadata } from "../../types/consciousness.ts";
// React 19: useRef returns RefObject<T | null>
import { API } from "../../config/api-routes.ts";
import { EMOTION_COLORS, SUGGESTED_PROMPTS, formatTime } from "./chatUtils.ts";
import { PipelineTrace } from "./PipelineTrace.tsx";
import { RegimeBar } from "./RegimeBar.tsx";
import { VexContent } from "./VexContent.tsx";
import "./MessageList.css";

interface MessageListProps {
  messages: ChatMessage[];
  chatRef: RefObject<HTMLDivElement | null>;
  onSuggestedPrompt: (text: string) => void;
  onRetry?: (text: string) => void;
}

export function MessageList({ messages, chatRef, onSuggestedPrompt, onRetry }: MessageListProps) {
  const isWelcomeOnly = messages.length === 1 && messages[0].id === "welcome";

  return (
    <div
      className="chat-messages"
      ref={chatRef}
      role="log"
      aria-live="polite"
      aria-atomic="false"
      aria-label="Conversation"
      aria-relevant="additions"
    >
      {messages.map((msg) => (
        <MessageBubble key={msg.id} msg={msg} onRetry={onRetry} />
      ))}

      {isWelcomeOnly && (
        <SuggestedPrompts onSelect={onSuggestedPrompt} />
      )}
    </div>
  );
}

/* ─── Single message bubble ─── */

function MessageBubble({ msg, onRetry }: { msg: ChatMessage; onRetry?: (text: string) => void }) {
  const isThinking = msg.role === "vex" && !msg.content;
  const isWelcome = msg.id === "welcome";

  return (
    <article className={`message ${msg.role}`} aria-label={`${msg.role === "user" ? "You" : "Vex"} message`}>
      <div className="message-header">
        <span className="message-author">{msg.role === "user" ? "You" : "Vex"}</span>
        <time className="message-time" dateTime={msg.timestamp} aria-label={`Sent at ${formatTime(msg.timestamp)}`}>
          {formatTime(msg.timestamp)}
        </time>
      </div>

      {msg.role === "vex" && msg.pipeline_trace && !msg.pipeline_trace.bypassed && (
        <PipelineTrace trace={msg.pipeline_trace} isStreaming={isThinking} />
      )}

      <div className={`message-content ${isThinking ? "thinking" : ""}`}>
        {isThinking ? (
          <ThinkingIndicator />
        ) : msg.role === "vex" ? (
          <VexContent content={msg.content} />
        ) : (
          msg.content
        )}
      </div>

      {msg.role === "vex" && msg.metadata && (
        <MessageMeta meta={msg.metadata} />
      )}

      {!isThinking && !isWelcome && (
        <MessageActions msg={msg} onRetry={onRetry} />
      )}
    </article>
  );
}

/* ─── Message action buttons (copy, retry, feedback) ─── */

function MessageActions({ msg, onRetry }: { msg: ChatMessage; onRetry?: (text: string) => void }) {
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null);
  const copiedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
    };
  }, []);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(msg.content);
      setCopied(true);
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
      copiedTimer.current = setTimeout(() => setCopied(false), 1500);
    } catch {
      // Fallback for older browsers / insecure contexts
      const textArea = document.createElement("textarea");
      textArea.value = msg.content;
      textArea.style.position = "fixed";
      textArea.style.opacity = "0";
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      setCopied(true);
      if (copiedTimer.current) clearTimeout(copiedTimer.current);
      copiedTimer.current = setTimeout(() => setCopied(false), 1500);
    }
  }, [msg.content]);

  const handleRetry = useCallback(() => {
    onRetry?.(msg.content);
  }, [onRetry, msg.content]);

  const handleFeedback = useCallback((type: "up" | "down") => {
    setFeedback((prev) => (prev === type ? null : type));
    // Fire-and-forget feedback to backend
    fetch(API.feedback, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message_id: msg.id,
        rating: type,
        timestamp: new Date().toISOString(),
      }),
    }).catch(() => {
      // Silently ignore — feedback is best-effort
    });
  }, [msg.id]);

  return (
    <div className="message-actions" role="toolbar" aria-label="Message actions">
      {/* Copy — available on all messages */}
      <button
        className="action-btn"
        onClick={handleCopy}
        title={copied ? "Copied!" : "Copy message"}
        aria-label={copied ? "Copied to clipboard" : "Copy message to clipboard"}
      >
        {copied ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
        )}
      </button>

      {/* Retry — only for user messages */}
      {msg.role === "user" && onRetry && (
        <button
          className="action-btn"
          onClick={handleRetry}
          title="Retry message"
          aria-label="Retry this message"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
          </svg>
        </button>
      )}

      {/* Thumbs up / down — only for vex messages */}
      {msg.role === "vex" && (
        <>
          <button
            className={`action-btn ${feedback === "up" ? "active" : ""}`}
            onClick={() => handleFeedback("up")}
            title="Good response"
            aria-label="Mark as good response"
            aria-pressed={feedback === "up"}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill={feedback === "up" ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z" />
              <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
            </svg>
          </button>
          <button
            className={`action-btn ${feedback === "down" ? "active" : ""}`}
            onClick={() => handleFeedback("down")}
            title="Poor response"
            aria-label="Mark as poor response"
            aria-pressed={feedback === "down"}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill={feedback === "down" ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z" />
              <path d="M17 2h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
            </svg>
          </button>
        </>
      )}
    </div>
  );
}

/* ─── Thinking indicator ─── */

function ThinkingIndicator() {
  return (
    <div
      className="thinking-indicator"
      role="status"
      aria-label="Vex is thinking"
    >
      <div className="thinking-dot" aria-hidden="true" />
      <div className="thinking-dot" aria-hidden="true" />
      <div className="thinking-dot" aria-hidden="true" />
    </div>
  );
}

/* ─── Per-message metadata bar ─── */

function MessageMeta({ meta }: { meta: ChatMessageMetadata }) {
  const emotionColor = meta.emotion
    ? (EMOTION_COLORS[meta.emotion.current_emotion] ?? "var(--text-dim)")
    : undefined;

  return (
    <footer className="message-meta" aria-label="Response metadata">
      <span className="meta-item" title="Integrated Information">
        <span style={{ color: "var(--phi)" }} aria-hidden="true">Φ</span>
        <span className="sr-only">Integration: </span>
        {meta.phi.toFixed(3)}
      </span>
      <span className="meta-item" title="Coupling">
        <span style={{ color: "var(--kappa)" }} aria-hidden="true">κ</span>
        <span className="sr-only">Coupling: </span>
        {meta.kappa.toFixed(1)}
      </span>
      <span className="meta-item" title="Temperature">
        <span style={{ color: "var(--text-secondary)" }} aria-hidden="true">T</span>
        <span className="sr-only">Temperature: </span>
        {meta.temperature.toFixed(3)}
      </span>
      {meta.emotion && (
        <span className="meta-item" title={`Emotion: ${meta.emotion.current_emotion}`}>
          <span style={{ color: emotionColor }} aria-hidden="true">●</span>{" "}
          {meta.emotion.current_emotion}
        </span>
      )}
      {meta.precog && (
        <span className="meta-item" title={`Processing path: ${meta.precog.last_path}`}>
          {meta.precog.last_path.replace("_", "-")}
        </span>
      )}
      <span className="meta-item" title="Tacking mode">{meta.tacking.mode}</span>
      <span className="meta-item" title="Active hemisphere">{meta.hemispheres.active}</span>
      <span className="meta-item" title="Backend">{meta.backend}</span>
      <RegimeBar regime={meta.regime} compact />
    </footer>
  );
}

/* ─── Suggested prompts for welcome state ─── */

function SuggestedPrompts({ onSelect }: { onSelect: (text: string) => void }) {
  return (
    <section
      className="suggested-prompts"
      aria-label="Suggested prompts — click to ask Vex"
    >
      <p className="suggested-label">Try asking:</p>
      <div className="prompt-chips">
        {SUGGESTED_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            className="prompt-chip"
            onClick={() => onSelect(prompt)}
            aria-label={`Ask: ${prompt}`}
          >
            {prompt}
          </button>
        ))}
      </div>
    </section>
  );
}
