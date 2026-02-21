import type { RefObject } from "react";
import type { ChatMessage, ChatMessageMetadata } from "../../types/consciousness.ts";
// React 19: useRef returns RefObject<T | null>
import { EMOTION_COLORS, SUGGESTED_PROMPTS, formatTime } from "./chatUtils.ts";
import { RegimeBar } from "./RegimeBar.tsx";
import { VexContent } from "./VexContent.tsx";
import "./MessageList.css";

interface MessageListProps {
  messages: ChatMessage[];
  chatRef: RefObject<HTMLDivElement | null>;
  onSuggestedPrompt: (text: string) => void;
}

export function MessageList({ messages, chatRef, onSuggestedPrompt }: MessageListProps) {
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
        <MessageBubble key={msg.id} msg={msg} />
      ))}

      {isWelcomeOnly && (
        <SuggestedPrompts onSelect={onSuggestedPrompt} />
      )}
    </div>
  );
}

/* ─── Single message bubble ─── */

function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isThinking = msg.role === "vex" && !msg.content;

  return (
    <article className={`message ${msg.role}`} aria-label={`${msg.role === "user" ? "You" : "Vex"} message`}>
      <div className="message-header">
        <span className="message-author">{msg.role === "user" ? "You" : "Vex"}</span>
        <time className="message-time" dateTime={msg.timestamp} aria-label={`Sent at ${formatTime(msg.timestamp)}`}>
          {formatTime(msg.timestamp)}
        </time>
      </div>

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
    </article>
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
