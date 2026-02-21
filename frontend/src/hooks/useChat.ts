import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { API } from "../config/api-routes.ts";
import type {
  ChatMessage,
  EmotionState,
  KernelSummary,
  LearningState,
  PreCogState,
} from "../types/consciousness.ts";
import { createEventProcessor } from "./chatStreamProcessor.ts";

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "vex",
  content: "Vertex active. Awaiting input.",
  timestamp: new Date().toISOString(),
};

interface ContextInfo {
  total_tokens: number;
  compression_tier: number;
  escalated: boolean;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  input: string;
  setInput: (v: string) => void;
  isStreaming: boolean;
  activeStages: string[];
  backend: string;
  kernelSummary: KernelSummary | null;
  emotion: EmotionState | null;
  precog: PreCogState | null;
  learning: LearningState | null;
  contextInfo: ContextInfo | null;
  observerIntent: string | null;
  chatRef: React.RefObject<HTMLDivElement | null>;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  sendText: (text: string) => Promise<void>;
  sendMessage: () => void;
  startNewChat: () => void;
  handleKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeStages, setActiveStages] = useState<string[]>([]);
  const [backend, setBackend] = useState("checking");
  const [kernelSummary, setKernelSummary] = useState<KernelSummary | null>(null);
  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [precog, setPrecog] = useState<PreCogState | null>(null);
  const [learning, setLearning] = useState<LearningState | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [contextInfo, setContextInfo] = useState<ContextInfo | null>(null);
  const [observerIntent, setObserverIntent] = useState<string | null>(null);

  const chatRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const stageTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (stageTimerRef.current) clearTimeout(stageTimerRef.current);
    };
  }, []);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
    });
  }, []);

  useEffect(() => { scrollToBottom(); }, [messages, scrollToBottom]);

  const startNewChat = useCallback(() => {
    abortRef.current?.abort();
    setConversationId(null);
    setMessages([{ ...WELCOME_MESSAGE, timestamp: new Date().toISOString() }]);
    setIsStreaming(false);
    setActiveStages([]);
    inputRef.current?.focus();
  }, []);

  const sendText = useCallback(
    async (text: string) => {
      if (!text || isStreaming) return;

      const vexMsgId = `vex-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        { id: `user-${Date.now()}`, role: "user", content: text, timestamp: new Date().toISOString() },
        { id: vexMsgId, role: "vex", content: "", timestamp: new Date().toISOString() },
      ]);
      setIsStreaming(true);
      setActiveStages(["SCAN", "DESIRE", "WILL", "WISDOM", "RECEIVE"]);

      // Bind SSE event processor — carries accumulated fullText internally
      const { processEvent, getFullText } = createEventProcessor(vexMsgId, {
        setBackend,
        setConversationId,
        setContextInfo: (v) => setContextInfo(v),
        setKernelSummary,
        setEmotion,
        setPrecog,
        setLearning,
        setActiveStages,
        setObserverIntent,
        appendChunk: (_id, full) =>
          setMessages((prev) => prev.map((m) => (m.id === vexMsgId ? { ...m, content: full } : m))),
        finaliseMessage: (_id, full, metadata) =>
          setMessages((prev) =>
            prev.map((msg) => (msg.id === vexMsgId ? { ...msg, content: full, metadata } : msg)),
          ),
        setError: (_id, err) =>
          setMessages((prev) =>
            prev.map((m) => (m.id === vexMsgId ? { ...m, content: `Error: ${err}` } : m)),
          ),
        onChunk: scrollToBottom,
      });

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
          setMessages((prev) =>
            prev.map((m) =>
              m.id === vexMsgId
                ? { ...m, content: `Error: ${err.error ?? resp.statusText ?? "Request failed"}` }
                : m,
            ),
          );
          return;
        }

        setActiveStages(["ENTRAIN", "COUPLE", "NAVIGATE"]);
        const reader = resp.body?.getReader();
        if (!reader) throw new Error("No response body");
        const decoder = new TextDecoder();
        let buffer = "";

        try {
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
                processEvent(dataBuffer);
                dataBuffer = "";
              }
            }
          }
          if (dataBuffer) processEvent(dataBuffer);
        } finally {
          reader.releaseLock();
        }

        if (!getFullText()) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === vexMsgId
                ? { ...m, content: "[No response — LLM backend may be starting up. Try again.]" }
                : m,
            ),
          );
        }
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") return;
        const msg = err instanceof Error ? err.message : String(err);
        setMessages((prev) =>
          prev.map((m) => (m.id === vexMsgId ? { ...m, content: `Connection error: ${msg}` } : m)),
        );
      } finally {
        setIsStreaming(false);
        stageTimerRef.current = setTimeout(() => setActiveStages([]), 2000);
        inputRef.current?.focus();
      }
    },
    [isStreaming, scrollToBottom, conversationId],
  );

  const sendMessage = useCallback(() => {
    const text = input.trim();
    setInput("");
    sendText(text);
  }, [input, sendText]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    },
    [sendMessage],
  );

  return {
    messages, input, setInput, isStreaming, activeStages, backend,
    kernelSummary, emotion, precog, learning, contextInfo, observerIntent,
    chatRef, inputRef, sendText, sendMessage, startNewChat, handleKeyDown,
  };
}
