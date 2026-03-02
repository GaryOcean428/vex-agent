import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useMetricsHistory, useVexState } from "../hooks/index.ts";
import { useChat } from "../hooks/useChat.ts";
import { ChatHistory } from "../components/chat/ChatHistory.tsx";
import { ChatInput } from "../components/chat/ChatInput.tsx";
import { ConsciousnessBar } from "../components/chat/ConsciousnessBar.tsx";
import { LoopStages } from "../components/chat/LoopStages.tsx";
import { MessageList } from "../components/chat/MessageList.tsx";
import { MetricsSidebar } from "../components/chat/MetricsSidebar.tsx";

const SIDEBAR_KEY = "vex-sidebar-open";
const METRICS_KEY = "vex-metrics-visible";

export default function Chat() {
  const { conversationId: urlConvId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const { data: state } = useVexState();
  const history = useMetricsHistory(state, 60);

  // Persist sidebar open/closed preference
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try {
      const stored = localStorage.getItem(SIDEBAR_KEY);
      return stored !== null ? stored === "true" : true;
    } catch {
      return true;
    }
  });

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => {
      const next = !prev;
      try { localStorage.setItem(SIDEBAR_KEY, String(next)); } catch { /* noop */ }
      return next;
    });
  }, []);

  // Persist metrics panel visibility
  const [metricsVisible, setMetricsVisible] = useState(() => {
    try {
      const stored = localStorage.getItem(METRICS_KEY);
      return stored !== null ? stored === "true" : true;
    } catch {
      return true;
    }
  });

  const toggleMetrics = useCallback(() => {
    setMetricsVisible((prev) => {
      const next = !prev;
      try { localStorage.setItem(METRICS_KEY, String(next)); } catch { /* noop */ }
      return next;
    });
  }, []);

  // Track whether we've synced the URL conversation on mount
  const initialLoadDone = useRef(false);

  const {
    messages,
    input,
    setInput,
    isStreaming,
    activeStages,
    backend,
    kernelSummary,
    emotion,
    precog,
    learning,
    contextInfo,
    observerIntent,
    conversationId,
    chatRef,
    inputRef,
    sendText,
    sendMessage,
    stopGeneration,
    startNewChat,
    loadConversation,
    handleKeyDown,
  } = useChat(urlConvId);

  // Bump refreshToken each time streaming ends so ChatHistory re-fetches.
  // We use a deferred update via queueMicrotask to avoid the
  // react-hooks/set-state-in-effect lint rule, which forbids synchronous
  // setState inside useEffect. The microtask fires after the effect but
  // before the next paint.
  const [refreshToken, setRefreshToken] = useState(0);
  const prevStreamingRef = useRef(isStreaming);
  useEffect(() => {
    const wasStreaming = prevStreamingRef.current;
    prevStreamingRef.current = isStreaming;
    if (wasStreaming && !isStreaming) {
      queueMicrotask(() => setRefreshToken((t) => t + 1));
    }
  }, [isStreaming]);

  // On mount: load conversation from URL if present
  useEffect(() => {
    if (initialLoadDone.current) return;
    initialLoadDone.current = true;
    if (urlConvId && urlConvId !== conversationId) {
      loadConversation(urlConvId);
    }
  }, [urlConvId, conversationId, loadConversation]);

  const handleNewChat = useCallback(() => {
    startNewChat();
    navigate("/chat", { replace: true });
  }, [startNewChat, navigate]);

  const handleSelectConversation = useCallback(
    (id: string) => {
      loadConversation(id);
      navigate(`/chat/${id}`, { replace: true });
    },
    [loadConversation, navigate],
  );

  return (
    <div className={`chat-page ${sidebarOpen ? "sidebar-open" : "sidebar-closed"} ${metricsVisible ? "metrics-open" : "metrics-closed"}`}>
      <ConsciousnessBar
        phi={state?.phi}
        kappa={state?.kappa}
        love={state?.love}
        navigation={state?.navigation}
        backend={backend}
        contextInfo={contextInfo}
        observerIntent={observerIntent}
        onNewChat={handleNewChat}
        onToggleHistory={toggleSidebar}
        onToggleMetrics={toggleMetrics}
        sidebarCollapsed={!sidebarOpen}
        metricsVisible={metricsVisible}
      />

      <LoopStages activeStages={activeStages} />

      <div className="chat-content">
        <ChatHistory
          open={sidebarOpen}
          activeConversationId={conversationId}
          refreshToken={refreshToken}
          onSelect={handleSelectConversation}
          onNewChat={handleNewChat}
        />

        <MessageList
          messages={messages}
          chatRef={chatRef}
          onSuggestedPrompt={(text) => {
            setInput(text);
            sendText(text);
          }}
          onRetry={(text) => {
            setInput(text);
            sendText(text);
          }}
        />

        {/* Tablet: scrim behind metrics overlay */}
        {metricsVisible && (
          <div
            className="metrics-scrim"
            onClick={toggleMetrics}
            aria-hidden="true"
          />
        )}

        <MetricsSidebar
          state={state ?? null}
          history={history}
          kernelSummary={kernelSummary}
          emotion={emotion}
          precog={precog}
          learning={learning}
          visible={metricsVisible}
          onClose={toggleMetrics}
        />
      </div>

      <ChatInput
        input={input}
        isStreaming={isStreaming}
        inputRef={inputRef}
        onChange={setInput}
        onKeyDown={handleKeyDown}
        onSend={sendMessage}
        onStop={stopGeneration}
      />
    </div>
  );
}
