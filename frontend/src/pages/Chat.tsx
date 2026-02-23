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
import "./Chat.css";

const SIDEBAR_KEY = "vex-sidebar-open";

export default function Chat() {
  const { data: state } = useVexState();
  const history = useMetricsHistory(state, 60);
  const navigate = useNavigate();
  const { conversationId: urlConversationId } = useParams<{ conversationId?: string }>();

  // Persist sidebar open/closed preference
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    try {
      const stored = localStorage.getItem(SIDEBAR_KEY);
      return stored !== null ? stored === "true" : true; // default open
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
    startNewChat,
    loadConversation,
    handleKeyDown,
  } = useChat();

  // On mount: load conversation from URL if present
  useEffect(() => {
    if (initialLoadDone.current) return;
    initialLoadDone.current = true;
    if (urlConversationId && urlConversationId !== conversationId) {
      loadConversation(urlConversationId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // intentionally run only once on mount

  // Sync conversation ID changes back to the URL
  useEffect(() => {
    if (!initialLoadDone.current) return; // skip during initial load
    if (conversationId && conversationId !== urlConversationId) {
      navigate(`/chat/${conversationId}`, { replace: true });
    } else if (!conversationId && urlConversationId) {
      navigate("/chat", { replace: true });
    }
  }, [conversationId, urlConversationId, navigate]);

  // Wrap startNewChat to also navigate
  const handleNewChat = useCallback(() => {
    startNewChat();
    navigate("/chat", { replace: true });
  }, [startNewChat, navigate]);

  // Wrap loadConversation to also navigate
  const handleSelectConversation = useCallback(
    (id: string) => {
      loadConversation(id);
      navigate(`/chat/${id}`, { replace: true });
    },
    [loadConversation, navigate],
  );

  return (
    <div className={`chat-page ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
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
      />

      <LoopStages activeStages={activeStages} />

      <div className="chat-content">
        <ChatHistory
          open={sidebarOpen}
          activeConversationId={conversationId}
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
        />

        <MetricsSidebar
          state={state ?? null}
          history={history}
          kernelSummary={kernelSummary}
          emotion={emotion}
          precog={precog}
          learning={learning}
        />
      </div>

      <ChatInput
        input={input}
        isStreaming={isStreaming}
        inputRef={inputRef}
        onChange={setInput}
        onKeyDown={handleKeyDown}
        onSend={sendMessage}
      />
    </div>
  );
}
