import { useMetricsHistory, useVexState } from "../hooks/index.ts";
import { useChat } from "../hooks/useChat.ts";
import { ChatInput } from "../components/chat/ChatInput.tsx";
import { ConsciousnessBar } from "../components/chat/ConsciousnessBar.tsx";
import { LoopStages } from "../components/chat/LoopStages.tsx";
import { MessageList } from "../components/chat/MessageList.tsx";
import { MetricsSidebar } from "../components/chat/MetricsSidebar.tsx";
import "./Chat.css";

export default function Chat() {
  const { data: state } = useVexState();
  const history = useMetricsHistory(state, 60);

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
    chatRef,
    inputRef,
    sendText,
    sendMessage,
    startNewChat,
    handleKeyDown,
  } = useChat();

  return (
    <div className="chat-page">
      <ConsciousnessBar
        phi={state?.phi}
        kappa={state?.kappa}
        love={state?.love}
        navigation={state?.navigation}
        backend={backend}
        contextInfo={contextInfo}
        observerIntent={observerIntent}
        onNewChat={startNewChat}
      />

      <LoopStages activeStages={activeStages} />

      <div className="chat-content">
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
