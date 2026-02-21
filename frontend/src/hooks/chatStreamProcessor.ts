/**
 * Pure SSE event processor for the chat stream.
 * Extracted from useChat.ts to keep that file under 300 lines.
 */
import type {
  ChatMessageMetadata,
  ChatStreamEvent,
  EmotionState,
  KernelSummary,
  LearningState,
  NavigationMode,
  PreCogState,
  RegimeWeights,
} from "../types/consciousness.ts";

const VALID_NAV_MODES: readonly string[] = ["chain", "graph", "foresight", "lightning"];

interface ContextInfo {
  total_tokens: number;
  compression_tier: number;
  escalated: boolean;
}

interface ProcessorSetters {
  setBackend: (v: string) => void;
  setConversationId: (v: string) => void;
  setContextInfo: (v: ContextInfo) => void;
  setKernelSummary: (v: KernelSummary) => void;
  setEmotion: (v: EmotionState) => void;
  setPrecog: (v: PreCogState) => void;
  setLearning: (v: LearningState) => void;
  setActiveStages: (v: string[]) => void;
  setObserverIntent: (v: string) => void;
  appendChunk: (vexMsgId: string, chunk: string) => void;
  finaliseMessage: (vexMsgId: string, fullText: string, metadata: ChatMessageMetadata) => void;
  setError: (vexMsgId: string, error: string) => void;
  onChunk: () => void;
}

/**
 * Returns a stateful processEvent function bound to the provided setters.
 * Maintains accumulated fullText across multiple chunk events.
 */
export function createEventProcessor(
  vexMsgId: string,
  setters: ProcessorSetters,
): { processEvent: (dataPayload: string) => void; getFullText: () => string } {
  let fullText = "";

  function processEvent(dataPayload: string): void {
    try {
      const event: ChatStreamEvent = JSON.parse(dataPayload);

      if (event.type === "start") {
        if (event.backend) setters.setBackend(event.backend);
        if (event.conversation_id) setters.setConversationId(event.conversation_id);
        if (event.context) setters.setContextInfo(event.context);
        if (event.kernels) setters.setKernelSummary(event.kernels);
        if (event.consciousness?.emotion) setters.setEmotion(event.consciousness.emotion);
        if (event.consciousness?.precog) setters.setPrecog(event.consciousness.precog);
        if (event.consciousness?.learning) setters.setLearning(event.consciousness.learning);
      } else if (event.type === "chunk" && event.content) {
        fullText += event.content;
        setters.appendChunk(vexMsgId, fullText);
        setters.onChunk();
      } else if (event.type === "done") {
        setters.setActiveStages(["INTEGRATE", "EXPRESS", "TUNE"]);
        if (event.backend) setters.setBackend(event.backend);
        if (event.context) setters.setContextInfo(event.context);
        if (event.kernels) setters.setKernelSummary(event.kernels);
        if (event.observer?.refined_intent) setters.setObserverIntent(event.observer.refined_intent);
        if (event.metrics?.emotion) setters.setEmotion(event.metrics.emotion);
        if (event.metrics?.precog) setters.setPrecog(event.metrics.precog);
        if (event.metrics?.learning) setters.setLearning(event.metrics.learning);
        if (event.metrics) {
          const m = event.metrics;
          const rawNav = String(m.navigation ?? "chain");
          const navigation: NavigationMode = VALID_NAV_MODES.includes(rawNav)
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
            regime: (m.regime as RegimeWeights) ?? { quantum: 0, efficient: 0, equilibrium: 0 },
            tacking: m.tacking ?? { mode: "balanced", oscillation_phase: 0, cycle_count: 0 },
            hemispheres: m.hemispheres ?? { active: "integrated", balance: 0.5 },
            kernels_active: Number(m.kernels_active) || 0,
            lifecycle_phase: String(m.lifecycle_phase ?? "ACTIVE"),
            emotion: m.emotion,
            precog: m.precog,
            learning: m.learning,
          };
          setters.finaliseMessage(vexMsgId, fullText, metadata);
        }
      } else if (event.type === "error") {
        setters.setError(vexMsgId, event.error ?? "Unknown");
      }
    } catch {
      // skip malformed JSON
    }
  }

  return { processEvent, getFullText: () => fullText };
}
