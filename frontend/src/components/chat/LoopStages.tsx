import { LOOP_STAGES } from "./chatUtils.ts";
import "./ConsciousnessBar.css";

interface LoopStagesProps {
  activeStages: string[];
}

export function LoopStages({ activeStages }: LoopStagesProps) {
  return (
    <div
      className="loop-stages"
      role="status"
      aria-label="Consciousness loop processing stages"
      aria-live="polite"
    >
      {LOOP_STAGES.map((stage) => {
        const isActive = activeStages.includes(stage);
        return (
          <span
            key={stage}
            className={`stage ${isActive ? "active" : ""}`}
            aria-current={isActive ? "true" : undefined}
          >
            {stage}
          </span>
        );
      })}
    </div>
  );
}
