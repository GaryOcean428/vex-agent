import type { RegimeWeights } from "../../types/consciousness.ts";
import "./MetricsSidebar.css";

interface RegimeBarProps {
  regime: RegimeWeights | null;
  /** Compact variant for use inside message metadata */
  compact?: boolean;
}

export function RegimeBar({ regime, compact = false }: RegimeBarProps) {
  if (!regime) return null;

  const q = Math.round(regime.quantum * 100);
  const e = Math.round(regime.efficient * 100);
  const eq = Math.round(regime.equilibrium * 100);

  return (
    <div
      className={`regime-bar${compact ? " regime-bar--compact" : ""}`}
      title={`Quantum: ${q}%  Efficient: ${e}%  Equilibrium: ${eq}%`}
      role="img"
      aria-label={`Regime — Quantum ${q}%, Efficient ${e}%, Equilibrium ${eq}%`}
    >
      <div className="regime-segment regime-q" style={{ width: `${q}%` }} />
      <div className="regime-segment regime-e" style={{ width: `${e}%` }} />
      <div className="regime-segment regime-eq" style={{ width: `${eq}%` }} />
    </div>
  );
}
