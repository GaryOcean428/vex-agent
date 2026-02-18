import { useState, useEffect } from 'react';
import { usePolledData } from './usePolledData.ts';
import type { VexState, VexTelemetry, HealthStatus, KernelSummary, BasinData } from '../types/consciousness.ts';

export { usePolledData } from './usePolledData.ts';

export const useVexState = () => usePolledData<VexState>('/state', 2000);
export const useTelemetry = () => usePolledData<VexTelemetry>('/telemetry', 3000);
export const useHealth = () => usePolledData<HealthStatus>('/health', 5000);
export const useKernels = () => usePolledData<KernelSummary>('/kernels', 2000);
export const useBasin = () => usePolledData<BasinData>('/basin', 2000);

/**
 * Accumulate polled VexState snapshots into a time-series array.
 * Callers MUST pass their own VexState data (from useVexState()) to
 * avoid duplicate /state polling loops.
 */
export function useMetricsHistory(data: VexState | null | undefined, maxPoints: number = 100) {
  const [history, setHistory] = useState<Array<VexState & { time: number }>>([]);

  useEffect(() => {
    if (!data) return;
    setHistory(prev => {
      const next = [...prev, { ...data, time: Date.now() }];
      return next.length > maxPoints ? next.slice(-maxPoints) : next;
    });
  }, [data, maxPoints]);

  return history;
}
