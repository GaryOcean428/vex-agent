import { useEffect, useReducer } from 'react';
import { API } from '../config/api-routes.ts';
import type {
  BasinData,
  BasinHistoryResponse,
  CoordizerStats,
  GraphNodesResponse,
  HealthStatus,
  KernelListResponse,
  KernelSummary,
  MemoryStatsResponse,
  ModalStatus,
  SleepStateResponse,
  TrainingStats,
  VexState,
  VexTelemetry,
} from '../types/consciousness.ts';
import { usePolledData } from './usePolledData.ts';

export { usePolledData } from './usePolledData.ts';

export const useVexState = () => usePolledData<VexState>(API.state, 2000);
export const useTelemetry = () => usePolledData<VexTelemetry>(API.telemetry, 3000);
export const useHealth = () => usePolledData<HealthStatus>(API.health, 5000);
export const useKernels = () => usePolledData<KernelSummary>(API.kernels, 2000);
export const useBasin = () => usePolledData<BasinData>(API.basin, 2000);

// New hooks for Phase 1 dashboard endpoints
export const useKernelList = () => usePolledData<KernelListResponse>(API.kernelList, 2000);
export const useBasinHistory = () => usePolledData<BasinHistoryResponse>(API.basinHistory, 5000);
export const useGraphNodes = () => usePolledData<GraphNodesResponse>(API.graphNodes, 3000);
export const useMemoryStats = () => usePolledData<MemoryStatsResponse>(API.memoryStats, 5000);
export const useSleepState = () => usePolledData<SleepStateResponse>(API.sleepState, 3000);
export const useTrainingStats = () => usePolledData<TrainingStats>(API.trainingStats, 10000);
export const useCoordizerStats = () => usePolledData<CoordizerStats>(API.coordizerStats, 10000);
export const useModalStatus = () => usePolledData<ModalStatus>(API.trainingModalStatus, 15000);
export const useHealthReachability = () => usePolledData<Record<string, { reachable: boolean; error?: string }>>(API.healthReachability, 30000);

/**
 * Accumulate polled VexState snapshots into a time-series array.
 * Callers MUST pass their own VexState data (from useVexState()) to
 * avoid duplicate /state polling loops.
 */
export function useMetricsHistory(data: VexState | null | undefined, maxPoints: number = 100) {
  // Use dispatch with payload — avoids setState and ref-mutation-during-render lint rules
  const [history, dispatch] = useReducer(
    (
      prev: Array<VexState & { time: number }>,
      action: { data: VexState; maxPoints: number },
    ) => {
      const next = [...prev, { ...action.data, time: Date.now() }];
      return next.length > action.maxPoints ? next.slice(-action.maxPoints) : next;
    },
    [],
  );

  useEffect(() => {
    if (data) dispatch({ data, maxPoints });
  }, [data, maxPoints]);

  return history;
}
