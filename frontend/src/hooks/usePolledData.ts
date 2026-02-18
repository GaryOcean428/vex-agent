import { useState, useEffect, useCallback, useRef } from 'react';

interface PolledResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function usePolledData<T>(
  endpoint: string,
  intervalMs: number = 2000,
): PolledResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const fetchData = useCallback(async () => {
    // Cancel any in-flight request to prevent stale data overwrites
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(endpoint, { signal: controller.signal });
      if (res.status === 401) {
        // Session expired — redirect to login
        window.location.href = '/login';
        return;
      }
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = await res.json();
      if (!controller.signal.aborted) {
        setData(json);
        setError(null);
        setLoading(false);
      }
    } catch (err) {
      if ((err as Error).name === 'AbortError') return;
      if (!controller.signal.aborted) {
        setError((err as Error).message);
        setLoading(false);
      }
    }
  }, [endpoint]);

  useEffect(() => {
    fetchData();
    const timer = setInterval(fetchData, intervalMs);
    return () => {
      clearInterval(timer);
      abortRef.current?.abort();
    };
  }, [fetchData, intervalMs]);

  return { data, loading, error, refetch: fetchData };
}
