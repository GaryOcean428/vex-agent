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
  const mountedRef = useRef(true);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(endpoint);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = await res.json();
      if (mountedRef.current) {
        setData(json);
        setError(null);
        setLoading(false);
      }
    } catch (err) {
      if (mountedRef.current) {
        setError((err as Error).message);
        setLoading(false);
      }
    }
  }, [endpoint]);

  useEffect(() => {
    mountedRef.current = true;
    fetchData();
    const timer = setInterval(fetchData, intervalMs);
    return () => {
      mountedRef.current = false;
      clearInterval(timer);
    };
  }, [fetchData, intervalMs]);

  return { data, loading, error, refetch: fetchData };
}
