import { useEffect, useRef, useState } from 'react';
import { health } from './api';
import type { HealthPayload } from './types';

/** Backend status for the sidebar chip; re-polls whenever settings change. */
export function useHealth(tick: number): HealthPayload | null {
  const [payload, setPayload] = useState<HealthPayload | null>(null);
  const attempts = useRef(0);

  useEffect(() => {
    let cancelled = false;
    // Re-arm on every settings change: the model shown must be the model in use.
    attempts.current = 0;
    setPayload(null);

    const poll = async () => {
      try {
        const next = await health();
        if (cancelled) return;
        setPayload(next);
        // First load boots the workflow (embeddings + two Chroma stores): retry.
        if (!next.ready && attempts.current < 20) {
          attempts.current += 1;
          window.setTimeout(poll, 3000);
        }
      } catch (err) {
        if (cancelled) return;
        setPayload({ ready: false, provider: '?', model: '?', error: String(err) });
      }
    };

    void poll();
    return () => {
      cancelled = true;
    };
  }, [tick]);

  return payload;
}
