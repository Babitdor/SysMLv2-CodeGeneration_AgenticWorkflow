// fetch + SSE reader. The server frames as `event: <type>\ndata: {...}\n\n`.
import type { SSEEvent } from './types';

/** One `\n\n`-framed SSE block -> its JSON payload; null when unreadable. */
export function parseFrame(frame: string): SSEEvent | null {
  const data = frame
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trim())
    .join('');
  if (!data) return null;
  try {
    return JSON.parse(data) as SSEEvent;
  } catch {
    // A frame we cannot read is not worth killing the stream over.
    return null;
  }
}

/** POST a JSON body and fail loudly on a non-stream answer. */
export async function openEventStream(
  path: string,
  payload: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => String(res.status));
    throw new Error(detail || String(res.status));
  }
  if (!res.body) throw new Error('the server sent no stream');
  return res;
}

/** Read `\n\n`-framed events until the stream ends. `ping` frames are keepalives. */
export async function consumeEventStream(
  res: Response,
  onEvent: (event: SSEEvent) => void,
): Promise<void> {
  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let cut = buffer.indexOf('\n\n');
      while (cut !== -1) {
        const event = parseFrame(buffer.slice(0, cut));
        buffer = buffer.slice(cut + 2);
        if (event && event.type !== 'ping') onEvent(event);
        cut = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.cancel().catch(() => undefined);
  }
}
