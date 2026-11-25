// Client for the durable thread API: the sidebar and the transcript both live
// here, so history survives a reload instead of one rolling /api/chat session.
import { consumeEventStream, openEventStream } from './sse';
import type {
  Message,
  SSEEvent,
  Thread,
  ThreadChatPayload,
  ThreadSummary,
} from './types';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = String(res.status);
    try {
      const body = (await res.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body: the status is all we have */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

const threadPath = (id: string) => '/api/threads/' + encodeURIComponent(id);

export const listThreads = async (): Promise<ThreadSummary[]> =>
  (await request<{ threads: ThreadSummary[] }>('/api/threads')).threads;

export const createThread = async (title = ''): Promise<ThreadSummary> =>
  (
    await request<{ thread: ThreadSummary }>('/api/threads', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ title }),
    })
  ).thread;

export const loadThread = async (id: string): Promise<Thread> =>
  (await request<{ thread: Thread }>(threadPath(id))).thread;

export const renameThread = async (id: string, title: string): Promise<ThreadSummary> =>
  (
    await request<{ thread: ThreadSummary }>(threadPath(id), {
      method: 'PATCH',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ title }),
    })
  ).thread;

export const deleteThread = async (id: string): Promise<void> => {
  await request<{ ok: boolean }>(threadPath(id), { method: 'DELETE' });
};

export async function streamThreadChat(
  id: string,
  payload: ThreadChatPayload,
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  await consumeEventStream(
    await openEventStream(threadPath(id) + '/chat', payload, signal),
    onEvent,
  );
}

const SYSML_FENCE = /```sysml\s*\n([\s\S]*?)```/i;

/**
 * The model a follow-up turn should extend: the turn's own field, else the
 * fenced block of a thread written before models were stored structurally.
 */
export function lastModel(messages: Pick<Message, 'role' | 'text' | 'model'>[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const model = message.model?.trim();
    if (model) return model;
    const match = message.text.match(SYSML_FENCE);
    if (match) return match[1].trim();
  }
  return '';
}
