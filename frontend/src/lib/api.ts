import type { DocumentInfo, HealthPayload, ToolSchema } from './types';

/** Signed-in account as the server describes it (never a secret). */
export interface SessionUser {
  id: string;
  email: string;
  role: string;
}

export interface MeResponse {
  user: SessionUser;
  signup_open: boolean;
}

/**
 * Called when any request comes back 401: the session expired or was revoked
 * server-side. AuthGate subscribes and falls back to the sign-in form, so a
 * stale tab does not just throw "401" at the user.
 */
let unauthorizedHandler: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

/** Exported for the newer clients (artifacts.ts) so 401 → AuthGate stays one path. */
export async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = String(res.status);
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    if (res.status === 401) unauthorizedHandler?.();
    const error = new Error(detail) as Error & { status?: number };
    error.status = res.status;
    throw error;
  }
  return (await res.json()) as T;
}

export const postJson = <T,>(path: string, payload?: unknown): Promise<T> =>
  json<T>(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload ?? {}),
  });

export const me = () => json<MeResponse>('/api/auth/me');

export const login = (email: string, password: string) =>
  postJson<{ user: SessionUser }>('/api/auth/login', { email, password });

export const register = (email: string, password: string) =>
  postJson<{ user: SessionUser }>('/api/auth/register', { email, password });

export const logout = () => postJson<{ ok: boolean }>('/api/auth/logout');

export const health = () => json<HealthPayload>('/api/health');

export const fetchTools = async (): Promise<ToolSchema[]> =>
  (await json<{ tools: ToolSchema[] }>('/api/tools')).tools;

export const validateCode = (code: string) =>
  json<{ status: string; ok: boolean; diagnostics: { severity: string; line: number; column: number; message: string }[]; summary: string; detail: string }>(
    '/api/validate',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code }),
    },
  );

export const approveCode = (code: string, task: string) =>
  json<{ stored: boolean; validation?: string; entry_id?: string; approved_entries?: number }>(
    '/api/approve',
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ code, task }),
    },
  );

/** Drop the cached workflow and the UI's provider overrides; the next call rebuilds. */
export const resetWorkflow = () =>
  json<{ ok: boolean; provider: string; model: string }>('/api/reset', { method: 'POST' });

export interface ConfigPayload {
  provider?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  base_url?: string;
  api_key?: string;
  effort?: string;
}

export const configure = (payload: ConfigPayload, saveAs?: string) =>
  json<{
    ok: boolean;
    provider: string;
    model: string;
    temperature: number;
    active: string | null;
  }>('/api/config', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(saveAs ? { ...payload, name: saveAs } : payload),
  });

/** One saved backend configuration, as the server summarises it. */
export interface ConfigProfile {
  name: string;
  provider: string;
  model: string;
  temperature: number;
  base_url: string;
  effort: string;
  max_tokens?: string | number | null;
  timeout?: string | number | null;
  api_key_set: boolean;
}

export interface ConfigState {
  provider: string;
  model: string;
  temperature: number;
  base_url: string;
  effort: string;
  max_tokens?: string | number | null;
  timeout?: string | number | null;
  api_key_set: boolean;
  persisted: boolean;
  active: string | null;
  profiles: ConfigProfile[];
}

export const readConfig = () => json<ConfigState>('/api/config');

const profilePath = (name: string) =>
  '/api/config/profiles/' + encodeURIComponent(name);

export const applyProfile = (name: string) =>
  json<{ ok: boolean; active: string; provider: string; model: string; temperature: number }>(
    profilePath(name) + '/apply',
    { method: 'POST' },
  );

export const deleteProfile = (name: string) =>
  json<{ ok: boolean; active: string | null }>(profilePath(name), { method: 'DELETE' });

/**
 * One account's own backend, as the server describes it. There is no api_key
 * field on purpose: the key is write-only, and api_key_set says whether one is
 * stored. source is "deployment" when the account saved nothing and inherits
 * the administrator's configuration.
 */
export interface ProviderConfigState {
  provider: string;
  model: string;
  base_url: string;
  effort: string;
  temperature: number | null;
  api_key_set: boolean;
  source: 'user' | 'deployment';
}

export interface ProviderConfigPayload {
  provider: string;
  model: string;
  base_url: string;
  /** Blank keeps the stored key; '' on a fresh account stores nothing. */
  api_key?: string;
  temperature?: number | null;
  effort?: string;
}

export const readProviderConfig = () => json<ProviderConfigState>('/api/provider-config');

export const saveProviderConfig = (payload: ProviderConfigPayload) =>
  json<ProviderConfigState>('/api/provider-config', {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });

/** Drop this account's own backend: the deployment default applies again. */
export const clearProviderConfig = () =>
  json<ProviderConfigState>('/api/provider-config', { method: 'DELETE' });

export const listDocuments = async (): Promise<DocumentInfo[]> =>
  (await json<{ documents: DocumentInfo[] }>('/api/documents')).documents;

export async function uploadDocuments(files: FileList | File[]): Promise<DocumentInfo[]> {
  const form = new FormData();
  for (const file of Array.from(files)) form.append('files', file);
  const res = await fetch('/api/documents', { method: 'POST', body: form });
  if (!res.ok) {
    let detail = String(res.status);
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return ((await res.json()) as { documents: DocumentInfo[] }).documents;
}
