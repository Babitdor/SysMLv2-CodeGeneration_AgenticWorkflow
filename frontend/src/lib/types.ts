// The wire contract: mirrors the SSE table in the plan and webapp/pipeline.py.

export type Route = 'general_chat' | 'sysml_generation' | 'document' | 'artifact_edit' | 'web_search' | 'visualize';
export type ToolStatus = 'start' | 'done' | 'failed';
export type ValidationStatus = 'valid' | 'invalid' | 'error';

export interface ToolStep {
  id: string;
  name: string;
  status: ToolStatus;
  attempt?: number;
  detail?: string;
  /** The wire's call_id: one row per real call, not one per tool name. */
  callId?: string;
  /** The one-line preview the backend picked, e.g. the path it read. */
  summary?: string;
  /** Raw JSON arguments, for the expanded row. */
  args?: string;
  /** The returned text as the server trimmed it (first ~1200 chars). */
  output?: string;
  /** Size of the untrimmed result, so a cut output still reports its weight. */
  chars?: number;
  truncated?: boolean;
  /** Wall time of this call or pipeline stage. */
  ms?: number;
}

export interface Diagnostic {
  severity: string;
  line: number;
  column: number;
  message: string;
}

export interface ValidationResult {
  status: ValidationStatus;
  ok: boolean;
  attempt: number;
  final: boolean;
  diagnostics: Diagnostic[];
  summary: string;
  detail?: string;
  /** antlr = server grammar; structural = local fallback when the server is down */
  source?: 'antlr' | 'structural';
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  /** The SysML model the turn produced, stored on the turn by the server. */
  model?: string;
  /** The verdict that came with that model, so Approve survives a reload. */
  validation?: ValidationResult;
  /** The canvas artifact this turn produced; absent on older or deleted ones. */
  artifact_id?: string;
  tools?: ToolStep[];
  /** Set once the turn ended without the stream's own `done` event. */
  stopped?: boolean;
}

/** Sidebar row: everything the server knows about a thread but its turns. */
export interface ThreadSummary {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface StoredTurn {
  role: 'user' | 'assistant';
  content: string;
  /** Present for a modelled turn; legacy threads carry the fence in content. */
  model?: string;
  validation?: ValidationResult;
  /** The canvas artifact this turn produced; absent on older or deleted ones. */
  artifact_id?: string;
  tool_trace?: Extract<SSEEvent, { type: 'tool' }>[];
  at?: number;
}

export interface Thread {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages: StoredTurn[];
}

/** Body of POST /api/threads/{id}/chat — mirrors ThreadChatRequest. */
export interface ThreadChatPayload {
  message: string;
  doc_ids?: string[];
  artifact?: string;
  canvas_kind?: ArtifactKind;
  canvas_title?: string;
}

export interface ToolSchema {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface HealthPayload {
  ready: boolean;
  provider: string;
  model: string;
  temperature?: number;
  base_url?: string;
  api_key_set?: boolean;
  rag_chunks?: number;
  approved_entries?: number;
  deepagents_version?: string;
  error?: string;
}

export interface DocumentInfo {
  id: string;
  name: string;
  chars: number;
  chunks: number;
}

/** A canvas document is edited as one of these; kind is per version, not fixed. */
export type ArtifactKind = 'code' | 'markdown';

/** Tab-strip row: metadata only, so listing a thread's artifacts stays cheap. */
export interface ArtifactSummary {
  id: string;
  title: string;
  kind: ArtifactKind;
  current_version: number;
  /** How many versions are still retained (the tail is pruned at 50). */
  versions: number;
  updated_at: number;
}

/** One row of the version rail. */
export interface ArtifactVersion {
  version: number;
  kind: ArtifactKind;
  origin: 'generate' | 'edit' | 'restore';
  chars: number;
  at: number;
}

/**
 * One artifact with the content of a single version: `current_version` is the
 * pointer, `version` is the one whose text came back (the same unless an older
 * version was asked for). `versions` here is the rail, not the count above.
 */
export interface Artifact {
  id: string;
  thread_id: string;
  title: string;
  kind: ArtifactKind;
  current_version: number;
  updated_at: number;
  versions: ArtifactVersion[];
  version: number;
  content: string;
}

export type SSEEvent =
  | { type: 'route'; route: Route; mode?: string; reason?: string; source?: 'llm' | 'fallback'; format?: 'v1' | 'v2' | null }
  // Optional trace fields are snake_case on the wire; ToolStep is their camelCase twin.
  | { type: 'tool'; name: string; status: ToolStatus; attempt?: number; detail?: string; call_id?: string; summary?: string; args?: string; output?: string; chars?: number; truncated?: boolean; ms?: number }
  | { type: 'token'; text: string }
  | { type: 'code'; code: string; attempt: number; final: boolean }
  | { type: 'document_delta'; text: string }
  | { type: 'edit_delta'; content: string; line: number; old: string; new: string }
  | { type: 'edit_complete'; changed: boolean }
  | { type: 'visualize'; code: string }
  | ({ type: 'validation' } & Omit<ValidationResult, 'source'>)
  | { type: 'pipeline'; status: 'done' | 'exhausted' | 'error'; attempts: number; valid: boolean }
  | { type: 'artifact'; artifact: ArtifactSummary }
  | { type: 'title'; title: string }
  | { type: 'ping' }
  | { type: 'error'; detail: string }
  | { type: 'done' };
