import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from 'react';
import type { Dispatch, ReactNode } from 'react';
import {
  createArtifact,
  deleteArtifact,
  listArtifacts,
  loadArtifact,
  restoreVersion,
  saveVersion,
} from '../lib/artifacts';
import {
  createThread,
  deleteThread,
  lastModel,
  listThreads,
  loadThread,
  renameThread,
  streamThreadChat,
} from '../lib/threads';
import type {
  Artifact,
  ArtifactKind,
  ArtifactSummary,
  ArtifactVersion,
  Message,
  Route,
  SSEEvent,
  Thread,
  ThreadSummary,
  ToolStep,
  ValidationResult,
} from '../lib/types';

/**
 * The open canvas document. `id` is null until an unsaved draft is stored, and
 * `draft` is what the editor shows — it differs from `content` only while the
 * document is dirty, so one comparison drives Save, Close and the discard guard.
 */
export interface CanvasState {
  id: string | null;
  title: string;
  kind: ArtifactKind;
  version: number;
  versions: ArtifactVersion[];
  content: string;
  draft: string;
  dirty: boolean;
}

export interface WorkspaceState {
  threads: ThreadSummary[];
  threadId: string | null;
  title: string;
  messages: Message[];
  running: boolean;
  /** First paint waits for the sidebar so the thread does not flash empty. */
  loading: boolean;
  route: Route | null;
  agentSteps: ToolStep[];
  validationLogs: ValidationResult[];
  /** The model of the turn in flight; it lands on the message when it settles. */
  artifact: string;
  /** The model shown in the right-hand visualizer, or null when it is closed. */
  visualizer: { code: string } | null;
  /** The model shown in the right-hand index panel; never open with the visualizer. */
  model: { code: string } | null;
  /** This thread's canvas documents, newest first: the tab strip. */
  artifacts: ArtifactSummary[];
  /** The document the canvas shows; never open with the visualizer or the index. */
  canvas: CanvasState | null;
  /** A version body is in flight: the strip can select before the text arrives. */
  canvasLoading: boolean;
  liveDocument?: string;
  proposal?: {
    base: CanvasState;
    content: string;
    replacements: { line: number; old: string; new: string }[];
    complete: boolean;
  };
  docIds: string[];
  /** Bumped when backend settings change so telemetry re-polls. */
  healthTick: number;
  /** Set when the first load failed: an empty app must not look like no history. */
  bootError: string | null;
}

const newId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

/** Tab-strip shape of a loaded artifact: the count, not the version rail. */
function summaryOf(artifact: Artifact): ArtifactSummary {
  const { id, title, kind, current_version, updated_at, versions } = artifact;
  return { id, title, kind, current_version, versions: versions.length, updated_at };
}

function initialState(): WorkspaceState {
  return {
    threads: [],
    threadId: null,
    title: '',
    messages: [],
    running: false,
    loading: true,
    route: null,
    agentSteps: [],
    validationLogs: [],
    artifact: '',
    visualizer: null,
    model: null,
    artifacts: [],
    canvas: null,
    canvasLoading: false,
    docIds: [],
    healthTick: 0,
    bootError: null,
  };
}

function appendToLastAgent(state: WorkspaceState, text: string): Message[] {
  const messages = [...state.messages];
  const last = messages[messages.length - 1];
  if (last && last.role === 'assistant') {
    messages[messages.length - 1] = { ...last, text: last.text + text };
  } else {
    messages.push({ id: newId(), role: 'assistant', text });
  }
  return messages;
}

/** Drop the keys the wire left out: a done event must not blank a start row's. */
function defined<T extends object>(value: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(value).filter(([, field]) => field !== undefined),
  ) as Partial<T>;
}

function upsertStep(previous: ToolStep[], step: ToolStep): ToolStep[] {
  const steps = [...previous];
  if (step.status === 'start') {
    steps.push(step);
    return steps;
  }
  // A call id settles the row it started; pipeline stages carry none, so they
  // keep the name+attempt pairing they have always used.
  const index = steps.findIndex(
    (s) =>
      s.status === 'start' &&
      (step.callId !== undefined
        ? s.callId === step.callId
        : s.name === step.name &&
          (step.attempt === undefined || s.attempt === step.attempt)),
  );
  if (index === -1) {
    steps.push(step);
  } else {
    // Merge rather than overwrite: the start row already holds the summary.
    const { id: _id, ...fields } = step;
    steps[index] = { ...steps[index], ...defined(fields) };
  }
  return steps;
}

function traceStep(event: Extract<SSEEvent, { type: 'tool' }>): ToolStep {
  return {
    id: newId(), name: event.name, status: event.status, attempt: event.attempt,
    detail: event.detail, callId: event.call_id, summary: event.summary, args: event.args,
    output: event.output, chars: event.chars, truncated: event.truncated, ms: event.ms,
  };
}

function settled(state: WorkspaceState, stopped: boolean): WorkspaceState {
  const messages = [...state.messages];
  const last = messages[messages.length - 1];
  if (last && last.role === 'assistant') {
    messages[messages.length - 1] = {
      ...last,
      // Same shape the server stores, so the card reads one source either way.
      model: state.artifact.trim() || last.model,
      validation: state.validationLogs[state.validationLogs.length - 1] ?? last.validation,
      tools: state.agentSteps.filter((step) => step.name !== 'routing').map((step) =>
        step.status === 'start' ? { ...step, status: 'failed', detail: 'Interrupted' } : step,
      ),
      stopped: stopped || last.stopped,
    };
  }
  return {
    ...state, messages, running: false, liveDocument: undefined,
    proposal: !stopped && state.proposal?.complete ? state.proposal : undefined,
  };
}

export function applyEvent(state: WorkspaceState, event: SSEEvent): WorkspaceState {
  switch (event.type) {
    case 'route':
      return {
        ...state,
        route: event.route,
        agentSteps: state.agentSteps.map((step) =>
          step.name === 'routing'
            ? { ...step, status: 'done', detail: event.reason ?? event.route }
            : step,
        ),
      };

    case 'tool':
      return {
        ...state,
        agentSteps: upsertStep(state.agentSteps, traceStep(event)),
      };

    case 'token':
      return { ...state, messages: appendToLastAgent(state, event.text) };

    case 'code':
      return { ...state, artifact: event.code };
    case 'visualize':
      return { ...state, visualizer: { code: event.code }, model: null };
    case 'document_delta':
      return { ...state, liveDocument: (state.liveDocument ?? '') + event.text };
    case 'edit_delta':
      if (!state.proposal) return state;
      return {
        ...state,
        proposal: {
          ...state.proposal,
          content: event.content,
          replacements: [...state.proposal.replacements, {
            line: event.line, old: event.old, new: event.new,
          }],
        },
      };
    case 'edit_complete':
      return {
        ...state,
        proposal: state.proposal && event.changed ? { ...state.proposal, complete: true } : undefined,
      };

    case 'validation': {
      const { type: _type, ...rest } = event;
      const result: ValidationResult = { ...rest, source: 'antlr' };
      const logs = state.validationLogs.filter((log) => log.attempt !== result.attempt);
      return { ...state, validationLogs: [...logs, result] };
    }

    case 'pipeline':
      // The stream's own end settles the turn: clearing `running` here would
      // drop the model before it is folded into the message text.
      return state;

    case 'artifact':
      // Metadata only: the provider opens the body, so this stays a pure upsert
      // and a dirty canvas is never silently replaced here.
      return {
        ...state,
        messages: state.messages.map((message, index) =>
          index === state.messages.length - 1 && message.role === 'assistant'
            ? { ...message, artifact_id: event.artifact.id }
            : message,
        ),
        artifacts: [
          event.artifact,
          ...state.artifacts.filter((item) => item.id !== event.artifact.id),
        ],
      };

    case 'title':
      return {
        ...state,
        title: event.title || state.title,
        threads: state.threads.map((thread) =>
          thread.id === state.threadId && event.title
            ? { ...thread, title: event.title }
            : thread,
        ),
      };

    case 'error':
      // Still running: the stream closes after this frame and settles the turn.
      return { ...state, proposal: undefined, messages: appendToLastAgent(state, event.detail) };

    default:
      return state;
  }
}

export type Action =
  | { type: 'event'; event: SSEEvent }
  | { type: 'send'; text: string }
  | { type: 'tokens'; text: string }
  | { type: 'done' }
  | { type: 'stopped' }
  | { type: 'error'; detail: string }
  | { type: 'threadCreated'; thread: ThreadSummary }
  | { type: 'threadsLoaded'; threads: ThreadSummary[] }
  | { type: 'bootStarted' }
  | { type: 'bootFailed'; detail: string }
  | { type: 'threadOpened'; thread: Thread }
  | { type: 'threadTitled'; title: string }
  | { type: 'renamed'; id: string; title: string }
  | { type: 'removed'; id: string }
  | { type: 'newChat' }
  | { type: 'attach'; docIds: string[] }
  | { type: 'detach'; docId: string }
  | { type: 'visualize'; code: string }
  | { type: 'closeVisualizer' }
  | { type: 'openModel'; code: string }
  | { type: 'closeModel' }
  | { type: 'artifactsLoaded'; artifacts: ArtifactSummary[] }
  | { type: 'canvasBusy'; loading: boolean }
  | { type: 'canvasOpened'; artifact: Artifact }
  | { type: 'canvasDrafted'; draft: { title: string; kind: ArtifactKind; content: string } }
  | { type: 'canvasEdited'; text: string }
  | { type: 'canvasProposal'; accept: boolean }
  | { type: 'canvasSaved'; artifact: Artifact; source: CanvasState }
  | { type: 'canvasClosed' }
  | { type: 'canvasRemoved'; id: string }
  | { type: 'refreshHealth' };

export function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  switch (action.type) {
    case 'send':
      return {
        ...state,
        loading: false,
        running: true,
        liveDocument: undefined,
        proposal: state.canvas ? {
          base: state.canvas, content: state.canvas.draft, replacements: [], complete: false,
        } : undefined,
        route: null,
        artifact: '',
        validationLogs: [],
        messages: [
          ...state.messages,
          { id: newId(), role: 'user', text: action.text },
          { id: newId(), role: 'assistant', text: '' },
        ],
        // Seeded before routing answers so the trace reads "Routing intent -> ..." later.
        agentSteps: [{ id: newId(), name: 'routing', status: 'start' }],
      };
    case 'tokens':
      return action.text ? { ...state, messages: appendToLastAgent(state, action.text) } : state;
    case 'event':
      return applyEvent(state, action.event);
    // A late settle from a superseded stream must not touch another thread.
    case 'done':
      return state.running ? settled(state, false) : state;
    case 'stopped':
      return state.running ? settled(state, true) : state;
    case 'error':
      // The provider settles the turn right after; keep any partial model.
      return { ...state, messages: appendToLastAgent(state, action.detail) };
    case 'threadCreated':
      return {
        ...state,
        loading: false,
        threadId: action.thread.id,
        title: action.thread.title,
        threads: [action.thread, ...state.threads.filter((t) => t.id !== action.thread.id)],
      };
    case 'threadsLoaded':
      return { ...state, loading: false, bootError: null, threads: action.threads };
    case 'bootStarted':
      return { ...state, loading: true, bootError: null };
    case 'bootFailed':
      return { ...state, loading: false, bootError: action.detail };
    case 'threadOpened':
      return {
        ...state,
        loading: false,
        threadId: action.thread.id,
        title: action.thread.title,
        running: false,
        route: null,
        agentSteps: [],
        validationLogs: [],
        artifact: '',
        visualizer: null,
        model: null,
        // A thread switch must not leave another thread's canvas on screen.
        artifacts: [],
        canvas: null,
        canvasLoading: false,
        liveDocument: undefined,
        proposal: undefined,
        messages: action.thread.messages.map((turn) => ({
          id: newId(),
          role: turn.role,
          text: turn.content,
          model: turn.model,
          validation: turn.validation,
          artifact_id: turn.artifact_id,
          tools: turn.tool_trace?.reduce<ToolStep[]>((steps, event) =>
            upsertStep(steps, traceStep(event)), [],
          ).map((step) => step.status === 'start'
            ? { ...step, status: 'failed', detail: 'Interrupted' }
            : step),
        })),
      };
    case 'threadTitled':
      return applyEvent(state, { type: 'title', title: action.title });
    case 'renamed':
      return {
        ...state,
        title: action.id === state.threadId ? action.title : state.title,
        threads: state.threads.map((thread) =>
          thread.id === action.id ? { ...thread, title: action.title } : thread,
        ),
      };
    case 'removed': {
      const threads = state.threads.filter((thread) => thread.id !== action.id);
      if (action.id !== state.threadId) return { ...state, threads };
      return { ...initialState(), loading: false, threads, docIds: state.docIds };
    }
    case 'newChat':
      return { ...initialState(), loading: false, threads: state.threads, docIds: state.docIds };
    case 'visualize':
      return { ...state, visualizer: { code: action.code }, model: null };
    case 'closeVisualizer':
      return { ...state, visualizer: null };
    case 'openModel':
      return { ...state, model: { code: action.code }, visualizer: null };
    case 'closeModel':
      return { ...state, model: null };
    case 'artifactsLoaded':
      return { ...state, artifacts: action.artifacts };
    case 'canvasBusy':
      return { ...state, canvasLoading: action.loading };
    case 'canvasOpened': {
      const { artifact } = action;
      if (artifact.thread_id !== state.threadId) return state;
      return {
        ...state,
        canvasLoading: false,
        visualizer: null,
        model: null,
        canvas: {
          id: artifact.id,
          title: artifact.title,
          kind: artifact.kind,
          version: artifact.version,
          versions: artifact.versions,
          content: artifact.content,
          draft: artifact.content,
          dirty: false,
        },
      };
    }
    case 'canvasDrafted':
      // An unsaved document: it has no id and is dirty from the first keystroke,
      // which is what makes Save create it instead of appending a version.
      return {
        ...state,
        canvasLoading: false,
        visualizer: null,
        model: null,
        canvas: {
          id: null,
          title: action.draft.title,
          kind: action.draft.kind,
          version: 0,
          versions: [],
          content: '',
          draft: action.draft.content,
          dirty: true,
        },
      };
    case 'canvasEdited': {
      if (!state.canvas) return state;
      return {
        ...state,
        canvas: {
          ...state.canvas,
          draft: action.text,
          dirty: action.text !== state.canvas.content,
        },
      };
    }
    case 'canvasProposal': {
      const { canvas, proposal } = state;
      if (!action.accept) return { ...state, proposal: undefined };
      if (!canvas || !proposal?.complete || canvas.id !== proposal.base.id ||
          canvas.version !== proposal.base.version || canvas.title !== proposal.base.title ||
          canvas.kind !== proposal.base.kind || canvas.draft !== proposal.base.draft) return state;
      return {
        ...state, proposal: undefined,
        canvas: { ...canvas, draft: proposal.content, dirty: proposal.content !== canvas.content },
      };
    }
    case 'canvasSaved': {
      const { artifact, source } = action;
      if (artifact.thread_id !== state.threadId) return state;
      const updated = {
        ...state,
        canvasLoading: false,
        artifacts: [
          summaryOf(artifact),
          ...state.artifacts.filter((item) => item.id !== artifact.id),
        ],
      };
      const canvas = state.canvas;
      if (!canvas || canvas.id !== source.id || canvas.version !== source.version ||
          canvas.title !== source.title || canvas.kind !== source.kind) return updated;
      const draft = canvas.draft === source.draft ? artifact.content : canvas.draft;
      return {
        ...updated,
        canvas: {
          id: artifact.id,
          title: artifact.title,
          kind: artifact.kind,
          version: artifact.version,
          versions: artifact.versions,
          content: artifact.content,
          draft,
          dirty: draft !== artifact.content,
        },
      };
    }
    case 'canvasClosed':
      return { ...state, canvas: null, canvasLoading: false };
    case 'canvasRemoved': {
      const artifacts = state.artifacts.filter((item) => item.id !== action.id);
      return {
        ...state,
        artifacts,
        canvas: state.canvas?.id === action.id ? null : state.canvas,
      };
    }
    case 'attach':
      return { ...state, docIds: Array.from(new Set([...state.docIds, ...action.docIds])) };
    case 'detach':
      return { ...state, docIds: state.docIds.filter((id) => id !== action.docId) };
    case 'refreshHealth':
      return { ...state, healthTick: state.healthTick + 1 };
    default:
      return state;
  }
}

interface WorkspaceApi {
  state: WorkspaceState;
  dispatch: Dispatch<Action>;
  send: (text: string) => Promise<void>;
  stop: () => void;
  openThread: (id: string) => Promise<void>;
  newChat: () => void;
  rename: (id: string, title: string) => Promise<void>;
  remove: (id: string) => Promise<void>;
  visualize: (code: string) => void;
  closeVisualizer: () => void;
  openModel: (code: string) => void;
  closeModel: () => void;
  /** Canvas: a stored artifact (optionally an older version), or an unsaved draft. */
  openArtifact: (id: string, version?: number) => Promise<void>;
  openDraft: (draft: { title: string; kind: ArtifactKind; content: string }) => void;
  closeCanvas: () => void;
  editCanvas: (text: string) => void;
  saveCanvas: () => Promise<void>;
  restoreCanvas: (version: number) => Promise<void>;
  removeCanvas: (id: string) => Promise<void>;
  loadArtifacts: (threadId: string) => Promise<void>;
  /** Re-polls /api/health (after settings changes or a workflow restart). */
  refreshHealth: () => void;
  boot: () => Promise<void>;
}

const WorkspaceContext = createContext<WorkspaceApi | null>(null);

const TOKEN_FLUSH_MS = 50;

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const abortRef = useRef<AbortController | null>(null);
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);

  // Boot: the sidebar, then the newest conversation. Never auto-create one.
  // Retryable, because "empty app" and "server down" must not look alike.
  const boot = useCallback(async () => {
    dispatch({ type: 'bootStarted' });
    try {
      const threads = await listThreads();
      dispatch({ type: 'threadsLoaded', threads });
      if (threads.length === 0) return;
      const thread = await loadThread(threads[0].id);
      dispatch({ type: 'threadOpened', thread });
    } catch (err) {
      dispatch({
        type: 'bootFailed',
        detail: err instanceof Error ? err.message : String(err),
      });
    }
  }, []);

  useEffect(() => {
    void boot();
  }, [boot]);

  // Unsaved canvas text belongs to the user: ask before discarding it. Every
  // path that replaces the open document goes through here.
  const discardable = useCallback(
    () =>
      !stateRef.current.canvas?.dirty ||
      window.confirm('Discard the unsaved changes in the canvas?'),
    [],
  );

  const loadArtifacts = useCallback(async (threadId: string) => {
    const artifacts = await listArtifacts(threadId);
    if (stateRef.current.threadId === threadId) dispatch({ type: 'artifactsLoaded', artifacts });
  }, []);

  // The tab strip follows the open thread, wherever it was opened from (boot,
  // the sidebar, or the thread the first send just created).
  useEffect(() => {
    const id = state.threadId;
    if (!id) return;
    void loadArtifacts(id).catch(() => undefined);
  }, [state.threadId, loadArtifacts]);

  const openArtifact = useCallback(
    async (id: string, version?: number) => {
      if (!discardable()) return;
      dispatch({ type: 'canvasBusy', loading: true });
      try {
        dispatch({ type: 'canvasOpened', artifact: await loadArtifact(id, version) });
      } catch (err) {
        // Clear the spinner, then let the caller surface the failure.
        dispatch({ type: 'canvasBusy', loading: false });
        throw err;
      }
    },
    [discardable],
  );

  const openDraft = useCallback(
    (draft: { title: string; kind: ArtifactKind; content: string }) => {
      if (discardable()) dispatch({ type: 'canvasDrafted', draft });
    },
    [discardable],
  );

  const closeCanvas = useCallback(() => {
    if (discardable()) dispatch({ type: 'canvasClosed' });
  }, [discardable]);

  const editCanvas = useCallback((text: string) => {
    dispatch({ type: 'canvasEdited', text });
  }, []);

  const saveCanvas = useCallback(async () => {
    const current = stateRef.current;
    const canvas = current.canvas;
    if (!canvas || (canvas.id !== null && !canvas.dirty)) return;
    if (canvas.id === null) {
      let threadId = current.threadId;
      if (!threadId) {
        const thread = await createThread();
        threadId = thread.id;
        dispatch({ type: 'threadCreated', thread });
      }
      const created = await createArtifact(
        threadId,
        canvas.title,
        canvas.kind,
        canvas.draft,
      );
      dispatch({ type: 'canvasSaved', artifact: created, source: canvas });
      return;
    }
    dispatch({
      type: 'canvasSaved',
      artifact: await saveVersion(canvas.id, canvas.kind, canvas.draft),
      source: canvas,
    });
  }, []);

  const restoreCanvas = useCallback(async (version: number) => {
    const canvas = stateRef.current.canvas;
    if (!canvas?.id) return;
    // Restoring appends, so the version being replaced stays in the rail.
    dispatch({ type: 'canvasSaved', artifact: await restoreVersion(canvas.id, version), source: canvas });
  }, []);

  const removeCanvas = useCallback(async (id: string) => {
    await deleteArtifact(id);
    dispatch({ type: 'canvasRemoved', id });
  }, []);

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim();
    const current = stateRef.current;
    if (!trimmed || current.running || abortRef.current) return;

    const artifact = current.canvas?.draft ?? lastModel(current.messages);
    let threadId = current.threadId;
    dispatch({ type: 'send', text: trimmed });

    const controller = new AbortController();
    abortRef.current = controller;

    // Tokens arrive per character; one dispatch per 50 ms keeps the parse cheap.
    let pending = '';
    let pendingDocument = '';
    let timer: number | null = null;
    const flush = () => {
      timer = null;
      if (abortRef.current !== controller) return;
      if (pending) dispatch({ type: 'tokens', text: pending });
      if (pendingDocument) dispatch({ type: 'event', event: { type: 'document_delta', text: pendingDocument } });
      pending = '';
      pendingDocument = '';
    };
    const queue = () => {
      if (timer === null) timer = window.setTimeout(flush, TOKEN_FLUSH_MS);
    };

    try {
      if (!threadId) {
        const thread = await createThread();
        if (controller.signal.aborted) return;
        threadId = thread.id;
        dispatch({ type: 'threadCreated', thread });
      }

      await streamThreadChat(
        threadId,
        {
          message: trimmed, doc_ids: current.docIds, artifact,
          canvas_kind: current.canvas?.kind, canvas_title: current.canvas?.title,
        },
        (event) => {
          if (abortRef.current !== controller || controller.signal.aborted) return;
          if (event.type === 'token') {
            pending += event.text;
            queue();
            return;
          }
          if (event.type === 'document_delta') {
            pendingDocument += event.text;
            queue();
            return;
          }
          flush();
          dispatch({ type: 'event', event });
          // The turn's artifact exists now; open it, unless a dirty draft is up.
          if (event.type === 'artifact' && !stateRef.current.canvas?.dirty) {
            void openArtifact(event.artifact.id).catch(() => undefined);
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (!controller.signal.aborted) {
        const detail = err instanceof Error ? err.message : String(err);
        flush();
        dispatch({ type: 'error', detail: 'Request failed: ' + detail });
      }
    } finally {
      if (timer !== null) window.clearTimeout(timer);
      if (abortRef.current === controller) {
        flush();
        abortRef.current = null;
        dispatch({ type: controller.signal.aborted ? 'stopped' : 'done' });
      }
      const active = threadId;
      if (active) {
        // Recency and the derived title come from the server, not from guesses.
        listThreads()
          .then((threads) => dispatch({ type: 'threadsLoaded', threads }))
          .catch(() => undefined);
      }
    }
  }, [openArtifact]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const openThread = useCallback(async (id: string) => {
    if (id === stateRef.current.threadId || !discardable()) return;
    abortRef.current?.abort();
    const thread = await loadThread(id);
    dispatch({ type: 'threadOpened', thread });
  }, [discardable]);

  const newChat = useCallback(() => {
    if (!discardable()) return;
    abortRef.current?.abort();
    abortRef.current = null;
    dispatch({ type: 'newChat' });
  }, [discardable]);

  const rename = useCallback(async (id: string, title: string) => {
    const named = String(title || '').trim();
    if (!named) return;
    const thread = await renameThread(id, named);
    dispatch({ type: 'renamed', id, title: thread.title });
  }, []);

  const remove = useCallback(async (id: string) => {
    await deleteThread(id);
    dispatch({ type: 'removed', id });
  }, []);

  const visualize = useCallback((code: string) => {
    if (code.trim()) dispatch({ type: 'visualize', code });
  }, []);

  const closeVisualizer = useCallback(() => {
    dispatch({ type: 'closeVisualizer' });
  }, []);

  const openModel = useCallback((code: string) => {
    if (code.trim()) dispatch({ type: 'openModel', code });
  }, []);

  const closeModel = useCallback(() => {
    dispatch({ type: 'closeModel' });
  }, []);

  const refreshHealth = useCallback(() => {
    dispatch({ type: 'refreshHealth' });
  }, []);

  const value = useMemo(
    () => ({
      state,
      dispatch,
      send,
      stop,
      openThread,
      newChat,
      rename,
      remove,
      visualize,
      closeVisualizer,
      openModel,
      closeModel,
      openArtifact,
      openDraft,
      closeCanvas,
      editCanvas,
      saveCanvas,
      restoreCanvas,
      removeCanvas,
      loadArtifacts,
      refreshHealth,
      boot,
    }),
    [
      state,
      send,
      stop,
      openThread,
      newChat,
      rename,
      remove,
      visualize,
      closeVisualizer,
      openModel,
      closeModel,
      openArtifact,
      openDraft,
      closeCanvas,
      editCanvas,
      saveCanvas,
      restoreCanvas,
      removeCanvas,
      loadArtifacts,
      refreshHealth,
      boot,
    ],
  );
  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}

export function useWorkspace(): WorkspaceApi {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error('useWorkspace must be used inside WorkspaceProvider');
  return value;
}
