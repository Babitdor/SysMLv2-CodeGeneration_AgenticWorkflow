import { useEffect, useRef, useState } from 'react';
import { Loader2, X } from 'lucide-react';
import { renderModel } from '../lib/render';
import type { Compartments, Orientation, RenderOptions } from '../lib/render';
import { useMediaQuery } from '../lib/useMediaQuery';
import { useWorkspace } from '../state/WorkspaceContext';

const INLINE = '(min-width: 1100px)';
const CONTROL =
  'rounded-sm border border-rule bg-bg-2 px-1.5 py-0.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-dim';

/** Right-hand panel: draws the current model with the kernel's %viz renderer. */
export function Visualizer() {
  const { state, closeVisualizer } = useWorkspace();
  const inline = useMediaQuery(INLINE);
  const code = state.visualizer?.code ?? '';

  const [elementInput, setElementInput] = useState('');
  const [request, setRequest] = useState<Required<RenderOptions>>({
    element: '',
    orientation: 'TB',
    compartments: 'default',
  });
  const [nonce, setNonce] = useState(0);
  const [loading, setLoading] = useState(Boolean(code.trim()));
  const [dataUrl, setDataUrl] = useState('');
  const [error, setError] = useState('');
  const [cached, setCached] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const inFlight = useRef<AbortController | null>(null);

  // One render per request: opening the panel draws the default view, and each
  // control change or Render press (a new `nonce`) draws again. The cleanup
  // aborts, because one render can hold a kernel slot for minutes.
  useEffect(() => {
    if (!code.trim()) return;
    const controller = new AbortController();
    inFlight.current = controller;
    renderModel(code, request, controller.signal)
      .then((reply) => {
        if (controller.signal.aborted) return;
        setLoading(false);
        if (reply.success && reply.data_url) {
          setDataUrl(reply.data_url);
          setCached(Boolean(reply.cached));
        } else {
          setError(reply.message || 'The visualizer returned no diagram.');
        }
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setLoading(false);
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (inFlight.current === controller) inFlight.current = null;
      });
    return () => controller.abort();
  }, [code, request, nonce]);

  useEffect(() => {
    if (!loading) return;
    const started = Date.now();
    const id = window.setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(id);
  }, [loading]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeVisualizer();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [closeVisualizer]);

  /** Arm a render; `fresh` also applies the element name typed in the box. */
  const draw = (next: Required<RenderOptions>, fresh = false) => {
    setLoading(true);
    setError('');
    setElapsed(0);
    setRequest(next);
    if (fresh) setNonce((value) => value + 1);
  };

  const cancel = () => {
    inFlight.current?.abort();
    inFlight.current = null;
    setLoading(false);
    if (!dataUrl) setError('Render cancelled.');
  };

  const panel = (
    <aside
      aria-label="Model visualizer"
      className={
        inline
          ? 'slide-in flex w-[440px] shrink-0 flex-col border-l border-rule bg-bg-2/60 xl:w-[520px]'
          : 'slide-in flex h-full w-full flex-col border-l border-rule bg-bg sm:w-[520px]'
      }
    >
      <header className="flex items-center gap-2 border-b border-rule px-3 py-2">
        <span className="label">visualizer</span>
        {loading && (
          <span className="font-mono text-[10.5px] tracking-[0.1em] text-ink-dim">{elapsed}s</span>
        )}
        {!loading && cached && dataUrl && (
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-ok/80">
            cached
          </span>
        )}
        <button
          type="button"
          onClick={closeVisualizer}
          className="ml-auto rounded-sm p-1 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
          title="Close visualizer"
          aria-label="Close visualizer"
        >
          <X className="size-3.5" />
        </button>
      </header>

      <div className="flex flex-wrap items-center gap-1.5 border-b border-rule-soft px-3 py-2">
        <select
          value={request.orientation}
          onChange={(event) =>
            draw({ ...request, orientation: event.target.value as Orientation })
          }
          className={CONTROL}
          title="Orientation"
          aria-label="Orientation"
        >
          <option value="TB">TB</option>
          <option value="LR">LR</option>
        </select>
        <select
          value={request.compartments}
          onChange={(event) =>
            draw({ ...request, compartments: event.target.value as Compartments })
          }
          className={CONTROL}
          title="Compartments"
          aria-label="Compartments"
        >
          <option value="default">default</option>
          <option value="compact">compact</option>
          <option value="tree">tree</option>
        </select>
        <input
          value={elementInput}
          onChange={(event) => setElementInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') draw({ ...request, element: elementInput }, true);
          }}
          placeholder="element"
          className={`${CONTROL} w-24 normal-case`}
          aria-label="Element name"
        />
        {loading ? (
          <button
            type="button"
            onClick={cancel}
            className={`${CONTROL} ml-auto transition-colors hover:text-ink`}
          >
            cancel
          </button>
        ) : (
          <button
            type="button"
            onClick={() => draw({ ...request, element: elementInput }, true)}
            className={`${CONTROL} ml-auto text-accent transition-colors hover:text-ink`}
          >
            {error ? 'retry' : 'render'}
          </button>
        )}
      </div>

      <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-3">
        {loading ? (
          <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim">
            <Loader2 className="size-3.5 animate-spin" />
            rendering · {elapsed}s
          </span>
        ) : error ? (
          <p className="whitespace-pre-wrap text-center font-mono text-[11.5px] leading-relaxed text-bad/90">
            {error}
          </p>
        ) : dataUrl ? (
          <img src={dataUrl} alt="SysML diagram" className="max-h-full max-w-full" />
        ) : null}
      </div>
    </aside>
  );

  if (inline) return panel;
  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        aria-label="Close visualizer"
        onClick={closeVisualizer}
        className="absolute inset-0 bg-bg/70"
      />
      <div className="relative z-10 h-full w-full sm:w-[520px]">{panel}</div>
    </div>
  );
}
