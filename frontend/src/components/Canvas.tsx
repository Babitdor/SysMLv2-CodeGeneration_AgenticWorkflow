import { useEffect, useState } from 'react';
import { Network, Plus, Trash2, X } from 'lucide-react';
import type { ArtifactSummary } from '../lib/types';
import { useWorkspace } from '../state/WorkspaceContext';
import { ArtifactEditor } from './ArtifactEditor';
import { Markdown } from './Markdown';
import { approveCode } from '../lib/api';
import { useAccount } from '../lib/account';

const GHOST =
  'rounded-md border border-rule px-2.5 py-1.5 text-xs text-ink-dim transition-colors hover:border-accent/60 hover:text-accent focus-visible:outline-accent disabled:opacity-40';

const PANEL = '@container flex h-[min(45dvh,32rem)] min-h-0 w-full shrink-0 flex-col border-t border-rule bg-panel lg:h-auto lg:w-[clamp(23rem,43vw,56rem)] lg:border-l lg:border-t-0';

function clock(at: number): string {
  if (!at) return '';
  return new Date(at * 1000).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * The canvas: the thread's artifacts as tabs, the version rail for the open one,
 * and an editor whose preview is always visible. It takes the same right-hand
 * slot as the output pane, the visualizer and the model index, so only one of
 * them is ever open.
 */
export function Canvas() {
  const { state, dispatch, openArtifact, openDraft, closeCanvas, saveCanvas, restoreCanvas, removeCanvas, visualize } =
    useWorkspace();
  const account = useAccount();
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const canvas = state.canvas;
  const perform = async (action: () => Promise<void>) => {
    setBusy(true);
    setNotice('');
    try {
      await action();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  // Cmd/Ctrl+S saves the document instead of opening the browser's save dialog.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (canvas && !busy && (event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void saveCanvas().catch(() => undefined);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [saveCanvas, canvas, busy]);

  if (state.liveDocument) return (
    <aside className={PANEL} aria-label="Canvas document">
      <header className="border-b border-rule px-4 py-3 text-sm font-medium">Drafting document…</header>
      <div className="min-h-0 flex-1 overflow-auto px-5 py-4">
        <Markdown text={state.liveDocument} />
      </div>
    </aside>
  );
  if (!canvas) return null;

  // The rail is newest-first, so its head is the current version: a version
  // further down is older, which is the only state Restore needs to know about.
  const current = canvas.versions[0]?.version ?? canvas.version;
  const stale = canvas.version < current;
  const candidate = state.proposal;
  const proposal = candidate?.replacements.length && canvas.id === candidate.base.id &&
    canvas.title === candidate.base.title && canvas.kind === candidate.base.kind
    ? candidate : undefined;
  const canApply = !!proposal?.complete && canvas.id === proposal.base.id &&
    canvas.title === proposal.base.title && canvas.kind === proposal.base.kind &&
    canvas.version === proposal.base.version && canvas.draft === proposal.base.draft;

  const open = (id: string, version?: number) => {
    void perform(() => openArtifact(id, version));
  };

  const discard = () => {
    if (canvas.id === null) {
      closeCanvas();
      return;
    }
    if (window.confirm(`Delete "${canvas.title}" and all of its versions?`)) {
      const id = canvas.id;
      void perform(() => removeCanvas(id));
    }
  };

  return (
    <aside className={PANEL} aria-label="Canvas">
      <div className="flex items-center gap-1 overflow-x-auto border-b border-rule px-2 py-1.5">
        {state.artifacts.map((item: ArtifactSummary) => (
          <button
            key={item.id}
            type="button"
            onClick={() => open(item.id)}
            title={item.title}
            className={
              'max-w-[160px] shrink-0 truncate rounded-sm px-2 py-1 text-left font-mono text-[10.5px] uppercase tracking-[0.1em] transition-colors ' +
              (item.id === canvas.id ? 'bg-rule-soft text-ink' : 'text-ink-dim hover:text-ink')
            }
          >
            {item.title}
          </button>
        ))}
        <span className="flex-1" />
        <button
          type="button"
          className={GHOST}
          title="New Markdown document"
          aria-label="New Markdown document"
          onClick={() => openDraft({ title: 'untitled', kind: 'markdown', content: '' })}
        >
          <Plus className="size-3" /> <span className="sr-only">New document</span>
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-2 border-b border-rule px-3 py-2">
        <h2 className="min-w-0 truncate text-sm font-semibold" title={canvas.title}>{canvas.title}</h2>
        {canvas.dirty && (
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-warn">
            unsaved
          </span>
        )}
        <select
          value={canvas.version}
          aria-label="Version"
          disabled={canvas.id === null || canvas.versions.length === 0}
          onChange={(event) => {
            if (canvas.id !== null) open(canvas.id, Number(event.target.value));
          }}
          className="rounded-sm border border-rule bg-transparent px-1 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-dim disabled:opacity-40"
        >
          {canvas.versions.length === 0 ? (
            <option value={0}>v—</option>
          ) : (
            canvas.versions.map((item) => (
              <option key={item.version} value={item.version}>
                {`v${item.version} · ${item.origin} · ${clock(item.at)}`}
              </option>
            ))
          )}
        </select>
        <span className="flex-1" />
        <button
          type="button"
          className={GHOST}
          onClick={() => void perform(saveCanvas)}
          disabled={!canvas.dirty || busy}
          title="Save as a new version (Ctrl+S)"
        >
          Save
        </button>
        <button
          type="button"
          className={GHOST}
          onClick={() => void perform(() => restoreCanvas(canvas.version))}
          disabled={!stale || busy}
          title={stale ? 'Copy this version forward as the newest one' : 'Already the newest version'}
        >
          Restore
        </button>
        {canvas.kind === 'code' && <>
          <button type="button" className={GHOST} aria-label="Visualize SysML"
            onClick={() => visualize(canvas.draft)} disabled={!canvas.draft.trim()}>
            <Network className="size-3.5" />
          </button>
          {account?.role === 'admin' && <button type="button" className={GHOST}
            disabled={busy || !canvas.draft.trim()}
            onClick={() => void perform(async () => {
              const result = await approveCode(canvas.draft, canvas.title);
              setNotice(result.stored ? 'Model approved and saved to the knowledge base.' : result.validation ?? 'Approval failed.');
            })}>
            Approve
          </button>}
        </>}
        <button
          type="button"
          className={GHOST}
          onClick={discard}
          title={canvas.id === null ? 'Discard this draft' : 'Delete this artifact'}
          aria-label={canvas.id === null ? 'Discard draft' : 'Delete artifact'}
        >
          {canvas.id === null ? <X className="size-3" /> : <Trash2 className="size-3" />}
        </button>
        <button
          type="button"
          className={GHOST}
          onClick={closeCanvas}
          title="Close the canvas"
          aria-label="Close canvas"
        >
          Close
        </button>
      </div>

      {notice && <p role="status" className="border-b border-rule px-4 py-2 text-xs text-ink">{notice}</p>}
      {proposal && <section aria-label="Proposed changes" className="shrink-0 border-b border-rule bg-panel">
        <div className="flex flex-wrap items-center gap-2 px-4 py-2">
          <h3 className="mr-auto text-sm font-medium">
            {proposal.complete ? 'Review proposed changes' : 'Agent is editing…'}
          </h3>
          <button type="button" className={GHOST} disabled={!proposal.complete}
            onClick={() => dispatch({ type: 'canvasProposal', accept: false })}>Discard</button>
          <button type="button" className={GHOST} disabled={!canApply}
            onClick={() => dispatch({ type: 'canvasProposal', accept: true })}>Apply to draft</button>
        </div>
        {proposal.complete && !canApply && <p className="px-4 pb-2 text-xs text-warn">
          The Canvas draft changed during this turn. Ask the agent to propose the edit again.
        </p>}
        <div className="max-h-[min(18dvh,12rem)] overflow-auto font-mono text-xs">
          {proposal.replacements.map(({ line, old, new: added }, index) => (
            <div key={index} className="border-t border-rule">
              <p className="px-4 py-1 text-ink-dim">Line {line}</p>
              {old && <pre className="whitespace-pre-wrap break-words bg-bad/10 px-4 py-1 text-bad"><del>{old}</del></pre>}
              {added && <pre className="whitespace-pre-wrap break-words bg-ok/10 px-4 py-1 text-ok"><ins>{added}</ins></pre>}
            </div>
          ))}
        </div>
      </section>}
      <ArtifactEditor proposed={proposal?.content} />
    </aside>
  );
}
