import { useEffect } from 'react';
import { Plus, Trash2, X } from 'lucide-react';
import type { ArtifactSummary } from '../lib/types';
import { useWorkspace } from '../state/WorkspaceContext';
import { ArtifactEditor } from './ArtifactEditor';

const GHOST =
  'rounded-sm border border-rule px-2 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-accent/60 hover:text-accent disabled:opacity-40 disabled:hover:border-rule disabled:hover:text-ink-dim';

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
  const { state, openArtifact, openDraft, closeCanvas, saveCanvas, restoreCanvas, removeCanvas } =
    useWorkspace();
  const canvas = state.canvas;

  // Cmd/Ctrl+S saves the document instead of opening the browser's save dialog.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        void saveCanvas().catch(() => undefined);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [saveCanvas]);

  if (!canvas) return null;

  // The rail is newest-first, so its head is the current version: a version
  // further down is older, which is the only state Restore needs to know about.
  const current = canvas.versions[0]?.version ?? canvas.version;
  const stale = canvas.version < current;

  const open = (id: string, version?: number) => {
    void openArtifact(id, version).catch(() => undefined);
  };

  const discard = () => {
    if (canvas.id === null) {
      closeCanvas();
      return;
    }
    if (window.confirm(`Delete "${canvas.title}" and all of its versions?`)) {
      void removeCanvas(canvas.id).catch(() => undefined);
    }
  };

  return (
    <aside className="flex w-[560px] shrink-0 flex-col border-l border-rule bg-panel/40">
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
          title="New markdown document"
          aria-label="New markdown document"
          onClick={() => openDraft({ title: 'untitled', kind: 'markdown', content: '' })}
        >
          <Plus className="size-3" />
        </button>
      </div>

      <div className="flex items-center gap-2 border-b border-rule px-3 py-2">
        <span className="label">canvas · {canvas.kind}</span>
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
          onClick={() => void saveCanvas().catch(() => undefined)}
          disabled={!canvas.dirty}
          title="Save as a new version (Ctrl+S)"
        >
          save
        </button>
        <button
          type="button"
          className={GHOST}
          onClick={() => void restoreCanvas(canvas.version).catch(() => undefined)}
          disabled={!stale}
          title={stale ? 'Copy this version forward as the newest one' : 'Already the newest version'}
        >
          restore
        </button>
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
          close
        </button>
      </div>

      <ArtifactEditor />
    </aside>
  );
}
