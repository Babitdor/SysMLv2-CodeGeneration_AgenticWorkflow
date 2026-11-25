import { useState } from 'react';
import { resetWorkflow } from '../lib/api';
import { useHealth } from '../lib/useHealth';
import { useWorkspace } from '../state/WorkspaceContext';

const KEY = 'font-mono text-[10px] uppercase tracking-[0.14em] text-ink-dim/70';
const VALUE = 'font-mono text-[11px] text-ink/85';
const GHOST =
  'rounded-sm border border-rule px-2 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-accent/60 hover:text-accent disabled:opacity-40 disabled:hover:border-rule disabled:hover:text-ink-dim';

function Cell({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex min-w-0 items-baseline gap-1.5">
      <em className={KEY + ' not-italic'}>{label}</em>
      <strong className={VALUE + ' truncate font-normal'}>{value}</strong>
    </span>
  );
}

/** What the agent is running on, plus the two run controls. */
export function StatusBar({ onOpenSettings }: { onOpenSettings: () => void }) {
  const { state, newChat, refreshHealth } = useWorkspace();
  const payload = useHealth(state.healthTick);
  const [restarting, setRestarting] = useState(false);

  const restart = async () => {
    setRestarting(true);
    try {
      await resetWorkflow();
    } catch {
      // The dot reports the outcome: a failed reset leaves the workflow down.
    } finally {
      setRestarting(false);
      refreshHealth();
    }
  };

  const dot = payload?.ready ? 'bg-ok' : payload ? 'bg-bad' : 'bg-ink-dim/50';
  // The counters come from the worker's already-built workflow; /api/health must
  // not be what builds it, so an unknown count is shown as unknown, not as zero.
  const store =
    payload?.ready && payload.rag_chunks !== undefined && payload.approved_entries !== undefined
      ? `${payload.rag_chunks} / ${payload.approved_entries}`
      : '—';

  return (
    <header className="flex shrink-0 items-center justify-between gap-4 border-b border-rule px-3 py-2 sm:px-4">
      <span className="flex shrink-0 items-baseline gap-2 font-mono text-[12px] uppercase tracking-[0.16em]">
        <span className="text-accent">◇</span>
        <span className="text-ink/90">SysML v2</span>
        <span className="text-ink-dim/50">/</span>
        <span className="text-ink-dim">deep agent</span>
      </span>

      <div className="flex min-w-0 items-center gap-3">
        <span className="hidden items-baseline gap-3 md:flex">
          <Cell label="backend" value={payload?.provider ?? '—'} />
          <Cell label="model" value={payload?.model ?? '—'} />
          <Cell
            label="temp"
            value={payload?.temperature !== undefined ? String(payload.temperature) : '—'}
          />
          <Cell label="store" value={store} />
        </span>
        <span
          className={'inline-block size-1.5 shrink-0 rounded-full ' + dot}
          title={payload?.error ?? 'backend health'}
        />
        <button type="button" className={GHOST} onClick={onOpenSettings}>
          settings
        </button>
        <button type="button" className={GHOST} onClick={newChat}>
          new run
        </button>
        <button
          type="button"
          className={GHOST}
          onClick={() => void restart()}
          disabled={restarting}
          title="Rebuild the workflow (embeddings + knowledge stores)"
        >
          {restarting ? 'restarting…' : 'restart'}
        </button>
      </div>
    </header>
  );
}
