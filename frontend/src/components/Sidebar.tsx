import { useEffect, useRef, useState } from 'react';
import {
  Check,
  ListTree,
  Menu,
  PanelLeft,
  Pencil,
  Paperclip,
  Plus,
  Settings,
  Trash2,
  X,
} from 'lucide-react';
import { lastModel } from '../lib/threads';
import { useHealth } from '../lib/useHealth';
import { useWorkspace } from '../state/WorkspaceContext';

interface Props {
  narrow: boolean;
  rail: boolean;
  drawerOpen: boolean;
  onToggleRail: () => void;
  onCloseDrawer: () => void;
  onOpenSettings: () => void;
  onOpenAttachments: () => void;
}

function ThreadRow({
  id,
  title,
  active,
  rail,
  onOpen,
  onRename,
  onDelete,
}: {
  id: string;
  title: string;
  active: boolean;
  rail: boolean;
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(title);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  const commit = () => {
    setEditing(false);
    if (draft.trim() && draft.trim() !== title) onRename(draft.trim());
  };

  if (editing) {
    return (
      <div className="flex items-center gap-1 rounded-md bg-panel px-1.5 py-1">
        <input
          ref={input}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              commit();
            }
            if (event.key === 'Escape') {
              event.preventDefault();
              setDraft(title);
              setEditing(false);
            }
          }}
          onBlur={commit}
          aria-label="Thread title"
          className="min-w-0 flex-1 bg-transparent font-sans text-[13px] text-ink outline-none"
        />
        <button
          type="button"
          onMouseDown={(event) => event.preventDefault()}
          onClick={commit}
          className="rounded-sm p-0.5 text-ink-dim hover:text-ok"
          title="Save title"
          aria-label="Save title"
        >
          <Check className="size-3" />
        </button>
      </div>
    );
  }

  return (
    <div
      className={
        'group flex items-center gap-1 rounded-md transition-colors ' +
        (active ? 'bg-rule-soft text-ink' : 'text-ink-dim hover:bg-rule-soft/70 hover:text-ink')
      }
    >
      <button
        type="button"
        onClick={onOpen}
        aria-current={active ? 'page' : undefined}
        title={title}
        className={
          'flex min-w-0 flex-1 items-center gap-2 py-1.5 text-left text-[13px] ' +
          (rail ? 'justify-center px-0' : 'px-2')
        }
      >
        {rail ? (
          <span className="flex size-6 shrink-0 items-center justify-center rounded-sm border border-rule font-mono text-[10px] uppercase">
            {title.slice(0, 1)}
          </span>
        ) : (
          <span className="truncate">{title}</span>
        )}
      </button>
      {!rail && (
        <span className="flex items-center opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
          <button
            type="button"
            onClick={() => {
              setDraft(title);
              setEditing(true);
            }}
            className="rounded-sm p-1 text-ink-dim hover:bg-rule hover:text-ink"
            title={'Rename ' + title}
            aria-label={'Rename ' + title}
          >
            <Pencil className="size-3" />
          </button>
          <button
            type="button"
            onClick={onDelete}
            className="rounded-sm p-1 text-ink-dim hover:bg-rule hover:text-bad"
            title={'Delete ' + title}
            aria-label={'Delete ' + title}
          >
            <Trash2 className="size-3" />
          </button>
        </span>
      )}
      <span className="sr-only">{id}</span>
    </div>
  );
}

export function Sidebar({
  narrow,
  rail,
  drawerOpen,
  onToggleRail,
  onCloseDrawer,
  onOpenSettings,
  onOpenAttachments,
}: Props) {
  const { state, openThread, newChat, rename, remove } = useWorkspace();
  const payload = useHealth(state.healthTick);
  const compact = rail && !narrow;

  useEffect(() => {
    if (!narrow || !drawerOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onCloseDrawer();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [narrow, drawerOpen, onCloseDrawer]);

  const ready = payload?.ready ?? false;
  const dot = ready ? 'bg-ok' : payload ? 'bg-bad' : 'bg-ink-dim/50';

  const body = (
    <>
      {/* The wordmark lives in the status bar; this row is only the rail controls. */}
      <div className="flex items-center justify-end gap-2 px-3 py-3">
        {!narrow && (
          <button
            type="button"
            onClick={onToggleRail}
            className="rounded-sm p-1.5 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
            title={compact ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-label={compact ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-expanded={!compact}
          >
            <PanelLeft className="size-4" />
          </button>
        )}
        {narrow && (
          <button
            type="button"
            onClick={onCloseDrawer}
            className="rounded-sm p-1.5 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
            title="Close sidebar"
            aria-label="Close sidebar"
          >
            <X className="size-4" />
          </button>
        )}
      </div>

      <div className={compact ? 'px-2' : 'px-3'}>
        <button
          type="button"
          onClick={() => {
            newChat();
            onCloseDrawer();
          }}
          className={
            'flex w-full items-center gap-2 rounded-lg border border-rule px-2.5 py-2 text-[13px] text-ink transition-colors hover:border-accent/60 hover:text-accent ' +
            (compact ? 'justify-center' : '')
          }
          title="New chat"
        >
          <Plus className="size-3.5" />
          {!compact && 'New chat'}
        </button>
      </div>

      <nav aria-label="Conversations" className="mt-3 min-h-0 flex-1 overflow-y-auto px-2">
        {state.threads.length === 0 ? (
          !compact && (
            <p className="px-2 py-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-dim/60">
              no conversations yet
            </p>
          )
        ) : (
          <ul className="flex flex-col gap-0.5">
            {state.threads.map((thread) => (
              <li key={thread.id}>
                <ThreadRow
                  id={thread.id}
                  title={thread.title}
                  active={thread.id === state.threadId}
                  rail={compact}
                  onOpen={() => {
                    void openThread(thread.id);
                    onCloseDrawer();
                  }}
                  onRename={(title) => void rename(thread.id, title)}
                  onDelete={() => void remove(thread.id)}
                />
              </li>
            ))}
          </ul>
        )}
      </nav>

      <div className={'border-t border-rule-soft p-2 ' + (compact ? 'flex flex-col items-center gap-1' : 'flex flex-col gap-1')}>
        <button
          type="button"
          onClick={onOpenAttachments}
          className={
            'flex items-center gap-2 rounded-md px-2 py-1.5 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink ' +
            (compact ? 'justify-center' : 'w-full')
          }
          title="Attach engineering documents"
        >
          <Paperclip className="size-3.5" />
          {!compact && 'docs'}
          {state.docIds.length > 0 && <span className="text-accent">{state.docIds.length}</span>}
        </button>
        <button
          type="button"
          onClick={onOpenSettings}
          className={
            'flex items-center gap-2 rounded-md px-2 py-1.5 font-mono text-[11px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink ' +
            (compact ? 'justify-center' : 'w-full')
          }
          title="Backend settings"
        >
          <Settings className="size-3.5" />
          {!compact && 'settings'}
        </button>
        <div
          className={'flex items-center gap-2 px-2 pt-1 font-mono text-[10.5px] text-ink-dim ' + (compact ? 'justify-center' : '')}
          title={payload?.error ?? 'backend status'}
        >
          <span className={'inline-block size-1.5 shrink-0 rounded-full ' + dot} />
          {!compact && (
            <span className="truncate text-ink/80">{payload?.model ?? 'connecting…'}</span>
          )}
        </div>
      </div>
    </>
  );

  if (narrow) {
    return (
      <>
        {drawerOpen && (
          <div
            aria-hidden
            onClick={onCloseDrawer}
            className="fixed inset-0 z-30 bg-bg/70 backdrop-blur-sm"
          />
        )}
        <aside
          id="sidebar"
          aria-label="Conversations"
          className={
            'fixed inset-y-0 left-0 z-40 flex w-[264px] flex-col border-r border-rule bg-bg-2/95 transition-transform duration-200 ' +
            (drawerOpen ? 'translate-x-0' : '-translate-x-full')
          }
          aria-hidden={!drawerOpen}
          inert={!drawerOpen}
        >
          {body}
        </aside>
      </>
    );
  }

  return (
    <aside
      aria-label="Conversations"
      className={
        'flex shrink-0 flex-col border-r border-rule bg-bg-2/60 transition-[width] duration-200 ' +
        (compact ? 'w-16' : 'w-[264px]')
      }
    >
      {body}
    </aside>
  );
}

export function TopStrip({
  narrow,
  drawerOpen,
  onOpenDrawer,
}: {
  narrow: boolean;
  drawerOpen: boolean;
  onOpenDrawer: () => void;
}) {
  const { state, openModel } = useWorkspace();
  const model = lastModel(state.messages);

  return (
    <header className="relative z-20 flex items-center gap-3 px-4 py-2.5">
      {narrow && (
        <button
          type="button"
          onClick={onOpenDrawer}
          className="rounded-sm p-1.5 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
          title="Conversations"
          aria-label="Conversations"
          aria-expanded={drawerOpen}
          aria-controls="sidebar"
        >
          <Menu className="size-4" />
        </button>
      )}
      <h2 className="min-w-0 truncate text-[13px] text-ink-dim">
        {state.title || (narrow ? 'SysML v2' : 'New chat')}
      </h2>
      <button
        type="button"
        onClick={() => openModel(model)}
        disabled={!model}
        className="ml-auto rounded-sm p-1.5 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink disabled:opacity-40 disabled:hover:bg-transparent"
        title={model ? 'Model index' : 'Model index (no model yet)'}
        aria-label="Model index"
      >
        <ListTree className="size-4" />
      </button>
    </header>
  );
}
