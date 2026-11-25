import { useEffect, useRef, useState } from 'react';
import { ArrowUp, Paperclip, Square } from 'lucide-react';
import { useWorkspace } from '../state/WorkspaceContext';

const MIN_HEIGHT = 44;
const MAX_HEIGHT = 200;

export function Composer({ onOpenAttachments }: { onOpenAttachments: () => void }) {
  const { state, send, stop } = useWorkspace();
  const [text, setText] = useState('');
  const area = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const node = area.current;
    if (!node) return;
    node.style.height = 'auto';
    node.style.height =
      Math.min(Math.max(node.scrollHeight, MIN_HEIGHT), MAX_HEIGHT) + 'px';
  }, [text]);

  const submit = () => {
    if (!text.trim() || state.running) return;
    void send(text);
    setText('');
  };

  const canSend = text.trim().length > 0 && !state.running;

  return (
    <div className="px-4 pb-4 pt-1 sm:px-6">
      <div className="mx-auto w-full max-w-[768px]">
        <div className="flex items-end gap-2 rounded-2xl border border-rule bg-panel/80 px-2.5 py-2 shadow-lg shadow-bg/40 transition-colors focus-within:border-accent/60">
          <button
            type="button"
            onClick={onOpenAttachments}
            className="mb-1 flex size-8 shrink-0 items-center justify-center rounded-lg text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
            title="Attach engineering documents"
            aria-label="Attach engineering documents"
          >
            <Paperclip className="size-4" />
          </button>

          <textarea
            ref={area}
            value={text}
            rows={1}
            onChange={(event) => setText(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="Model a vehicle with four wheels and an engine…"
            className="max-h-[200px] min-h-[44px] flex-1 resize-none bg-transparent py-2 text-[15px] leading-relaxed text-ink outline-none placeholder:text-ink-dim/70"
            aria-label="Message"
          />

          {state.running ? (
            <button
              type="button"
              onClick={stop}
              className="mb-1 flex size-8 shrink-0 items-center justify-center rounded-lg border border-rule text-ink-dim transition-colors hover:border-bad/60 hover:text-bad"
              title="Stop"
              aria-label="Stop generating"
            >
              <Square className="size-3 fill-current" />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              className="mb-1 flex size-8 shrink-0 items-center justify-center rounded-lg bg-accent text-bg transition-colors hover:bg-accent/90 disabled:bg-rule disabled:text-ink-dim/60"
              title="Send"
              aria-label="Send message"
            >
              <ArrowUp className="size-4" />
            </button>
          )}
        </div>

        <p className="mt-2 flex items-center justify-between gap-3 px-1 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-dim/70">
          <span>enter sends · shift+enter newline</span>
          <span>{state.docIds.length > 0 ? state.docIds.length + ' doc(s) · grounded' : ''}</span>
        </p>

        <p className="mt-1 truncate px-1 font-mono text-[10px] uppercase tracking-[0.12em] text-ink-dim/50">
          deep agent: plans · validates · delegates · streaming from the local workflow
        </p>
      </div>
    </div>
  );
}
