import { useEffect, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { useWorkspace } from '../state/WorkspaceContext';
import { Message } from './Message';
import { ToolTraceAccordion } from './ToolTraceAccordion';
import { ValidationReport } from './ValidationReport';

const SAMPLES = [
  'model a vehicle with four wheels and an engine',
  'convert this SysML v1 block to v2',
  'what is the difference between a part and an item?',
];

export function ThreadView() {
  const { state, send, boot } = useWorkspace();
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [showPill, setShowPill] = useState(false);

  // Follow the stream only while the reader is already at the bottom.
  useEffect(() => {
    const node = scroller.current;
    if (!node || !pinned.current) return;
    node.scrollTop = node.scrollHeight;
  }, [state.messages]);

  const onScroll = () => {
    const node = scroller.current;
    if (!node) return;
    const atBottom = node.scrollHeight - node.scrollTop - node.clientHeight < 80;
    pinned.current = atBottom;
    setShowPill(!atBottom && state.messages.length > 0);
  };

  const toLatest = () => {
    const node = scroller.current;
    if (!node) return;
    pinned.current = true;
    setShowPill(false);
    node.scrollTo({ top: node.scrollHeight, behavior: 'smooth' });
  };

  const lastIndex = state.messages.length - 1;
  const empty = state.messages.length === 0;

  return (
    <div className="relative flex min-h-0 flex-1 flex-col">
      {state.bootError && (
        <div className="flex items-center gap-2 border-b border-rule-soft bg-panel/60 px-4 py-1.5 font-mono text-[10.5px] uppercase tracking-[0.12em] text-warn">
          <span className="truncate normal-case tracking-normal">
            history unavailable · {state.bootError}
          </span>
          <button
            type="button"
            onClick={() => void boot()}
            className="ml-auto shrink-0 rounded-sm border border-rule px-1.5 py-0.5 transition-colors hover:border-accent/60 hover:text-ink"
          >
            retry
          </button>
        </div>
      )}
      <div
        ref={scroller}
        onScroll={onScroll}
        className="min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-6"
      >
        <div className="mx-auto flex w-full max-w-[768px] flex-col gap-5">
          {empty ? (
            state.loading ? null : (
              <div className="rise mt-[10vh] max-w-[46ch]">
                <h1 className="font-serif text-[34px] leading-[1.15] text-ink">
                  Model the system,
                  <br />
                  <em className="text-accent">not the markup.</em>
                </h1>
                <p className="mt-3 text-[14.5px] leading-relaxed text-ink-dim">
                  Ask for a SysML v2 model and it is generated, grammar-checked and
                  repaired before it reaches you; ask anything else and the answer
                  comes back grounded in the documents you attach.
                </p>
                <ul className="mt-6 flex flex-col gap-2">
                  {SAMPLES.map((sample) => (
                    <li key={sample}>
                      <button
                        type="button"
                        onClick={() => void send(sample)}
                        className="flex items-start gap-2 text-left font-mono text-[12px] text-ink-dim transition-colors hover:text-ink"
                      >
                        <span className="mt-[7px] size-1 shrink-0 bg-accent/70" />
                        <span>{sample}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )
          ) : (
            <div
              role="log"
              aria-live="off"
              aria-label="Conversation"
              className="flex flex-col gap-5"
            >
              {state.agentSteps.length > 0 && <ToolTraceAccordion />}
              {state.messages.map((message, index) => (
                <Message
                  key={message.id}
                  message={message}
                  streaming={state.running && index === lastIndex}
                  liveModel={index === lastIndex ? state.artifact : undefined}
                />
              ))}
              {state.running && <ValidationReport />}
            </div>
          )}
        </div>
      </div>

      <div aria-live="polite" className="sr-only">
        {state.running ? 'Agent is replying' : ''}
      </div>

      {showPill && (
        <button
          type="button"
          onClick={toLatest}
          className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-1.5 rounded-full border border-rule bg-panel px-3 py-1 font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-dim shadow-lg transition-colors hover:border-accent/60 hover:text-ink"
          title="Jump to latest"
        >
          <ArrowDown className="size-3" />
          latest
        </button>
      )}
    </div>
  );
}
