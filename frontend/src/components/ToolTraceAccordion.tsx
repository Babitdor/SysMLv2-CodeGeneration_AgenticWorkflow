import { useState } from 'react';
import { Check, ChevronDown, Circle, Loader2, X } from 'lucide-react';
import { useWorkspace } from '../state/WorkspaceContext';
import type { ToolStep } from '../lib/types';

const LABELS: Record<string, string> = {
  routing: 'Routing intent',
  generate_sysml: 'Generating SysML',
  validate_sysml: 'Validating syntax',
  read_file: 'Reading file',
  grep: 'Searching text',
  glob: 'Finding files',
  ls: 'Listing files',
  retrieve_documents: 'Retrieving documents',
  write_file: 'Writing file',
  edit_file: 'Editing file',
  bash: 'Running command',
  web_search: 'Searching the web',
  fetch_url: 'Fetching page',
};

/** Named reads are counted apart from actions in the header line. */
const READS = new Set(['read_file', 'grep', 'ls', 'glob', 'retrieve_documents']);

function describe(step: ToolStep): string {
  const base = LABELS[step.name] ?? step.name;
  if (step.name === 'routing') return base + (step.detail ? ' → ' + step.detail : '');
  const attempt = step.attempt ? ' (attempt ' + step.attempt + ')' : '';
  return base + attempt;
}

function duration(ms: number): string {
  return ms >= 1000 ? (ms / 1000).toFixed(1) + ' s' : ms + ' ms';
}

/** 8421 -> "8 421": the char counts read better grouped than as one run. */
const group = (count: number) => String(count).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');

function StatusIcon({ status }: { status: ToolStep['status'] }) {
  if (status === 'start') return <Loader2 className="size-3 animate-spin text-accent" />;
  if (status === 'done') return <Check className="size-3 text-ok" />;
  if (status === 'failed') return <X className="size-3 text-bad" />;
  return <Circle className="size-2 text-ink-dim" />;
}

function TraceRow({
  step,
  open,
  onToggle,
}: {
  step: ToolStep;
  open: boolean;
  onToggle: () => void;
}) {
  // Only a row with a payload is a disclosure; a pipeline stage stays one line.
  const expandable = Boolean(step.args || step.output);
  const body = (
    <>
      <StatusIcon status={step.status} />
      <span className={step.status === 'failed' ? 'text-bad/90' : 'text-ink/80'}>
        {describe(step)}
      </span>
      {step.summary && (
        <span className="min-w-0 truncate text-ink-dim/80">{step.summary}</span>
      )}
      {step.ms !== undefined && (
        <span className="ml-auto shrink-0 pl-2 text-[10.5px] text-ink-dim/60">
          {duration(step.ms)}
        </span>
      )}
    </>
  );

  return (
    <li>
      {expandable ? (
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className="flex w-full items-center gap-2 py-0.5 text-left font-mono text-[11.5px]"
        >
          <ChevronDown
            className={
              'size-3 shrink-0 text-ink-dim transition-transform ' + (open ? '' : '-rotate-90')
            }
          />
          {body}
        </button>
      ) : (
        <div className="flex items-center gap-2 py-0.5 font-mono text-[11.5px] text-ink-dim">
          {body}
        </div>
      )}
      {expandable && open && (
        <div className="pb-1.5 pl-7 pr-1 pt-1">
          {step.args && (
            <pre className="mb-1 max-h-32 overflow-auto whitespace-pre-wrap break-words border border-rule-soft bg-panel/40 px-2 py-1 font-mono text-[11px] leading-relaxed text-ink-dim">
              {step.args}
            </pre>
          )}
          {step.output && (
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words border border-rule-soft bg-panel/40 px-2 py-1 font-mono text-[11px] leading-relaxed text-ink/80">
              {step.output}
            </pre>
          )}
          {step.truncated && (
            <p className="pt-1 font-mono text-[10.5px] text-ink-dim/70">
              truncated{step.chars !== undefined ? ' · ' + group(step.chars) + ' chars' : ''}
            </p>
          )}
        </div>
      )}
    </li>
  );
}

export function ToolTraceAccordion() {
  const { state } = useWorkspace();
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);

  const reads = state.agentSteps.filter((step) => READS.has(step.name)).length;
  const actions = state.agentSteps.length - reads;

  return (
    <div className="border border-rule-soft bg-panel/40">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left"
      >
        <span className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim">
          <ChevronDown
            className={'size-3 transition-transform ' + (open ? '' : '-rotate-90')}
          />
          tool trace · {state.agentSteps.length}
          {actions > 0 && ' · ' + actions + ' ops'}
          {reads > 0 && ' · ' + reads + ' reads'}
        </span>
        <span className="flex items-center gap-1.5">
          {state.agentSteps.slice(-3).map((step) => (
            <StatusIcon key={step.id} status={step.status} />
          ))}
        </span>
      </button>
      {open && (
        <ol className="border-t border-rule-soft px-3 py-2">
          {state.agentSteps.map((step) => (
            <TraceRow
              key={step.id}
              step={step}
              open={expanded === step.id}
              onToggle={() => setExpanded((current) => (current === step.id ? null : step.id))}
            />
          ))}
        </ol>
      )}
    </div>
  );
}
