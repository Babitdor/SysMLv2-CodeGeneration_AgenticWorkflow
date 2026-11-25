import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronRight, Loader2, X } from 'lucide-react';
import { analyzeImpact, loadModelIndex } from '../lib/model';
import type { ImpactReport, ModelElement, ModelIssue, ModelReport, ModelRow } from '../lib/model';
import { useMediaQuery } from '../lib/useMediaQuery';
import { useWorkspace } from '../state/WorkspaceContext';

const INLINE = '(min-width: 1100px)';
const OUTLINE_ROWS = 400;
// One shared empty list: a fresh [] would re-run every memo keyed on `elements`.
const NO_ELEMENTS: ModelElement[] = [];

// Review rules are absence-of-evidence hints; label them in the model's words.
const RULE_LABELS: Record<string, string> = {
  reference: 'unresolved',
  traceability: 'no satisfy link',
  consistency: 'possibly disconnected',
  syntax: 'syntax',
};

const severityTone: Record<string, string> = {
  error: 'text-bad/90',
  warning: 'text-warn/90',
  info: 'text-ink-dim',
};

function kindBadge(kind: string) {
  return kind.replace(/_/g, ' ').toUpperCase();
}

/** One element row: kind badge, name, line. */
function OutlineRow({
  element,
  active,
  indent,
  onClick,
}: {
  element: ModelElement;
  active: boolean;
  indent: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full items-baseline gap-2 px-2 py-[3px] text-left font-mono text-[11.5px] transition-colors ${
        active ? 'bg-rule-soft text-ink' : 'text-ink-dim hover:bg-rule-soft/60'
      }`}
      style={{ paddingLeft: `${8 + Math.min(indent, 8) * 12}px` }}
    >
      <span className="w-[13ch] shrink-0 truncate text-[9.5px] tracking-[0.08em] text-ink-dim/70">
        {kindBadge(element.kind)}
      </span>
      <span className="min-w-0 flex-1 truncate">{element.name}</span>
      <span className="shrink-0 text-[10px] text-ink-dim/50">{element.line}</span>
    </button>
  );
}

function IssueGroup({ rule, issues }: { rule: string; issues: ModelIssue[] }) {
  const [open, setOpen] = useState(issues.length <= 3);
  return (
    <div className="border-b border-rule-soft last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-1.5 px-3 py-1.5 text-left"
        aria-expanded={open}
      >
        {open ? (
          <ChevronDown className="size-3 shrink-0 text-ink-dim" />
        ) : (
          <ChevronRight className="size-3 shrink-0 text-ink-dim" />
        )}
        <span className="label">{RULE_LABELS[rule] ?? rule}</span>
        <span className="font-mono text-[10px] text-ink-dim/70">{issues.length}</span>
      </button>
      {open && (
        <ul className="pb-1.5">
          {issues.map((issue, index) => (
            <li
              key={`${issue.rule}-${issue.line}-${index}`}
              className="flex gap-2 px-5 py-[2px] font-mono text-[11px] leading-snug"
            >
              <span className="w-[4ch] shrink-0 text-right text-ink-dim/50">{issue.line}</span>
              <span className={severityTone[issue.severity] ?? 'text-ink-dim'}>{issue.message}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ImpactList({ title, rows }: { title: string; rows: ModelRow[] }) {
  if (rows.length === 0) return null;
  return (
    <div className="mt-1">
      <div className="label">{title} · {rows.length}</div>
      <ul>
        {rows.map((row) => (
          <li
            key={`${title}-${row.id}`}
            className="flex gap-2 px-1 py-[1px] font-mono text-[11px] text-ink-dim"
          >
            <span className="min-w-0 flex-1 truncate">{row.name}</span>
            {row.via && <span className="shrink-0 text-[9.5px] text-ink-dim/60">{row.via}</span>}
            <span className="w-[4ch] shrink-0 text-right text-[10px] text-ink-dim/50">{row.line}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Right-hand panel: what the model is made of, what it fails to link, and
 *  what a change to one element would reach. */
export function ModelPanel() {
  const { state, closeModel } = useWorkspace();
  const inline = useMediaQuery(INLINE);
  const code = state.model?.code ?? '';

  const [loading, setLoading] = useState(Boolean(code.trim()));
  const [report, setReport] = useState<ModelReport | null>(null);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState<string | null>(null);
  const [impact, setImpact] = useState<ImpactReport | null>(null);
  const [impactLoading, setImpactLoading] = useState(false);
  const [impactError, setImpactError] = useState('');
  const impactAbort = useRef<AbortController | null>(null);

  // One index per model: opening the panel asks once, closing aborts the walk.
  useEffect(() => {
    if (!code.trim()) return;
    const controller = new AbortController();
    loadModelIndex(code, controller.signal)
      .then((reply) => {
        if (controller.signal.aborted) return;
        setReport(reply);
        setLoading(false);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setLoading(false);
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => controller.abort();
  }, [code]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') closeModel();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [closeModel]);

  useEffect(() => () => impactAbort.current?.abort(), []);

  const elements = report?.elements ?? NO_ELEMENTS;
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return elements;
    return elements.filter(
      (element) =>
        element.name.toLowerCase().includes(needle) ||
        element.id.toLowerCase().includes(needle) ||
        element.kind.includes(needle),
    );
  }, [elements, query]);

  // Outline indent: how many parents an element has, ignoring any parent cycle.
  const depth = useMemo(() => {
    const byId = new Map(elements.map((element) => [element.id, element]));
    const levels = new Map<string, number>();
    for (const element of elements) {
      let level = 0;
      let chain = element.parent;
      const walk = new Set<string>();
      while (chain && !walk.has(chain)) {
        walk.add(chain);
        const parent = byId.get(chain);
        if (!parent) break;
        level += 1;
        chain = parent.parent;
      }
      levels.set(element.id, level);
    }
    return levels;
  }, [elements]);

  const groups = useMemo(() => {
    const byRule = new Map<string, ModelIssue[]>();
    for (const issue of report?.review ?? []) {
      const bucket = byRule.get(issue.rule) ?? [];
      bucket.push(issue);
      byRule.set(issue.rule, bucket);
    }
    return [...byRule.entries()];
  }, [report]);

  const element = selected ? elements.find((item) => item.id === selected) ?? null : null;
  const edges = useMemo(() => {
    if (!element || !report) return [];
    return report.edges
      .filter((edge) => edge.source === element.id || edge.target === element.id)
      .slice(0, 24);
  }, [element, report]);

  const select = (id: string) => {
    setSelected(id === selected ? null : id);
    setImpact(null);
    setImpactError('');
  };

  const runImpact = () => {
    if (!element) return;
    impactAbort.current?.abort();
    const controller = new AbortController();
    impactAbort.current = controller;
    setImpactLoading(true);
    setImpactError('');
    analyzeImpact(code, element.id, controller.signal)
      .then((reply) => {
        if (controller.signal.aborted) return;
        setImpact(reply);
        setImpactLoading(false);
      })
      .catch((err) => {
        if (controller.signal.aborted) return;
        setImpactLoading(false);
        setImpactError(err instanceof Error ? err.message : String(err));
      });
  };

  const panel = (
    <aside
      aria-label="Model index"
      className={
        inline
          ? 'slide-in flex w-[440px] shrink-0 flex-col border-l border-rule bg-bg-2/60 xl:w-[520px]'
          : 'slide-in flex h-full w-full flex-col border-l border-rule bg-bg sm:w-[520px]'
      }
    >
      <header className="flex items-center gap-2 border-b border-rule px-3 py-2">
        <span className="label">model</span>
        {report && (
          <span className="font-mono text-[10.5px] tracking-[0.1em] text-ink-dim">
            {report.revision}
          </span>
        )}
        {report && report.validator !== 'antlr' && (
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-bad/80">
            {report.validator}
          </span>
        )}
        {report?.truncated && (
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-warn/80">
            capped
          </span>
        )}
        <button
          type="button"
          onClick={closeModel}
          className="ml-auto rounded-sm p-1 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
          title="Close model"
          aria-label="Close model"
        >
          <X className="size-3.5" />
        </button>
      </header>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-rule-soft px-3 py-2 font-mono text-[10.5px] text-ink-dim">
        <span>{report?.stats.elements ?? 0} elements</span>
        <span>{report?.stats.edges ?? 0} edges</span>
        <span>{report?.stats.requirements ?? 0} requirements</span>
        <span className={report && report.stats.unresolved > 0 ? 'text-bad/80' : ''}>
          {report?.stats.unresolved ?? 0} unresolved
        </span>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="filter"
          aria-label="Filter elements"
          className="ml-auto w-24 rounded-sm border border-rule bg-bg px-1.5 py-0.5 font-mono text-[10.5px] text-ink placeholder:text-ink-dim/60"
        />
      </div>

      <div className="min-h-0 flex-1 overflow-auto py-1">
        {loading ? (
          <span className="flex items-center gap-2 px-3 py-2 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim">
            <Loader2 className="size-3.5 animate-spin" />
            indexing
          </span>
        ) : error ? (
          <p className="whitespace-pre-wrap px-3 py-2 font-mono text-[11.5px] leading-relaxed text-bad/90">
            {error}
          </p>
        ) : rows.length === 0 ? (
          <p className="px-3 py-2 font-mono text-[11.5px] text-ink-dim">
            {elements.length === 0 ? 'nothing indexed' : 'no element matches the filter'}
          </p>
        ) : (
          <>
            {rows.slice(0, OUTLINE_ROWS).map((item) => (
              <OutlineRow
                key={item.id}
                element={item}
                active={item.id === selected}
                indent={depth.get(item.id) ?? 0}
                onClick={() => select(item.id)}
              />
            ))}
            {rows.length > OUTLINE_ROWS && (
              <p className="px-3 py-1.5 font-mono text-[10.5px] text-ink-dim/70">
                {rows.length - OUTLINE_ROWS} more
              </p>
            )}
          </>
        )}
      </div>

      <div className="max-h-[52%] shrink-0 overflow-auto border-t border-rule">
        {element ? (
          <div className="px-3 py-2">
            <div className="flex items-baseline gap-2">
              <span className="min-w-0 flex-1 truncate font-mono text-[12px] text-ink">
                {element.id}
              </span>
              <span className="shrink-0 font-mono text-[10px] text-ink-dim">
                {element.kind} · {element.line}
              </span>
            </div>
            {edges.length > 0 && (
              <ul className="mt-1">
                {edges.map((edge, index) => (
                  <li
                    key={`${edge.kind}-${edge.line}-${index}`}
                    className="flex gap-1.5 font-mono text-[10.5px] text-ink-dim"
                  >
                    <span className="shrink-0 text-ink-dim/60">{edge.kind}</span>
                    <span className="min-w-0 flex-1 truncate">
                      {edge.source === element.id ? 'to' : 'from'}{' '}
                      {edge.source === element.id ? edge.target : edge.source}
                    </span>
                  </li>
                ))}
              </ul>
            )}
            <div className="mt-1.5 flex items-center gap-2">
              <button
                type="button"
                onClick={runImpact}
                disabled={impactLoading}
                className="rounded-sm border border-rule bg-bg px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] text-accent transition-colors hover:text-ink disabled:opacity-50"
              >
                {impactLoading ? 'walking' : 'impact'}
              </button>
              {impactError && (
                <span className="min-w-0 truncate font-mono text-[10.5px] text-bad/90">
                  {impactError}
                </span>
              )}
            </div>
            {impact && (
              <div className="mt-1">
                <ImpactList title="direct" rows={impact.direct} />
                <ImpactList title="transitive" rows={impact.transitive} />
                <ImpactList title="requirements reached" rows={impact.requirements} />
                {impact.direct.length + impact.transitive.length === 0 && (
                  <p className="font-mono text-[10.5px] text-ink-dim">
                    nothing in this model refers to it
                  </p>
                )}
              </div>
            )}
          </div>
        ) : null}

        {report && groups.length > 0 && (
          <div className="border-t border-rule-soft">
            {groups.map(([rule, issues]) => (
              <IssueGroup key={rule} rule={rule} issues={issues} />
            ))}
          </div>
        )}
        {report && groups.length === 0 && (
          <p className="px-3 py-2 font-mono text-[11px] text-ok/80">no findings</p>
        )}
      </div>
    </aside>
  );

  if (inline) return panel;
  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        aria-label="Close model"
        onClick={closeModel}
        className="absolute inset-0 bg-bg/70"
      />
      <div className="relative z-10 h-full w-full sm:w-[520px]">{panel}</div>
    </div>
  );
}
