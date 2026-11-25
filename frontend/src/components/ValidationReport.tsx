import { useWorkspace } from '../state/WorkspaceContext';
import type { ValidationResult } from '../lib/types';

function attemptLabel(log: ValidationResult): string {
  if (log.status === 'valid') return 'attempt ' + log.attempt + ' · passed';
  if (log.status === 'error') return 'attempt ' + log.attempt + ' · validator error';
  return 'attempt ' + log.attempt + ' · failed';
}

function tone(log: ValidationResult): string {
  if (log.status === 'valid') return 'text-ok';
  if (log.status === 'error') return 'text-warn';
  return 'text-bad';
}

interface Props {
  /** Defaults to the turn in flight; a stored turn passes its own verdict. */
  logs?: ValidationResult[];
}

/** Grammar results for the turn, folded into one strip under the model. */
export function ValidationReport({ logs }: Props = {}) {
  const { state } = useWorkspace();
  const shown = logs ?? state.validationLogs;
  if (shown.length === 0) return null;

  const latest = shown[shown.length - 1];
  const summary =
    latest.status === 'valid'
      ? 'valid · ' + attemptLabel(latest)
      : latest.status === 'error'
        ? 'validator unavailable · ' + attemptLabel(latest)
        : latest.diagnostics.length + ' problem(s) · ' + attemptLabel(latest);

  return (
    <details className="group rounded-lg border border-rule bg-panel/40">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-1.5 font-mono text-[10.5px] uppercase tracking-[0.12em]">
        <span className={tone(latest)}>{summary}</span>
        <span className="ml-auto text-ink-dim/60 group-open:hidden">open</span>
      </summary>
      <div className="border-t border-rule-soft">
        {shown.map((log, index) => (
          <section key={log.attempt + '-' + index} className="px-3 py-2">
            <header className="mb-1.5 flex items-center justify-between gap-2">
              <span
                className={
                  'font-mono text-[10.5px] uppercase tracking-[0.12em] ' + tone(log)
                }
              >
                {attemptLabel(log)}
              </span>
              {log.source === 'structural' && (
                <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-warn/80">
                  structural only
                </span>
              )}
            </header>
            {log.diagnostics.length === 0 ? (
              <p className="whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-ink-dim">
                {log.summary || 'No problems reported.'}
              </p>
            ) : (
              <ul className="flex flex-col gap-1">
                {log.diagnostics.map((diagnostic, diagnosticIndex) => (
                  <li
                    key={diagnosticIndex}
                    className="flex items-baseline gap-2 font-mono text-[11.5px] leading-relaxed"
                  >
                    <span className="shrink-0 text-ink-dim/80">
                      {diagnostic.line}:{diagnostic.column}
                    </span>
                    <span
                      className={
                        diagnostic.severity === 'error' ? 'text-bad/90' : 'text-ink/80'
                      }
                    >
                      {diagnostic.message}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        ))}
      </div>
    </details>
  );
}
