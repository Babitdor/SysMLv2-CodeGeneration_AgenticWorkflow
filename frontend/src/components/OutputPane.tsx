import { useMemo, useState } from 'react';
import { approveCode } from '../lib/api';
import { highlight, TOKEN_CLASS } from '../lib/markdown';
import { useMediaQuery } from '../lib/useMediaQuery';
import type { ValidationStatus } from '../lib/types';
import { useWorkspace } from '../state/WorkspaceContext';

const INLINE = '(min-width: 1100px)';
const GHOST =
  'rounded-sm border border-rule px-2 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-accent/60 hover:text-accent disabled:opacity-40 disabled:hover:border-rule disabled:hover:text-ink-dim';
const BADGE: Record<ValidationStatus | 'pending', string> = {
  valid: 'border-ok/40 text-ok',
  invalid: 'border-bad/40 text-bad',
  error: 'border-bad/40 text-bad',
  pending: 'border-rule text-ink-dim',
};

/**
 * The right-hand output pane: the turn's SysML, its verdict, and the two things
 * you can do with code that parses — copy it, or store it as an approved
 * solution. Hidden below 1100px, where the code is already in the transcript.
 */
export function OutputPane() {
  const { state } = useWorkspace();
  const inline = useMediaQuery(INLINE);
  const [verdict, setVerdict] = useState('');
  const [saving, setSaving] = useState(false);
  const [copied, setCopied] = useState(false);

  // Live stream first, then the turn's stored model so Approve survives a reload.
  const turn = [...state.messages]
    .reverse()
    .find((message) => message.role === 'assistant' && (message.model || message.validation));
  const code = state.artifact || turn?.model || '';
  const validation = state.validationLogs[state.validationLogs.length - 1] ?? turn?.validation;
  // While a turn streams, the verdict on screen is about the previous model.
  const status: ValidationStatus | 'pending' = state.running
    ? 'pending'
    : (validation?.status ?? 'pending');
  const prompt =
    [...state.messages].reverse().find((message) => message.role === 'user')?.text ?? '';
  const tokens = useMemo(() => highlight(code, 'sysml'), [code]);

  if (!inline || !code.trim()) return null;

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  const approve = async () => {
    setSaving(true);
    try {
      const reply = await approveCode(code, prompt);
      setVerdict(
        reply.stored
          ? `stored in approved_solutions · ${reply.approved_entries ?? 0} entries`
          : (reply.validation ?? 'rejected'),
      );
    } catch (err) {
      setVerdict(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <aside className="flex w-[440px] shrink-0 flex-col border-l border-rule bg-panel/40">
      <div className="flex items-center gap-2 border-b border-rule px-3 py-2">
        <span className="label">output · sysml</span>
        <span
          className={
            'rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] ' +
            BADGE[status]
          }
        >
          {status}
        </span>
        <span className="flex-1" />
        <button type="button" className={GHOST} onClick={() => void copy()}>
          {copied ? 'copied' : 'copy'}
        </button>
        <button
          type="button"
          className={GHOST}
          onClick={() => void approve()}
          disabled={status !== 'valid' || saving}
          title={status === 'valid' ? 'Store as an approved solution' : 'Needs a valid model first'}
        >
          {saving ? 'saving…' : 'approve → kb'}
        </button>
      </div>

      <pre className="min-h-0 flex-1 overflow-auto px-3 py-3 font-mono text-[11.5px] leading-[1.65] whitespace-pre text-ink/90">
        {tokens.map((token, index) =>
          token.kind === 'text' ? (
            <span key={index}>{token.text}</span>
          ) : (
            <span key={index} className={TOKEN_CLASS[token.kind]}>
              {token.text}
            </span>
          ),
        )}
      </pre>

      {(verdict || validation?.summary) && (
        <p className="border-t border-rule px-3 py-2 font-mono text-[10.5px] leading-relaxed text-ink-dim">
          {verdict || validation?.summary}
        </p>
      )}
    </aside>
  );
}
