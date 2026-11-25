import { useDeferredValue, useMemo } from 'react';
import { highlight, TOKEN_CLASS } from '../lib/markdown';
import { useWorkspace } from '../state/WorkspaceContext';
import { Markdown } from './Markdown';


/** The preview follows the draft or the current agent proposal without blocking typing. */
export function ArtifactEditor({ proposed }: { proposed?: string }) {
  const { state, editCanvas } = useWorkspace();
  const draft = state.canvas?.draft ?? '';
  const deferred = useDeferredValue(proposed ?? draft);
  const tokens = useMemo(() => highlight(deferred, 'sysml'), [deferred]);

  if (!state.canvas) return null;
  const { kind } = state.canvas;

  return (
    <div className="flex min-h-0 flex-1 flex-col @min-[38rem]:flex-row">
      <textarea
        value={draft}
        onChange={(event) => editCanvas(event.target.value)}
        spellCheck={false}
        aria-label="Artifact source"
        className="min-h-0 min-w-0 flex-1 resize-none overflow-auto bg-transparent px-4 py-4 font-mono text-xs leading-relaxed whitespace-pre text-ink focus-visible:outline-1 focus-visible:outline-accent"
      />
      <div
        aria-label={proposed === undefined ? 'Artifact preview' : 'Proposed artifact preview'}
        className={
          'min-h-0 min-w-0 flex-1 overflow-auto border-t border-rule px-4 py-4 @min-[38rem]:border-t-0 @min-[38rem]:border-l'
        }
      >
        {kind === 'markdown' ? (
          <Markdown text={deferred} />
        ) : (
          <pre className="font-mono text-[11.5px] leading-[1.65] whitespace-pre text-ink/90">
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
        )}
      </div>
    </div>
  );
}
