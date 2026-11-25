import { useDeferredValue, useMemo } from 'react';
import { highlight, TOKEN_CLASS } from '../lib/markdown';
import { useMediaQuery } from '../lib/useMediaQuery';
import { useWorkspace } from '../state/WorkspaceContext';
import { Markdown } from './Markdown';

const SPLIT = '(min-width: 1100px)';

/**
 * The canvas body: the source and its rendered preview, both on screen at once.
 * Below 1100px they stack rather than collapse into a toggle — the requirement
 * is "never a mode switch", and a hidden pane is still a mode switch.
 *
 * The preview reads a deferred copy of the draft: React renders the lagging
 * value and drops intermediate ones, so typing stays responsive on a
 * 40k-character document without a debounce timer of our own.
 */
export function ArtifactEditor() {
  const { state, editCanvas } = useWorkspace();
  const wide = useMediaQuery(SPLIT);
  const draft = state.canvas?.draft ?? '';
  const deferred = useDeferredValue(draft);
  const tokens = useMemo(() => highlight(deferred, 'sysml'), [deferred]);

  if (!state.canvas) return null;
  const { kind } = state.canvas;

  return (
    <div className={'flex min-h-0 flex-1 ' + (wide ? 'flex-row' : 'flex-col')}>
      <textarea
        value={draft}
        onChange={(event) => editCanvas(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== 'Tab') return;
          // Tab indents here; moving focus would lose the caret mid-model.
          event.preventDefault();
          const node = event.currentTarget;
          const { selectionStart: start, selectionEnd: end } = node;
          editCanvas(draft.slice(0, start) + '  ' + draft.slice(end));
          requestAnimationFrame(() => node.setSelectionRange(start + 2, start + 2));
        }}
        spellCheck={false}
        aria-label="Artifact source"
        className="min-h-0 flex-1 resize-none overflow-auto bg-transparent px-3 py-3 font-mono text-[11.5px] leading-[1.65] whitespace-pre text-ink/90 outline-none"
      />
      <div
        className={
          'min-h-0 flex-1 overflow-auto border-rule px-4 py-3 ' +
          (wide ? 'border-l' : 'border-t')
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
