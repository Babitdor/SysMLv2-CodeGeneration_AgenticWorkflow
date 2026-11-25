import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { Check, Copy } from 'lucide-react';
import { highlight, parseInline, parseMarkdown, TOKEN_CLASS } from '../lib/markdown';
import type { MdInline } from '../lib/markdown';

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Older browsers and non-secure origins: a hidden textarea still copies.
    try {
      const node = document.createElement('textarea');
      node.value = text;
      node.setAttribute('readonly', '');
      node.style.position = 'fixed';
      node.style.opacity = '0';
      document.body.appendChild(node);
      node.select();
      const ok = document.execCommand('copy');
      document.body.removeChild(node);
      return ok;
    } catch {
      return false;
    }
  }
}

function Inline({ text }: { text: string }) {
  const nodes = useMemo(() => parseInline(text), [text]);
  return (
    <>
      {nodes.map((node: MdInline, index) => {
        switch (node.kind) {
          case 'code':
            return (
              <code
                key={index}
                className="rounded-sm bg-rule-soft px-1 py-0.5 font-mono text-[0.92em] text-ink"
              >
                {node.text}
              </code>
            );
          case 'strong':
            return (
              <strong key={index} className="font-semibold text-ink">
                {node.text}
              </strong>
            );
          case 'em':
            return <em key={index}>{node.text}</em>;
          case 'link':
            return (
              <a
                key={index}
                href={node.href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent"
              >
                {node.text}
              </a>
            );
          default:
            return <span key={index}>{node.text}</span>;
        }
      })}
    </>
  );
}

interface CodeBlockProps {
  code: string;
  lang?: string;
  /** Extra actions for the head row (download, approve, ...). */
  actions?: ReactNode;
}

export function CodeBlock({ code, lang = 'text', actions }: CodeBlockProps) {
  const tokens = useMemo(() => highlight(code, lang), [code, lang]);
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    if (await copyText(code)) {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    }
  };

  return (
    <div className="overflow-hidden rounded-lg border border-rule bg-bg-2/70">
      <div className="flex items-center justify-between gap-2 border-b border-rule-soft bg-panel/60 px-3 py-1.5">
        <span className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-ink-dim">
          {lang || 'text'}
        </span>
        <span className="flex items-center gap-1">
          {actions}
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1 rounded-sm px-1.5 py-0.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
            title="Copy code"
            aria-label="Copy code"
          >
            {copied ? <Check className="size-3 text-ok" /> : <Copy className="size-3" />}
            {copied ? 'copied' : 'copy'}
          </button>
        </span>
      </div>
      <pre className="overflow-x-auto px-3 py-2.5 font-mono text-[12.5px] leading-[1.6]">
        <code>
          {tokens.map((token, index) =>
            token.kind === 'text' ? (
              <span key={index}>{token.text}</span>
            ) : (
              <span key={index} className={TOKEN_CLASS[token.kind]}>
                {token.text}
              </span>
            ),
          )}
        </code>
      </pre>
    </div>
  );
}

const HEADING_CLASS: Record<number, string> = {
  1: 'mt-5 mb-2 font-serif text-[26px] leading-tight text-ink',
  2: 'mt-5 mb-2 text-[19px] font-semibold leading-snug text-ink',
  3: 'mt-4 mb-1.5 text-[16px] font-semibold leading-snug text-ink',
  4: 'mt-4 mb-1.5 font-mono text-[12px] uppercase tracking-[0.12em] text-ink-dim',
};

interface MarkdownProps {
  text: string;
  /** Extra head-row actions for matching fenced blocks (the SysML model). */
  codeActions?: (block: { code: string; lang: string }) => ReactNode;
}

export function Markdown({ text, codeActions }: MarkdownProps) {
  const blocks = useMemo(() => parseMarkdown(text), [text]);

  return (
    <div className="md flex flex-col gap-3 text-[15px] leading-relaxed text-ink/90">
      {blocks.map((block, index) => {
        switch (block.kind) {
          case 'code':
            return (
              <CodeBlock
                key={index}
                code={block.code}
                lang={block.lang}
                actions={codeActions?.({ code: block.code, lang: block.lang })}
              />
            );
          case 'heading': {
            const Tag = ('h' + block.level) as 'h1' | 'h2' | 'h3' | 'h4';
            return (
              <Tag key={index} className={HEADING_CLASS[block.level]}>
                <Inline text={block.text} />
              </Tag>
            );
          }
          case 'list': {
            const Tag = block.ordered ? 'ol' : 'ul';
            return (
              <Tag
                key={index}
                className={
                  'flex flex-col gap-1.5 pl-5 ' +
                  (block.ordered ? 'list-decimal' : 'list-disc') +
                  ' marker:text-accent/70'
                }
              >
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex} className="pl-0.5">
                    <Inline text={item} />
                  </li>
                ))}
              </Tag>
            );
          }
          case 'quote':
            return (
              <blockquote
                key={index}
                className="border-l-2 border-rule pl-3 text-ink-dim"
              >
                <Inline text={block.text} />
              </blockquote>
            );
          case 'hr':
            return <hr key={index} className="my-1 border-0 border-t border-rule" />;
          default:
            return (
              <p key={index}>
                <Inline text={block.text} />
              </p>
            );
        }
      })}
    </div>
  );
}
