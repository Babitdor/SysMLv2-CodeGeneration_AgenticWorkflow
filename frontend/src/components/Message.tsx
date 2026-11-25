import { useState } from 'react';
import { Download, Network, Pencil, ShieldCheck } from 'lucide-react';
import { approveCode } from '../lib/api';
import { useWorkspace } from '../state/WorkspaceContext';
import { useAccount } from '../lib/account';
import type { Message as MessageModel, ValidationResult } from '../lib/types';
import { CodeBlock, Markdown } from './Markdown';
import { ValidationReport } from './ValidationReport';

function downloadName(code: string): string {
  const match = code.match(/package\s+['"]?([A-Za-z_][\w]*)/);
  const base = match ? match[1] : 'model-' + Date.now();
  return base.endsWith('.sysml') ? base : base + '.sysml';
}

function download(code: string) {
  const blob = new Blob([code], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = downloadName(code);
  anchor.click();
  URL.revokeObjectURL(url);
}

/** Model card actions: copy lives in CodeBlock, the rest live here. */
function ModelActions({
  code,
  validation,
  artifactId,
  task,
}: {
  code: string;
  validation?: ValidationResult;
  /** The stored artifact for this turn; absent on turns from before the canvas. */
  artifactId?: string;
  task?: string;
}) {
  const { visualize, openArtifact, openDraft } = useWorkspace();
  const account = useAccount();
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const openCanvas = () => {
    if (artifactId) {
      void openArtifact(artifactId).catch(() => undefined);
      return;
    }
    // A turn stored before artifacts existed still opens, just unsaved.
    openDraft({ title: 'model', kind: 'code', content: code });
  };

  const approve = async () => {
    setSaving(true);
    setNotice('storing…');
    try {
      const result = await approveCode(code, task || 'Manual SysML model approval');
      setNotice(
        result.stored ? 'stored in the approved-solutions RAG' : (result.validation ?? 'not stored'),
      );
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      {notice && (
        <span className="font-mono text-[10px] normal-case tracking-normal text-ink-dim">
          {notice}
        </span>
      )}
      <button
        type="button"
        onClick={() => visualize(code)}
        className="rounded-sm p-1 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
        title="Visualize"
        aria-label="Visualize model"
      >
        <Network className="size-3" />
      </button>
      <button
        type="button"
        onClick={openCanvas}
        className="rounded-sm p-1 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
        title="Open in canvas"
        aria-label="Open model in canvas"
      >
        <Pencil className="size-3" />
      </button>
      <button
        type="button"
        onClick={() => download(code)}
        className="rounded-sm p-1 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ink"
        title="Download .sysml"
        aria-label="Download model"
      >
        <Download className="size-3" />
      </button>
      {account?.role === 'admin' && <button
        type="button"
        onClick={() => void approve()}
        disabled={saving || validation?.ok === false}
        className="rounded-sm p-1 text-ink-dim transition-colors hover:bg-rule-soft hover:text-ok disabled:opacity-30"
        title="Store as an approved solution"
        aria-label="Approve model"
      >
        <ShieldCheck className="size-3" />
      </button>}
    </>
  );
}

interface Props {
  message: MessageModel;
  streaming: boolean;
  /** The model of the turn in flight; once settled it lives on the message. */
  liveModel?: string;
  task?: string;
}

export function Message({ message, streaming, liveModel, task }: Props) {
  const { openDraft, openArtifact } = useWorkspace();

  if (message.role === 'user') {
    return (
      <article className="rise flex justify-end" aria-label="Your message">
        <div className="max-w-[85%] whitespace-pre-wrap break-words rounded-2xl bg-panel px-3.5 py-2 text-[15px] leading-relaxed text-ink">
          {message.text}
        </div>
      </article>
    );
  }

  // New turns carry the model as a field; a legacy thread only has the fence,
  // which `codeActions` still decorates below. Streaming shows it live.
  const model = message.model?.trim() || (streaming ? liveModel?.trim() : '') || '';

  return (
    <article className="rise flex flex-col gap-3" aria-label="Agent reply">
      {message.text && (
        <Markdown
          text={message.text}
          codeActions={({ code, lang }) =>
            lang === 'sysml' ? (
              <ModelActions
                code={code}
                validation={message.validation}
                artifactId={message.artifact_id}
                task={task}
              />
            ) : null
          }
        />
      )}
      {message.text.trim() && !streaming && !model && (
        <button
          type="button"
          onClick={() =>
            message.artifact_id
              ? void openArtifact(message.artifact_id).catch(() => undefined)
              : openDraft({ title: 'note', kind: 'markdown', content: message.text })
          }
          className="self-start font-mono text-[10px] uppercase tracking-[0.12em] text-ink-dim/70 transition-colors hover:text-accent"
        >
          {message.artifact_id ? 'Open in Canvas' : 'Send to Canvas'}
        </button>
      )}
      {streaming && (
        <span className="caret inline-block h-[1em] w-[2px] translate-y-[2px] bg-accent align-baseline" />
      )}
      {model && (
        <CodeBlock
          code={model}
          lang="sysml"
          actions={
            <ModelActions
              code={model}
              validation={message.validation}
              artifactId={message.artifact_id}
              task={task}
            />
          }
        />
      )}
      {message.validation && <ValidationReport logs={[message.validation]} />}
      {message.stopped && (
        <span className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-ink-dim/70">
          stopped
        </span>
      )}
    </article>
  );
}
