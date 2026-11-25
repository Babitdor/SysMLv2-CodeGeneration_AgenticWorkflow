import { useEffect, useRef, useState } from 'react';
import { Loader2, Upload, X } from 'lucide-react';
import { listDocuments, uploadDocuments } from '../lib/api';
import type { DocumentInfo } from '../lib/types';
import { useWorkspace } from '../state/WorkspaceContext';

export function Attachments({ onClose }: { onClose: () => void }) {
  const { state, dispatch } = useWorkspace();
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listDocuments()
      .then(setDocs)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const upload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      const added = await uploadDocuments(files);
      setDocs((current) => [...current, ...added]);
      dispatch({ type: 'attach', docIds: added.map((doc) => doc.id) });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
      if (input.current) input.current.value = '';
    }
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-bg/80 p-4 backdrop-blur-sm">
      <div className="rise w-full max-w-[520px] border border-rule bg-panel">
        <header className="flex items-center justify-between border-b border-rule px-4 py-2.5">
          <span className="label">source documents</span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-sm p-1 text-ink-dim hover:text-ink"
            aria-label="Close"
          >
            <X className="size-3.5" />
          </button>
        </header>

        <div className="px-4 py-3">
          <p className="mb-3 text-[13px] leading-relaxed text-ink-dim">
            txt · md · sysml · kerml · pdf · docx. Attached documents ground the answer;
            they never open the SysML workspace on their own.
          </p>

          <button
            type="button"
            onClick={() => input.current?.click()}
            disabled={busy}
            className="flex w-full items-center justify-center gap-2 border border-dashed border-rule px-3 py-4 font-mono text-[11.5px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-accent/60 hover:text-ink"
          >
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : <Upload className="size-3.5" />}
            {busy ? 'reading…' : 'upload files'}
          </button>
          <input
            ref={input}
            type="file"
            multiple
            accept=".txt,.md,.sysml,.kerml,.pdf,.docx"
            className="hidden"
            onChange={(event) => void upload(event.target.files)}
          />

          {error && <p className="mt-2 font-mono text-[11px] text-bad">{error}</p>}

          {docs.length > 0 && (
            <ul className="mt-3 flex flex-col divide-y divide-rule-soft border-t border-rule-soft">
              {docs.map((doc) => {
                const attached = state.docIds.includes(doc.id);
                return (
                  <li key={doc.id} className="flex items-center justify-between gap-3 py-2">
                    <span className="min-w-0">
                      <span className="block truncate text-[13.5px] text-ink/90">{doc.name}</span>
                      <span className="font-mono text-[10.5px] text-ink-dim">
                        {doc.chars.toLocaleString()} chars · {doc.chunks} chunks
                      </span>
                    </span>
                    <button
                      type="button"
                      onClick={() =>
                        dispatch(
                          attached
                            ? { type: 'detach', docId: doc.id }
                            : { type: 'attach', docIds: [doc.id] },
                        )
                      }
                      className={
                        'shrink-0 border px-2 py-1 font-mono text-[10.5px] uppercase tracking-[0.1em] transition-colors ' +
                        (attached
                          ? 'border-accent/60 text-accent'
                          : 'border-rule text-ink-dim hover:text-ink')
                      }
                    >
                      {attached ? 'attached' : 'attach'}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
