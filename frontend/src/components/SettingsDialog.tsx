import { useCallback, useEffect, useState } from 'react';
import { Check, Loader2, LogOut, Trash2, X } from 'lucide-react';
import {
  applyProfile,
  clearProviderConfig,
  configure,
  credentialPath,
  deleteProfile,
  logout,
  me,
  readConfig,
  readProviderConfig,
  saveProviderConfig,
  updateCredential,
} from '../lib/api';
import type {
  ConfigPayload,
  ConfigState,
  CredentialService,
  ProviderConfigState,
  SessionUser,
} from '../lib/api';
import { useWorkspace } from '../state/WorkspaceContext';

const PROVIDERS = ['ollama', 'openai', 'openrouter', 'anthropic'];
const SERVICES: Record<CredentialService, string> = {
  ollama: 'Ollama',
  openai: 'OpenAI',
  openrouter: 'OpenRouter',
  anthropic: 'Anthropic',
  tavily: 'Tavily web search',
};

/** One input; the dialog has two forms, so the markup lives here once. */
function Field({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  hint,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  type?: string;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">{label}</span>
      <input
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
        className="border border-rule bg-bg px-2 py-1.5 font-mono text-[12px] text-ink outline-none placeholder:text-ink-dim/50 focus:border-accent/60"
      />
      {hint && (
        <span className="font-mono text-[10px] leading-snug text-ink-dim/70">{hint}</span>
      )}
    </label>
  );
}

/** The provider dropdown, shared by the account form and the deployment form. */
function ProviderSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="label">provider</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="border border-rule bg-bg px-2 py-1.5 font-mono text-[12px] text-ink outline-none focus:border-accent/60"
      >
        {PROVIDERS.map((provider) => (
          <option key={provider} value={provider}>
            {provider}
          </option>
        ))}
      </select>
    </label>
  );
}

/** This account's own backend as editable field values. */
interface OwnForm {
  provider: string;
  model: string;
  base_url: string;
  effort: string;
  temperature: string;
}

function ownFieldsFrom(state: ProviderConfigState): OwnForm {
  return {
    provider: state.provider,
    model: state.model,
    base_url: state.base_url,
    effort: state.effort,
    temperature: state.temperature === null ? '' : String(state.temperature),
  };
}

/** The deployment default as editable field values. */
function fieldsFrom(state: ConfigState): ConfigPayload {
  return {
    provider: state.provider,
    model: state.model,
    base_url: state.base_url,
    effort: state.effort,
    temperature: state.temperature,
  };
}

export function SettingsDialog({ onClose }: { onClose: () => void }) {
  const { dispatch } = useWorkspace();
  const [config, setConfig] = useState<ConfigState | null>(null);
  const [form, setForm] = useState<ConfigPayload>({});
  const [own, setOwn] = useState<ProviderConfigState | null>(null);
  const [ownForm, setOwnForm] = useState<OwnForm>({
    provider: 'ollama',
    model: '',
    base_url: '',
    effort: '',
    temperature: '',
  });
  const [service, setService] = useState<CredentialService>('openai');
  const [keyDraft, setKeyDraft] = useState('');
  const [keyResult, setKeyResult] = useState('');
  const [busy, setBusy] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [session, setSession] = useState<SessionUser | null>(null);
  const isAdmin = session?.role === 'admin';

  useEffect(() => {
    let cancelled = false;
    me()
      .then((payload) => {
        if (!cancelled) setSession(payload.user);
      })
      .catch(() => {
        /* 401 is handled by the gate above this dialog */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const note = (err: unknown) =>
    setResult(err instanceof Error ? err.message : String(err));

  // Each form opens on what the server has, not on a guess.
  const loadOwn = useCallback(async () => {
    try {
      const next = await readProviderConfig();
      setOwn(next);
      setOwnForm(ownFieldsFrom(next));
    } catch (err) {
      note(err);
    }
  }, []);

  const load = useCallback(async () => {
    try {
      const next = await readConfig();
      setConfig(next);
      setForm(fieldsFrom(next));
      setName(next.active ?? '');
    } catch (err) {
      note(err);
    }
  }, []);

  useEffect(() => {
    void loadOwn();
  }, [loadOwn]);

  useEffect(() => {
    // Reading the deployment configuration is an administrator route: asking as
    // anyone else would only earn a 403 to display.
    if (isAdmin) void load();
  }, [isAdmin, load]);

  const signOut = async () => {
    try {
      await logout();
    } finally {
      // A full reload drops every cached panel; the gate then shows the form.
      window.location.reload();
    }
  };

  const set = (key: keyof ConfigPayload, value: string) =>
    setForm((current) => ({ ...current, [key]: value }));

  const setOwnField = (key: keyof OwnForm, value: string) =>
    setOwnForm((current) => ({ ...current, [key]: value }));

  const numberOrNull = (text: string) => {
    const parsed = Number(text);
    return text.trim() === '' || !Number.isFinite(parsed) ? null : parsed;
  };

  const saveOwn = async () => {
    setBusy('own-save');
    setResult(null);
    try {
      const next = await saveProviderConfig({
        provider: ownForm.provider,
        model: ownForm.model.trim(),
        base_url: ownForm.base_url.trim(),
        effort: ownForm.effort.trim(),
        temperature: numberOrNull(ownForm.temperature),
      });
      setOwn(next);
      setOwnForm(ownFieldsFrom(next));
      setResult('your turns now use ' + next.provider + '/' + next.model);
      dispatch({ type: 'refreshHealth' });
    } catch (err) {
      note(err);
    } finally {
      setBusy(null);
    }
  };

  const clearOwn = async () => {
    setBusy('own-clear');
    setResult(null);
    try {
      const next = await clearProviderConfig();
      setOwn(next);
      setOwnForm(ownFieldsFrom(next));
      setResult('back to the deployment default: ' + next.provider + '/' + next.model);
      dispatch({ type: 'refreshHealth' });
    } catch (err) {
      note(err);
    } finally {
      setBusy(null);
    }
  };

  const changeKey = async (key: string | null) => {
    setBusy('credentials');
    setKeyResult('');
    try {
      const { credentials } = await updateCredential(service, key);
      setOwn((current) => current && { ...current, credentials });
      setKeyDraft('');
      setKeyResult(key === null ? 'API key removed.' : 'API key saved.');
      dispatch({ type: 'refreshHealth' });
    } catch (err) {
      setKeyResult(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const save = async (saveAs: string | null) => {
    setBusy(saveAs ? 'save' : 'apply');
    setResult(null);
    try {
      const payload: ConfigPayload = { ...form };
      if (payload.temperature !== undefined) {
        const parsed = Number(payload.temperature);
        payload.temperature = Number.isFinite(parsed) ? parsed : undefined;
      }
      const response = await configure(payload, saveAs ?? undefined);
      setResult(
        (saveAs ? 'saved and applied: ' : 'applied: ') +
          response.provider +
          '/' +
          response.model +
          ' · t' +
          response.temperature,
      );
      await load();
      // Telemetry in the top bar and any cached workflow state are now stale.
      dispatch({ type: 'refreshHealth' });
    } catch (err) {
      note(err);
    } finally {
      setBusy(null);
    }
  };

  const switchTo = async (target: string) => {
    setBusy('switch');
    setResult(null);
    try {
      await applyProfile(target);
      setResult('switched to ' + target);
      await load();
      dispatch({ type: 'refreshHealth' });
    } catch (err) {
      note(err);
    } finally {
      setBusy(null);
    }
  };

  const drop = async (target: string) => {
    setBusy('delete');
    setResult(null);
    try {
      await deleteProfile(target);
      await load();
    } catch (err) {
      note(err);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-bg/80 p-4 backdrop-blur-sm">
      <div className="rise max-h-full w-full max-w-[440px] overflow-y-auto border border-rule bg-panel">
        <header className="flex items-center justify-between border-b border-rule px-4 py-2.5">
          <span className="label">Settings</span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-sm p-1 text-ink-dim hover:text-ink"
            aria-label="Close"
          >
            <X className="size-3.5" />
          </button>
        </header>

        <div className="flex flex-col gap-3 px-4 py-3">
          <p className="font-mono text-[10.5px] leading-relaxed text-ink-dim/80">
            your own provider — the deployment default applies while this is empty
          </p>

          {!own && (
            <p className="font-mono text-[10.5px] text-ink-dim/70">
              reading your backend…
            </p>
          )}

          {own && (
            <>
              <ProviderSelect
                value={ownForm.provider}
                onChange={(value) => setOwnField('provider', value)}
              />
              <Field
                label="model"
                value={ownForm.model}
                onChange={(value) => setOwnField('model', value)}
                placeholder="glm-5.3-flash:cloud"
              />
              <Field
                label="base url"
                value={ownForm.base_url}
                onChange={(value) => setOwnField('base_url', value)}
                placeholder="http://127.0.0.1:11434"
                hint="openai-compatible endpoint; blank uses the provider default"
              />
              <Field
                label="temperature"
                value={ownForm.temperature}
                onChange={(value) => setOwnField('temperature', value)}
                placeholder="0.15"
                type="number"
              />
              <Field
                label="reasoning effort"
                value={ownForm.effort}
                onChange={(value) => setOwnField('effort', value)}
                placeholder="low | medium | high"
              />

              <p className="font-mono text-[10.5px] leading-relaxed text-ink-dim/80">
                {own.source === 'user'
                  ? 'in use for your turns: ' + own.provider + '/' + own.model
                  : 'inheriting the deployment default: ' + own.provider + '/' + own.model}
              </p>

              {result && <p className="font-mono text-[11px] text-ink">{result}</p>}

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => void saveOwn()}
                  disabled={busy !== null}
                  className="flex flex-1 items-center justify-center gap-2 border border-accent/60 py-2 font-mono text-[11.5px] uppercase tracking-[0.12em] text-accent transition-colors hover:bg-accent/10 disabled:opacity-50"
                >
                  {busy === 'own-save' && <Loader2 className="size-3.5 animate-spin" />}
                  save
                </button>
                <button
                  type="button"
                  onClick={() => void clearOwn()}
                  disabled={busy !== null || own.source === 'deployment'}
                  className="shrink-0 border border-rule px-2 py-2 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-bad/60 hover:text-bad disabled:opacity-40"
                  title="Use the deployment default again"
                >
                  {busy === 'own-clear' ? 'clearing…' : 'clear'}
                </button>
              </div>
              <form
                aria-labelledby="credential-heading"
                method="post"
                action={credentialPath(service)}
                onSubmit={(event) => {
                  event.preventDefault();
                  if (!busy) void changeKey(keyDraft);
                }}
                className="flex flex-col gap-3 border-t border-rule pt-3"
              >
                <h2 id="credential-heading" className="text-sm font-medium">Your API keys</h2>
                <p className="text-xs text-ink-dim">Save your model provider above to use its key. Web search uses your Tavily key.</p>
                <label className="flex flex-col gap-1">
                  <span className="label">Service</span>
                  <select
                    name="service"
                    value={service}
                    disabled={busy !== null}
                    onChange={(event) => {
                      setService(event.target.value as CredentialService);
                      setKeyDraft('');
                      setKeyResult('');
                    }}
                    className="min-h-12 border border-rule bg-bg px-2 focus-visible:outline-accent"
                  >
                    {(Object.keys(SERVICES) as CredentialService[]).map((item) => (
                      <option key={item} value={item}>
                        {SERVICES[item]}{own.credentials[item] ? ' · key saved' : ''}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex flex-col gap-1">
                  <span className="label">{SERVICES[service]} API key</span>
                  <input
                    type="password"
                    name="api_key"
                    autoComplete="off"
                    autoCorrect="off"
                    autoCapitalize="none"
                    spellCheck={false}
                    maxLength={4096}
                    required
                    disabled={busy !== null}
                    value={keyDraft}
                    onChange={(event) => setKeyDraft(event.target.value)}
                    placeholder={own.credentials[service] ? 'Enter a replacement key' : 'Paste your API key'}
                    className="min-h-12 border border-rule bg-bg px-2 focus-visible:outline-accent"
                  />
                </label>
                <div className="flex items-center gap-2">
                  <button type="submit" disabled={busy !== null} className="min-h-12 border border-accent/60 px-3 text-sm text-accent disabled:opacity-40">
                    Save key
                  </button>
                  <button type="button" onClick={() => void changeKey(null)}
                    disabled={busy !== null || !own.credentials[service]}
                    className="min-h-12 border border-rule px-3 text-sm text-ink-dim hover:text-bad disabled:opacity-40">
                    Remove key
                  </button>
                </div>
                <p role="status" className="text-xs text-ink-dim">{keyResult}</p>
              </form>
            </>
          )}

          {isAdmin && (
            <>
              <p className="label border-t border-rule pt-3">deployment default · admin</p>
              <div className="flex flex-col gap-1">
                <span className="label">saved configurations</span>
                {!config && (
                  <p className="font-mono text-[10.5px] text-ink-dim/70">
                    reading saved settings…
                  </p>
                )}
                {config?.profiles.length === 0 && (
                  <p className="font-mono text-[10.5px] text-ink-dim/70">
                    none yet — apply with a name below to keep one
                  </p>
                )}
                {(config?.profiles ?? []).map((profile) => {
                  const active = profile.name === config?.active;
                  return (
                    <div
                      key={profile.name}
                      className={
                        'flex items-center gap-2 border px-2 py-1.5 ' +
                        (active ? 'border-accent/50 bg-accent/5' : 'border-rule')
                      }
                    >
                      <button
                        type="button"
                        onClick={() => void switchTo(profile.name)}
                        disabled={busy !== null || active}
                        className="min-w-0 flex-1 text-left disabled:cursor-default"
                        title={active ? 'In use' : 'Switch to this configuration'}
                      >
                        <span className="block truncate font-mono text-[11.5px] text-ink">
                          {profile.name}
                        </span>
                        <span className="block truncate font-mono text-[10px] text-ink-dim">
                          {profile.provider} · {profile.model} · t{profile.temperature}
                          {profile.api_key_set ? ' · key' : ''}
                        </span>
                      </button>
                      {active && <Check className="size-3 shrink-0 text-accent" />}
                      <button
                        type="button"
                        onClick={() => void drop(profile.name)}
                        disabled={busy !== null}
                        className="rounded-sm p-1 text-ink-dim transition-colors hover:text-bad disabled:opacity-40"
                        aria-label={'Delete ' + profile.name}
                        title={'Delete ' + profile.name}
                      >
                        <Trash2 className="size-3" />
                      </button>
                    </div>
                  );
                })}
              </div>

              <ProviderSelect
                value={form.provider ?? ''}
                onChange={(value) => set('provider', value)}
              />
              <Field
                label="model"
                value={form.model === undefined ? '' : String(form.model)}
                onChange={(value) => set('model', value)}
                placeholder="glm-5.3-flash:cloud"
              />
              <Field
                label="temperature"
                value={form.temperature === undefined ? '' : String(form.temperature)}
                onChange={(value) => set('temperature', value)}
                placeholder="0.2"
                type="number"
              />
              <Field
                label="base url"
                value={form.base_url === undefined ? '' : String(form.base_url)}
                onChange={(value) => set('base_url', value)}
                placeholder="https://api.example.com/v1"
              />
              <Field
                label="reasoning effort"
                value={form.effort === undefined ? '' : String(form.effort)}
                onChange={(value) => set('effort', value)}
                placeholder="low | medium | high"
              />

              <p className="font-mono text-[10.5px] leading-relaxed text-ink-dim/80">
                saved server-side; every account without its own provider uses this.
              </p>

              <button
                type="button"
                onClick={() => void save(null)}
                disabled={busy !== null || !config}
                className="flex items-center justify-center gap-2 border border-accent/60 py-2 font-mono text-[11.5px] uppercase tracking-[0.12em] text-accent transition-colors hover:bg-accent/10 disabled:opacity-50"
              >
                {busy === 'apply' && <Loader2 className="size-3.5 animate-spin" />}
                apply to deployment
              </button>

              <div className="flex items-center gap-2">
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="name this configuration"
                  aria-label="Configuration name"
                  className="min-w-0 flex-1 border border-rule bg-bg px-2 py-1.5 font-mono text-[11.5px] text-ink outline-none placeholder:text-ink-dim/50 focus:border-accent/60"
                />
                <button
                  type="button"
                  onClick={() => void save(name.trim())}
                  disabled={busy !== null || !config || !name.trim()}
                  className="shrink-0 border border-rule px-2 py-1.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-accent/60 hover:text-accent disabled:opacity-40"
                >
                  {busy === 'save' ? 'saving…' : 'save'}
                </button>
              </div>
            </>
          )}

          <div className="flex items-center justify-between gap-2 border-t border-rule pt-2">
            <span className="min-w-0 truncate font-mono text-[10.5px] text-ink-dim">
              {session ? session.email : 'not signed in'}
              {session?.role === 'admin' ? ' · admin' : ''}
            </span>
            <button
              type="button"
              onClick={() => void signOut()}
              className="flex shrink-0 items-center gap-1.5 border border-rule px-2 py-1 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-dim transition-colors hover:border-bad/60 hover:text-bad"
            >
              <LogOut className="size-3" />
              sign out
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
