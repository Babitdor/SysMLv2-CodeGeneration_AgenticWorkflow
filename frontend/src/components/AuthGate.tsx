import { useEffect, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { Loader2 } from 'lucide-react';
import { login, me, register, setUnauthorizedHandler } from '../lib/api';
import type { SessionUser } from '../lib/api';
import { Account } from '../lib/account';

type Mode = 'login' | 'register';

const MIN_PASSWORD = 10;

/**
 * Sign-in wall for the whole app.
 *
 * Every data route is account-scoped now, so an unauthenticated tab has nothing
 * to show: this renders the form, and any later 401 (expired or revoked session)
 * drops the user back here instead of leaving a broken workspace on screen.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<SessionUser | null>(null);
  const [checking, setChecking] = useState(true);
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [signupOpen, setSignupOpen] = useState(true);

  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
    return () => setUnauthorizedHandler(null);
  }, []);

  useEffect(() => {
    let cancelled = false;
    me()
      .then((session) => {
        if (cancelled) return;
        setUser(session.user);
        setSignupOpen(session.signup_open);
      })
      .catch(() => {
        /* 401 is the normal logged-out answer: show the form */
      })
      .finally(() => {
        if (!cancelled) setChecking(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session =
        mode === 'login' ? await login(email, password) : await register(email, password);
      setUser(session.user);
      setPassword('');
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (user) return <Account.Provider value={user}>{children}</Account.Provider>;

  return (
    <div className="relative flex h-dvh items-center justify-center overflow-hidden p-4">
      <div aria-hidden className="blueprint pointer-events-none fixed inset-0 z-0 opacity-90" />
      <div className="rise relative z-10 w-full max-w-[360px] border border-rule bg-panel">
        <header className="flex items-center justify-between border-b border-rule px-4 py-2.5">
          <span className="label">sysml v2 agent</span>
          <span className="font-mono text-[10px] text-ink-dim">sign in</span>
        </header>

        {checking ? (
          <div className="flex items-center gap-2 px-4 py-6 font-mono text-[11px] text-ink-dim">
            <Loader2 className="size-3.5 animate-spin" />
            checking session…
          </div>
        ) : (
          <form onSubmit={submit} className="flex flex-col gap-3 px-4 py-4">
            <div className="flex gap-1 border border-rule p-0.5">
              {(['login', 'register'] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => {
                    setMode(option);
                    setError(null);
                  }}
                  className={
                    'flex-1 px-2 py-1 font-mono text-[10.5px] uppercase tracking-wide ' +
                    (mode === option
                      ? 'bg-accent/15 text-ink'
                      : 'text-ink-dim hover:text-ink')
                  }
                >
                  {option === 'login' ? 'sign in' : 'create account'}
                </button>
              ))}
            </div>

            <label className="flex flex-col gap-1">
              <span className="label">email</span>
              <input
                type="email"
                autoComplete="username"
                required
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="border border-rule bg-bg px-2 py-1.5 font-mono text-[12px] text-ink outline-none placeholder:text-ink-dim/50 focus:border-accent/60"
                placeholder="you@example.com"
              />
            </label>

            <label className="flex flex-col gap-1">
              <span className="label">password</span>
              <input
                type="password"
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                required
                minLength={mode === 'register' ? MIN_PASSWORD : undefined}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="border border-rule bg-bg px-2 py-1.5 font-mono text-[12px] text-ink outline-none placeholder:text-ink-dim/50 focus:border-accent/60"
                placeholder={mode === 'register' ? 'at least 10 characters' : '••••••••'}
              />
            </label>

            {error && (
              <p className="border border-bad/40 bg-bad/10 px-2 py-1.5 font-mono text-[10.5px] text-bad">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="flex items-center justify-center gap-2 border border-accent/50 bg-accent/10 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wide text-ink hover:bg-accent/20 disabled:opacity-50"
            >
              {busy && <Loader2 className="size-3.5 animate-spin" />}
              {mode === 'login' ? 'sign in' : 'create account'}
            </button>

            {mode === 'register' && !signupOpen && (
              <p className="font-mono text-[10.5px] text-ink-dim/80">
                registration is closed on this deployment — ask the administrator
                for an account.
              </p>
            )}
            {mode === 'register' && (
              <p className="font-mono text-[10.5px] text-ink-dim/70">
                the first account created becomes the administrator.
              </p>
            )}
          </form>
        )}
      </div>
    </div>
  );
}
