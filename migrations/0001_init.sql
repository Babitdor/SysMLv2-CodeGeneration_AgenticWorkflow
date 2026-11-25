-- 0001_init: accounts, sessions and every tenant-owned row.
--
-- Everything below is keyed by user_id, and every query in the app filters on
-- it, so a foreign id answers 404 instead of leaking another tenant's data.
-- gen_random_uuid() is core since Postgres 13 (no pgcrypto extension needed).

create table if not exists users (
    id            uuid primary key default gen_random_uuid(),
    email         text not null,
    password_hash text not null,
    role          text not null default 'user',
    created_at    timestamptz not null default now()
);

-- Case-insensitive uniqueness without the citext extension.
create unique index if not exists users_email_key on users (lower(email));

-- The raw session token only ever lives in the browser; this table holds its
-- sha256, so a database dump does not hand out working sessions.
create table if not exists sessions (
    token_hash text primary key,
    user_id    uuid not null references users(id) on delete cascade,
    created_at timestamptz not null default now(),
    expires_at timestamptz not null,
    revoked_at timestamptz
);

create index if not exists sessions_user_id_idx on sessions (user_id);

create table if not exists threads (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references users(id) on delete cascade,
    title      text not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists threads_owner_updated_idx
    on threads (user_id, updated_at desc);

-- One row per turn. The generated model and the validation verdict ride along
-- as columns so a code-only turn (no prose) still survives a reload.
create table if not exists thread_messages (
    id         bigserial primary key,
    thread_id  uuid not null references threads(id) on delete cascade,
    role       text not null,
    content    text not null default '',
    model      text not null default '',
    validation jsonb,
    created_at timestamptz not null default now()
);

create index if not exists thread_messages_thread_idx
    on thread_messages (thread_id, id);

-- Uploaded engineering documents. The extracted text and its chunks live here
-- instead of an unbounded in-process dict, so a restart keeps them and N
-- workers agree on what is attached.
create table if not exists documents (
    id         uuid primary key default gen_random_uuid(),
    user_id    uuid not null references users(id) on delete cascade,
    name       text not null,
    chars      integer not null,
    text       text not null,
    chunks     jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists documents_owner_created_idx
    on documents (user_id, created_at desc);

-- Deployment-wide backend configuration written by /api/config (admin only):
-- `overrides` (the live SYSML_* values) and `profiles` (named saves).
create table if not exists app_settings (
    key        text primary key,
    value      jsonb not null,
    updated_at timestamptz not null default now()
);
