-- 0002_artifacts: the canvas. An artifact is one named document owned by a
-- thread; its history is a table of immutable versions, so "go back to v1"
-- never destroys a version.
--
-- The artifact row carries a pointer to its current version rather than the
-- content: the left-hand pane edits code or markdown, and the same artifact may
-- have been either at different points, which is why `kind` lives on the version
-- and is only mirrored onto the artifact for the tab strip.

create table if not exists artifacts (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references users(id) on delete cascade,
    thread_id       uuid not null references threads(id) on delete cascade,
    title           text not null default '',
    kind            text not null default 'code',
    current_version integer not null default 1,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists artifacts_owner_thread_idx
    on artifacts (user_id, thread_id, updated_at desc);

create table if not exists artifact_versions (
    id          bigserial primary key,
    artifact_id uuid not null references artifacts(id) on delete cascade,
    version     integer not null,
    kind        text not null default 'code',
    content     text not null default '',
    origin      text not null default 'edit',
    created_at  timestamptz not null default now(),
    -- Two workers computing the same next version would silently rewrite
    -- history: the loser of this constraint errors instead.
    unique (artifact_id, version)
);

create index if not exists artifact_versions_artifact_idx
    on artifact_versions (artifact_id, version desc);

-- The transcript needs to reopen the artifact a turn produced. set null, not
-- cascade: deleting an artifact must not delete the conversation.
alter table thread_messages
    add column if not exists artifact_id uuid references artifacts(id) on delete set null;
