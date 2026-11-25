-- 0003_user_provider_config: one backend per account, in front of the deployment
-- default.
--
-- The deployment-wide configuration stays in app_settings and stays the
-- administrator's switch; this row lets any signed-in account point its own
-- turns at its own endpoint and key without reconfiguring everyone else. No row
-- means "use the deployment default", so clearing is a delete.
--
-- api_key_cipher is a Fernet token (see webapp/providers.py), not a bytea, so
-- the column reads like every other text column here; the token itself is
-- opaque and unreadable without SYSML_SECRET_KEY.

create table if not exists user_provider_config (
    user_id        uuid primary key references users(id) on delete cascade,
    provider       text not null,
    model          text not null default '',
    base_url       text not null default '',
    api_key_cipher text not null default '',
    temperature    real,
    effort         text not null default '',
    updated_at     timestamptz not null default now()
);
