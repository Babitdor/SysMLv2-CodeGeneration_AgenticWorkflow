create table user_service_credentials (
    user_id uuid not null references users(id) on delete cascade,
    service text not null check (service in ('ollama', 'openai', 'openrouter', 'anthropic', 'tavily')),
    key_cipher text not null,
    primary key (user_id, service)
);

insert into user_service_credentials (user_id, service, key_cipher)
select user_id, provider, api_key_cipher
from user_provider_config
where api_key_cipher <> '';

alter table user_provider_config drop column api_key_cipher;
