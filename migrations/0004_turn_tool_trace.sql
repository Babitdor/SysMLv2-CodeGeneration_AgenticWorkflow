ALTER TABLE thread_messages ADD COLUMN tool_trace jsonb NOT NULL DEFAULT '[]';
