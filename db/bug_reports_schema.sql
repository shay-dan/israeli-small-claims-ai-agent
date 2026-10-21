-- Dedicated schema for closed-beta bug reports only.
-- Run this once with a database-owner account. The app must use a separate
-- least-privilege role that can INSERT into these tables, not own the schema.
-- Create bug_report_writer before applying this schema. The runtime role can
-- insert only; it cannot read, update, delete, or alter the tables.

CREATE TABLE IF NOT EXISTS bug_reports (
    report_id UUID PRIMARY KEY,
    category TEXT NOT NULL CHECK (
        category IN ('ui', 'interview', 'draft_pdf', 'legal_sources', 'llm_provider', 'other')
    ),
    reproduction_steps TEXT NOT NULL CHECK (char_length(btrim(reproduction_steps)) BETWEEN 3 AND 2000),
    expected_behavior TEXT NOT NULL CHECK (char_length(btrim(expected_behavior)) BETWEEN 2 AND 1200),
    actual_behavior TEXT NOT NULL CHECK (char_length(btrim(actual_behavior)) BETWEEN 2 AND 1200),
    app_version TEXT NOT NULL CHECK (char_length(btrim(app_version)) BETWEEN 1 AND 80),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Contact details are isolated from the report body and only inserted after
-- explicit opt-in. They can be deleted independently by report_id.
CREATE TABLE IF NOT EXISTS bug_report_contacts (
    report_id UUID PRIMARY KEY REFERENCES bug_reports(report_id) ON DELETE CASCADE,
    email TEXT NOT NULL CHECK (char_length(email) BETWEEN 3 AND 254),
    consented_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS bug_reports_created_at_idx ON bug_reports (created_at DESC);

COMMENT ON TABLE bug_reports IS 'Closed-beta bug reports; do not store case facts, transcripts, screenshots, or identifiers.';
COMMENT ON TABLE bug_report_contacts IS 'Optional follow-up contacts, stored separately after explicit consent.';

GRANT USAGE ON SCHEMA public TO bug_report_writer;
GRANT INSERT ON TABLE bug_reports, bug_report_contacts TO bug_report_writer;
