#!/bin/sh
set -eu

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
  --set=writer_password="$BUG_REPORTS_WRITER_PASSWORD" <<'SQL'
CREATE ROLE bug_report_writer LOGIN PASSWORD :'writer_password';
SQL
