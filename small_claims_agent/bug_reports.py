from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .privacy import free_text_privacy_findings


BugCategory = Literal[
    "ui",
    "interview",
    "draft_pdf",
    "legal_sources",
    "llm_provider",
    "other",
]

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_INSERT_REPORT = """
INSERT INTO bug_reports (
    report_id, category, reproduction_steps, expected_behavior,
    actual_behavior, app_version
) VALUES (%s, %s, %s, %s, %s, %s)
"""
_INSERT_CONTACT = """
INSERT INTO bug_report_contacts (report_id, email, consented_at)
VALUES (%s, %s, %s)
"""


def _database_dsn_with_tls(database_url: str) -> str:
    """Require encrypted transport by default; allow explicit local-dev override."""
    parts = urlsplit(database_url)
    if parts.scheme not in {"postgres", "postgresql"}:
        raise ValueError("database URL must use the PostgreSQL URL scheme")
    query = parse_qsl(parts.query, keep_blank_values=True)
    if not any(key.lower() == "sslmode" for key, _ in query):
        query.append(("sslmode", "require"))
    return urlunsplit(parts._replace(query=urlencode(query)))


class BugReport(BaseModel):
    """Minimal, validated bug report. Never includes case state or chat history."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    report_id: UUID = Field(default_factory=uuid4)
    category: BugCategory
    reproduction_steps: str = Field(min_length=3, max_length=2000)
    expected_behavior: str = Field(min_length=2, max_length=1200)
    actual_behavior: str = Field(min_length=2, max_length=1200)
    app_version: str = Field(min_length=1, max_length=80)
    contact_email: str | None = Field(default=None, max_length=254)
    contact_email_consent: bool = False

    @model_validator(mode="after")
    def validate_privacy_and_contact(self) -> "BugReport":
        free_text = "\n".join(
            (self.reproduction_steps, self.expected_behavior, self.actual_behavior)
        )
        if free_text_privacy_findings(free_text):
            raise ValueError("free-text bug report contains possible personal information")
        if self.contact_email:
            if not _EMAIL_RE.fullmatch(self.contact_email):
                raise ValueError("contact email is invalid")
            if not self.contact_email_consent:
                raise ValueError("contact email requires explicit consent")
            self.contact_email = self.contact_email.lower()
        return self


class BugReportStoreError(RuntimeError):
    """A database failure with no driver message or connection details attached."""


def submit_bug_report(
    report: BugReport,
    database_url: str,
    *,
    connect: Callable | None = None,
) -> UUID:
    """Write only the report fields and separately consented contact email.

    The schema is provisioned out-of-band with ``db/bug_reports_schema.sql``.
    A separate, insert-only application DB role is recommended.
    """
    if not database_url or not database_url.strip():
        raise BugReportStoreError("Bug report storage is not configured")

    try:
        secure_dsn = _database_dsn_with_tls(database_url.strip())
        if connect is None:
            import psycopg

            connect = psycopg.connect
        with connect(secure_dsn, connect_timeout=5) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    _INSERT_REPORT,
                    (
                        report.report_id,
                        report.category,
                        report.reproduction_steps,
                        report.expected_behavior,
                        report.actual_behavior,
                        report.app_version,
                    ),
                )
                if report.contact_email:
                    cursor.execute(
                        _INSERT_CONTACT,
                        (
                            report.report_id,
                            report.contact_email,
                            datetime.now(timezone.utc),
                        ),
                    )
    except Exception:
        # Driver exceptions can include DSNs, hostnames, or server details.
        raise BugReportStoreError("Bug report could not be stored") from None
    return report.report_id
