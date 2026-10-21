from __future__ import annotations

from uuid import UUID
from pathlib import Path

import pytest
from pydantic import ValidationError

from small_claims_agent.bug_reports import (
    BugReport,
    BugReportStoreError,
    _database_dsn_with_tls,
    submit_bug_report,
)


def make_report(**overrides) -> BugReport:
    values = {
        "category": "interview",
        "reproduction_steps": "Start a fictional interview and answer the first question.",
        "expected_behavior": "The answer should be accepted.",
        "actual_behavior": "The same question is repeated.",
        "app_version": "test-build",
    }
    values.update(overrides)
    return BugReport(**values)


def test_bug_report_stores_only_minimal_structured_fields():
    report = make_report()

    assert report.category == "interview"
    assert report.app_version == "test-build"
    assert report.contact_email is None
    assert isinstance(report.report_id, UUID)
    assert "messages" not in report.model_dump()
    assert "case" not in report.model_dump()


@pytest.mark.parametrize(
    "field,value",
    [
        ("reproduction_steps", "Use claimant ID 123456789 in the report"),
        ("expected_behavior", "Send the update to user@example.com"),
        ("actual_behavior", "Call me at 0501234567"),
    ],
)
def test_personal_identifiers_are_rejected_from_report_body(field, value):
    with pytest.raises(ValidationError):
        make_report(**{field: value})


def test_contact_email_requires_explicit_separate_consent():
    with pytest.raises(ValidationError):
        make_report(contact_email="Tester@Example.com")

    report = make_report(
        contact_email="Tester@Example.com",
        contact_email_consent=True,
    )
    assert report.contact_email == "tester@example.com"


def test_email_without_consent_is_not_saved():
    report = make_report(contact_email=None, contact_email_consent=True)
    assert report.contact_email is None


def test_report_and_contact_are_inserted_separately_and_parameterized():
    class Cursor:
        def __init__(self):
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement, parameters):
            self.calls.append((statement, parameters))

    class Connection:
        def __init__(self):
            self.test_cursor = Cursor()
            self.kwargs = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return self.test_cursor

    connection = Connection()

    def fake_connect(dsn, **kwargs):
        assert dsn == "postgresql://example.invalid/test?sslmode=require"
        connection.kwargs = kwargs
        return connection

    report = make_report(
        contact_email="tester@example.com",
        contact_email_consent=True,
    )
    report_id = submit_bug_report(
        report,
        "postgresql://example.invalid/test",
        connect=fake_connect,
    )

    assert report_id == report.report_id
    assert connection.kwargs == {"connect_timeout": 5}
    assert len(connection.test_cursor.calls) == 2
    report_sql, report_params = connection.test_cursor.calls[0]
    contact_sql, contact_params = connection.test_cursor.calls[1]
    assert "INSERT INTO bug_reports" in report_sql
    assert "INSERT INTO bug_report_contacts" in contact_sql
    assert report_params == (
        report.report_id,
        "interview",
        report.reproduction_steps,
        report.expected_behavior,
        report.actual_behavior,
        "test-build",
    )
    assert contact_params[:2] == (report.report_id, "tester@example.com")
    assert "tester@example.com" not in report_params


def test_database_dsn_requires_tls_unless_explicitly_overridden():
    assert _database_dsn_with_tls("postgresql://db.example.invalid/app") == (
        "postgresql://db.example.invalid/app?sslmode=require"
    )
    assert _database_dsn_with_tls(
        "postgresql://db.example.invalid/app?connect_timeout=4"
    ) == "postgresql://db.example.invalid/app?connect_timeout=4&sslmode=require"
    assert _database_dsn_with_tls(
        "postgresql://localhost/app?sslmode=disable"
    ) == "postgresql://localhost/app?sslmode=disable"


def test_missing_database_url_does_not_attempt_connection():
    def fail_if_called(*_args, **_kwargs):
        pytest.fail("database connection must not be attempted without configuration")

    with pytest.raises(BugReportStoreError, match="not configured"):
        submit_bug_report(make_report(), "", connect=fail_if_called)


def test_database_error_message_does_not_leak_dsn_or_driver_details():
    def fail_connection(*_args, **_kwargs):
        raise RuntimeError("postgres://user:super-secret-password@db.example.invalid")

    with pytest.raises(BugReportStoreError) as error:
        submit_bug_report(
            make_report(),
            "postgresql://user:secret@db.example.invalid/test",
            connect=fail_connection,
        )

    assert "secret" not in str(error.value)
    assert "db.example" not in str(error.value)


def test_streamlit_explains_when_bug_report_database_is_not_configured(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BUG_REPORTS_DATABASE_URL", "")
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "app.py",
        default_timeout=20,
    ).run()

    assert not app.exception
    assert any("שליחת דיווח עדיין לא הוגדרה" in item.value for item in app.info)
