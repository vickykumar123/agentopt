"""Redaction of secrets/PII before text is written into a committed artifact."""

from __future__ import annotations

from agentopt.redaction import redact_text


def test_redacts_common_pii_and_secrets() -> None:
    text = "email a@b.com, call 555-123-4567, ssn 123-45-6789, key sk-ABCDEFGHIJKLMNOP1234"
    redacted, fields = redact_text(text)
    assert set(fields) == {"email", "phone", "ssn", "api_key"}
    for leak in ("a@b.com", "555-123-4567", "123-45-6789", "sk-ABCDEFGHIJKLMNOP1234"):
        assert leak not in redacted
    assert "[REDACTED:email]" in redacted


def test_clean_text_is_unchanged() -> None:
    redacted, fields = redact_text("just a normal question about pricing")
    assert fields == []
    assert redacted == "just a normal question about pricing"


def test_custom_patterns_from_config_are_applied() -> None:
    redacted, fields = redact_text("order ACME-9931 shipped", [r"ACME-\d+"])
    assert "custom" in fields
    assert "ACME-9931" not in redacted


def test_invalid_custom_pattern_is_matched_literally_not_raised() -> None:
    # An unbalanced-paren regex must not crash review — it falls back to a literal.
    redacted, fields = redact_text("weird ( paren here", ["("])
    assert "custom" in fields
    assert "(" not in redacted
