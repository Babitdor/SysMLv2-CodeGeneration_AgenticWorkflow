"""Modular SysML v2 validation service.

The default backend is this repository's ANTLR grammar. Set
``SYSML_VALIDATOR_URL`` to POST ``{"code": ...}`` to a hosted compiler instead.
If the grammar is unavailable, a structural checker keeps the
generate -> validate -> repair loop working out of the box.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_ISSUES = 200


class ValidationIssue(BaseModel):
    """One parser or structural finding."""

    severity: Literal["error", "warning", "info"] = "error"
    line: int = Field(default=0, ge=0)
    column: int = Field(default=0, ge=0)
    message: str
    rule: str = "syntax"


class ValidationReport(BaseModel):
    """Deterministic verdict consumed by both the UI and the repair loop."""

    ok: bool
    issues: List[ValidationIssue] = Field(default_factory=list)
    summary: str = ""
    validator: str = "antlr"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")


def _plural(count: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _summarize(ok: bool, issues: List[ValidationIssue]) -> str:
    errors = sum(1 for issue in issues if issue.severity == "error")
    warnings = sum(1 for issue in issues if issue.severity == "warning")
    if ok:
        return "Parsed successfully" + (
            f" · {_plural(warnings, 'warning')}" if warnings else ""
        )
    parts = [_plural(errors, "syntax error") if errors else "Validation failed"]
    if warnings:
        parts.append(_plural(warnings, "warning"))
    return " · ".join(parts)


def _dedupe(issues: List[ValidationIssue]) -> List[ValidationIssue]:
    seen = set()
    unique: List[ValidationIssue] = []
    for issue in issues:
        key = (issue.severity, issue.line, issue.column, issue.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
        if len(unique) >= MAX_ISSUES:
            break
    return unique


def _antlr_report(code: str) -> ValidationReport:
    """Parse with the vendored SysML v2 grammar."""
    from tools.sysml_parser.LoggingParsers.SysMLv2LoggingParser import (
        SysMLv2LoggingParser,
    )

    parser = SysMLv2LoggingParser(input_string=code)
    parser.parse()

    issues: List[ValidationIssue] = []
    for diagnostic in getattr(parser.error_logger, "diagnostics", []):
        severity = diagnostic.severity
        if severity not in ("error", "warning", "info"):
            severity = "error"
        issues.append(
            ValidationIssue(
                severity=severity,
                line=diagnostic.line,
                column=diagnostic.column,
                message=diagnostic.message,
                rule="antlr",
            )
        )

    issues = _dedupe(issues)
    ok = not any(issue.severity == "error" for issue in issues)
    if not ok and not issues:  # defensive: parser failed without a diagnostic
        issues = [
            ValidationIssue(
                message="The SysML grammar rejected the model without a diagnostic.",
                rule="antlr",
            )
        ]
    return ValidationReport(
        ok=ok, issues=issues, summary=_summarize(ok, issues), validator="antlr"
    )


def _strip_line_comment(line: str) -> str:
    """Best-effort comment strip for the keyword checks (strings stay intact)."""
    quote: Optional[str] = None
    i = 0
    while i < len(line):
        char = line[i]
        if quote:
            if char == "\\":
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "/" and line[i : i + 2] == "//":
            return line[:i]
        i += 1
    return line


_EMPTY_DEF = (
    "package",
    "part def",
    "attribute def",
    "item def",
    "action def",
    "state def",
    "requirement def",
    "constraint def",
    "port def",
    "connection def",
    "interface def",
    "allocation def",
    "view def",
    "enum def",
)

_DECLARATION = (
    "package",
    "part",
    "attribute",
    "item",
    "action",
    "state",
    "requirement",
    "constraint",
    "port",
    "connection",
    "interface",
    "allocation",
    "view",
    "analysis",
    "calculation",
    "concern",
    "import",
    "alias",
    "comment",
    "doc",
    "metadata",
    "enum",
    "variant",
    "individual",
    "use case",
)


def _structural_report(code: str) -> ValidationReport:
    """Brace/string/comment + obvious keyword checks, no compiler required."""
    issues: List[ValidationIssue] = []
    stack: List[Tuple[str, int, int]] = []
    pairs = {"}": "{", ")": "(", "]": "["}
    line, column = 1, 1
    index = 0
    in_block = False
    block_start = (1, 1)
    in_string: Optional[str] = None
    string_start = (1, 1)

    while index < len(code):
        char = code[index]
        nxt = code[index + 1] if index + 1 < len(code) else ""

        if in_block:
            if char == "*" and nxt == "/":
                in_block = False
                index += 2
                column += 2
                continue
            if char == "\n":
                line += 1
                column = 1
            else:
                column += 1
            index += 1
            continue

        if in_string:
            if char == "\\" and nxt:
                index += 2
                column += 2
                continue
            if char == in_string:
                in_string = None
            elif char == "\n":
                issues.append(
                    ValidationIssue(
                        message=(
                            "Unterminated string starting at line "
                            f"{string_start[0]}, column {string_start[1]}."
                        ),
                        line=string_start[0],
                        column=string_start[1],
                        rule="structure",
                    )
                )
                in_string = None
                line += 1
                column = 1
                index += 1
                continue
            else:
                column += 1
            index += 1
            continue

        if char == "/" and nxt == "/":
            newline = code.find("\n", index)
            if newline == -1:
                break
            index = newline
            line += 1
            column = 1
            continue
        if char == "/" and nxt == "*":
            in_block = True
            block_start = (line, column)
            index += 2
            column += 2
            continue
        if char in ("'", '"'):
            in_string = char
            string_start = (line, column)
            index += 1
            column += 1
            continue
        if char == "\n":
            line += 1
            column = 1
            index += 1
            continue
        if char in "{[(":
            stack.append((char, line, column))
        elif char in "}])":
            if not stack:
                issues.append(
                    ValidationIssue(
                        message=f"Unmatched closing '{char}'.",
                        line=line,
                        column=column,
                        rule="structure",
                    )
                )
            elif stack[-1][0] != pairs[char]:
                opener, open_line, open_column = stack.pop()
                issues.append(
                    ValidationIssue(
                        message=(
                            f"'{opener}' opened at line {open_line}, column "
                            f"{open_column} is closed by '{char}'."
                        ),
                        line=line,
                        column=column,
                        rule="structure",
                    )
                )
            else:
                stack.pop()
        column += 1
        index += 1

    if in_block:
        issues.append(
            ValidationIssue(
                message="Unterminated block comment.",
                line=block_start[0],
                column=block_start[1],
                rule="structure",
            )
        )
    if in_string:
        issues.append(
            ValidationIssue(
                message=(
                    "Unterminated string starting at line "
                    f"{string_start[0]}, column {string_start[1]}."
                ),
                line=string_start[0],
                column=string_start[1],
                rule="structure",
            )
        )
    for opener, open_line, open_column in stack:
        issues.append(
            ValidationIssue(
                message=f"Unclosed '{opener}'.",
                line=open_line,
                column=open_column,
                rule="structure",
            )
        )

    for number, raw_line in enumerate(code.splitlines(), start=1):
        content = _strip_line_comment(raw_line).strip()
        if not content:
            continue
        for keyword in _EMPTY_DEF:
            lowered = content.lower()
            if lowered == keyword + ";" or lowered == keyword + " {":
                issues.append(
                    ValidationIssue(
                        message=f"'{keyword}' is missing a name.",
                        line=number,
                        column=max(1, raw_line.lower().find(keyword) + 1),
                        rule="keyword",
                    )
                )
                break

    if not any(
        keyword in code.lower()
        for keyword in _DECLARATION
    ):
        issues.append(
            ValidationIssue(
                message="No recognizable SysML declaration was found.",
                line=1,
                column=1,
                rule="keyword",
            )
        )

    issues = _dedupe(issues)
    ok = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        ok=ok,
        issues=issues,
        summary=_summarize(ok, issues),
        validator="structural",
    )


def _http_report(url: str, code: str) -> ValidationReport:
    """Delegate to a hosted validator that speaks the same report shape."""
    payload = json.dumps({"code": code}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body: Dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        issue = ValidationIssue(
            message=f"Validation service unreachable: {exc}",
            rule="backend",
        )
        return ValidationReport(
            ok=False,
            issues=[issue],
            summary="Validation service unreachable",
            validator="http",
        )

    if "issues" in body or "ok" in body:
        issues: List[ValidationIssue] = []
        for raw in body.get("issues", []):
            try:
                issues.append(ValidationIssue(**raw))
            except Exception:
                issues.append(
                    ValidationIssue(message=str(raw), rule="backend")
                )
        issues = _dedupe(issues)
        ok = bool(body.get("ok", not any(i.severity == "error" for i in issues)))
        return ValidationReport(
            ok=ok,
            issues=issues,
            summary=str(body.get("summary") or _summarize(ok, issues)),
            validator=str(body.get("validator") or "http"),
        )

    detail = str(body.get("detail") or "")
    ok = detail.strip().startswith("✅")
    issues = (
        []
        if ok
        else [ValidationIssue(message=detail or "Hosted validation failed.", rule="http")]
    )
    return ValidationReport(
        ok=ok,
        issues=issues,
        summary="Parsed successfully" if ok else "Validation failed",
        validator="http",
    )


def validate_sysml(code: str, backend: str = "auto") -> ValidationReport:
    """Validate SysML v2 text with the configured backend.

    ``backend`` is ``auto`` (env URL, then ANTLR, then structural), ``antlr``,
    ``structural`` or ``http``. The structural path is intentionally small: it
    catches the brace/keyword mistakes that make the UI loop usable when no
    compiler is installed.
    """
    text = (code or "").strip()
    if not text:
        issue = ValidationIssue(
            message="The model is empty.", line=1, column=1, rule="structure"
        )
        return ValidationReport(
            ok=False,
            issues=[issue],
            summary="1 syntax error",
            validator="structural",
        )

    url = (os.environ.get("SYSML_VALIDATOR_URL") or "").strip()

    if backend == "http":
        if not url:
            issue = ValidationIssue(
                message="SYSML_VALIDATOR_URL is not configured.", rule="backend"
            )
            return ValidationReport(
                ok=False,
                issues=[issue],
                summary="Validation service not configured",
                validator="http",
            )
        return _http_report(url, text)

    if backend == "structural":
        return _structural_report(text)

    if backend not in ("auto", "antlr"):
        raise ValueError(f"unknown validation backend: {backend}")

    if backend == "auto" and url:
        return _http_report(url, text)

    try:
        return _antlr_report(text)
    except Exception as exc:  # grammar missing or crashed on hostile input
        logger.warning("ANTLR validation failed (%s); using structural checks", exc)
        report = _structural_report(text)
        report.issues.append(
            ValidationIssue(
                severity="info",
                message=f"ANTLR backend unavailable; structural checks used ({exc}).",
                rule="backend",
            )
        )
        report.summary = _summarize(report.ok, report.issues)
        return report


def report_detail(report: ValidationReport) -> str:
    """Human-readable verdict for legacy endpoints and tool output."""
    if report.ok:
        return "✅ " + (report.summary or "Code is valid!")
    lines = ["❌ Validation failed:"]
    for issue in report.issues:
        if issue.severity == "info":
            continue
        where = f"line {issue.line}, column {issue.column}: " if issue.line else ""
        lines.append(f"- {where}{issue.message}")
    if len(lines) == 1:
        lines.append("- " + (report.summary or "Unknown validation failure."))
    return "\n".join(lines)


def validate_sysml_code(code: str) -> Dict[str, Any]:
    """Dict-shaped verdict for the SSE pipeline and the JSON endpoints."""
    report = validate_sysml(code)
    return {
        "status": "valid" if report.ok else "invalid",
        "ok": report.ok,
        "validator": report.validator,
        "diagnostics": [issue.model_dump() for issue in report.issues],
        "summary": report.summary,
    }


def format_validation_result(result: Dict[str, Any]) -> str:
    """Render a ``validate_sysml_code`` verdict as plain text."""
    passed = result.get("status") == "valid"
    headline = result.get("summary") or ("Code is valid!" if passed else "Validation failed")
    lines = [("✅ " if passed else "❌ ") + headline]
    for issue in result.get("diagnostics") or []:
        severity = str(issue.get("severity", "error")).upper()
        where = (
            f"line {issue.get('line', 0)}, column {issue.get('column', 0)}: "
            if issue.get("line")
            else ""
        )
        lines.append(f"- {severity}: {where}{issue.get('message', '')}")
    return "\n".join(lines)
