"""SysML validation with explicit compiler/fallback evidence."""

import asyncio
import hashlib
import json
import os
import re
from pathlib import PurePosixPath
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from .sysml_parser.LoggingParsers.SysMLv2LoggingParser import SysMLv2LoggingParser
from .sysml_parser.SemanticError import SemanticError, ValidationResult
from .sysml_parser.SemanticValidator import validate_semantics


def _current_work():
    try:
        from mbse.services.resource_limits import current_work
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("mbse"):
            raise
        return None
    return current_work()


def validate_via_sidecar(code: str) -> ValidationResult:
    service_url = os.getenv("SYSML_KERNEL_SERVICE_URL", "http://sysml-kernel:8888")

    def incomplete(reason: str) -> ValidationResult:
        return ValidationResult(backend="sidecar", completed=False, fallback_reason=reason)

    if not service_url:
        return incomplete("SYSML_KERNEL_SERVICE_URL is not configured")

    url = f"{service_url.rstrip('/')}/validate"
    payload = json.dumps({"sysml_text": code}).encode("utf-8")
    try:
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=310) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, ValueError) as exc:
        return incomplete(f"Sidecar request failed ({type(exc).__name__})")

    if not isinstance(data, dict) or type(data.get("success")) is not bool:
        return incomplete("Malformed sidecar response: expected a boolean success field")
    errors = data.get("errors")
    if not isinstance(errors, list) or any(not isinstance(e, str) for e in errors):
        return incomplete("Malformed sidecar response: expected a list of diagnostic strings")
    if type(data.get("completed")) is not bool:
        return incomplete("Malformed sidecar response: expected a boolean completed field")

    result = ValidationResult(backend="sidecar", completed=data["completed"])
    err_re = re.compile(
        r"^(ERROR|WARNING):\s*(.*?)(?:\s*\((?:.*?)line\s*:\s*(\d+)\s*column\s*:\s*(\d+)\))?$",
        re.IGNORECASE,
    )
    for diagnostic in errors:
        diagnostic = diagnostic.strip()
        if not diagnostic:
            continue
        match = err_re.match(diagnostic)
        if match:
            severity, message, line, column = match.groups()
            severity = severity.lower()
            if severity == "error" and any(term in message.lower() for term in (
                "mismatched input", "extraneous input", "expecting", "syntax error"
            )):
                result.syntax_errors.append(diagnostic)
                continue
            result.semantic_errors.append(SemanticError(
                message=message, line=int(line or 0), column=int(column or 0), severity=severity
            ))
        else:
            result.semantic_errors.append(SemanticError(message=diagnostic))

    # Failed responses, including warning-only diagnostics, cannot become passes.
    if not data["success"] and not result.has_errors:
        result.completed = False
        result.semantic_errors.append(SemanticError(
            message="Compiler reported failure without error diagnostics."
        ))
    return result


class SysMLValidationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_path: str | None = Field(default=None, description="Preferred: saved .sysml path under /workspace/current/; do not resend model text")
    code: str | None = Field(default=None, description="Inline model text, only when no saved file is available; use exactly one input")


class SysMLValidatorTool(BaseTool):
    """Validate model text without overstating fallback checks."""

    name: str = "SysML-validator-tool"
    description: str = (
        "Validates saved SysML v2 syntax and semantics. Prefer file_path for the actual saved file; "
        "inline code is also supported. File mode returns a SHA256 receipt and detects concurrent edits. "
        "Reports the validation backend, diagnostics and incomplete checks. "
        "Local fallback checks do not establish compiler acceptance."
    )
    args_schema: type[BaseModel] = SysMLValidationInput
    sandbox: Any = Field(default=None, exclude=True, repr=False)
    model_directory: str = Field(default="/workspace/current/", exclude=True)

    def _run(self, code: str | None = None, file_path: str | None = None) -> str:
        if (code is None) == (file_path is None):
            return "Validation incomplete: supply exactly one of file_path or code."
        if file_path is None:
            return self._checked_code(code, "inline model")
        path = PurePosixPath(file_path)
        if (not file_path.startswith(self.model_directory) or path.suffix != ".sysml"
                or ".." in path.parts or "\\" in file_path or "\x00" in file_path):
            return f"Validation incomplete: file_path must be a .sysml file under {self.model_directory}."
        if self.sandbox is None:
            return "Validation incomplete: no sandbox is attached; supply inline code."
        try:
            response = self.sandbox.download_files([file_path])[0]
            raw = response.content
            if response.error or raw is None:
                return "Validation incomplete: saved model could not be read."
            if len(raw) > 1024 * 1024:
                return "Validation incomplete: saved model exceeds the 1 MiB validation limit."
            text = raw.decode("utf-8")
            report = self._checked_code(text, file_path)
            after = self.sandbox.download_files([file_path])[0]
            if after.error or after.content != raw:
                work = _current_work()
                if work:
                    work.stop_reason = "Repair stopped: the saved model changed during validation; inspect the writer/input conflict."
                return "Validation incomplete: saved model changed during validation; validate its current version."
            return f"Validated file: {file_path}\nSHA256: {hashlib.sha256(raw).hexdigest()}\n{report}"
        except (OSError, ValueError, IndexError):
            return "Validation incomplete: saved model could not be read as stable UTF-8 text."

    def _validate_code(self, code: str) -> ValidationResult:
        try:
            parser = SysMLv2LoggingParser(input_string=code)
            parser.parse()
            if parser.has_errors():
                return ValidationResult(
                    syntax_errors=parser.get_errors_list(), backend="local-syntax"
                )

            result = validate_via_sidecar(code)
            # Only transport/configuration/protocol failures invoke fallback.
            # Preserve compiler-declared failure or incomplete execution.
            if not result.fallback_reason:
                return result
            return ValidationResult(
                semantic_errors=validate_semantics(code, syntax_errors=None),
                backend="local-semantic",
                fallback_reason=result.fallback_reason,
            )
        except Exception as exc:
            return ValidationResult(completed=False, backend="validator", semantic_errors=[
                SemanticError(message=f"Error during validation: {exc}")])

    def _checked_code(self, code: str, path: str) -> str:
        work = _current_work()
        if work:
            blocked = work.begin_validation(path, code)
            if blocked:
                return blocked
        result = self._validate_code(code)
        if work:
            work.finish_validation(path, result)
        return result.format_report()

    async def _arun(self, code: str | None = None, file_path: str | None = None) -> str:
        args = (code,) if file_path is None else (code, file_path)
        try:
            from mbse.services.resource_limits import kernel_call
        except ModuleNotFoundError as exc:
            if not exc.name or not exc.name.startswith("mbse"):
                raise
            return await asyncio.to_thread(self._run, *args)
        # The runtime's bounded executor owns blocking parser/HTTP work when present.
        return await kernel_call(self._run, *args)


sysml_validator_tool = SysMLValidatorTool()
