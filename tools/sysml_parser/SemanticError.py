"""Semantic error dataclass for SysML v2 validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SemanticError:
    """A single semantic error found during SysML v2 validation."""

    message: str
    line: int = 0
    column: int = 0
    severity: str = "error"  # "error" | "warning"
    error_type: str = "semantic"  # "semantic" | "reference" | "type" | "structural"

    def __str__(self) -> str:
        prefix = "WARNING" if self.severity == "warning" else "ERROR"
        loc = f" ({self.line}:{self.column})" if self.line else ""
        return f"{prefix}: {self.message}{loc}"


@dataclass
class ValidationResult:
    """Combined result of syntax + semantic validation."""

    syntax_errors: list[str] = field(default_factory=list)
    semantic_errors: list[SemanticError] = field(default_factory=list)
    backend: str = ""
    completed: bool = True
    fallback_reason: str = ""

    @property
    def has_errors(self) -> bool:
        return not self.completed or bool(self.syntax_errors) or any(
            e.severity == "error" for e in self.semantic_errors
        )

    @property
    def has_warnings(self) -> bool:
        return any(e.severity == "warning" for e in self.semantic_errors)

    def format_report(self) -> str:
        """Format a human-readable validation report."""
        lines: list[str] = []
        if self.backend:
            lines.append(f"Validation backend: {self.backend}")
        if self.fallback_reason:
            lines.append(f"Compiler validation unavailable: {self.fallback_reason}")
        if not self.completed:
            lines.append("Validation incomplete; compiler acceptance has not been established.")

        if self.syntax_errors:
            lines.append(f"Found {len(self.syntax_errors)} syntax error(s):")
            lines.extend(self.syntax_errors)

        if self.semantic_errors:
            errors = [e for e in self.semantic_errors if e.severity == "error"]
            warnings = [e for e in self.semantic_errors if e.severity == "warning"]
            if errors:
                lines.append(f"\nFound {len(errors)} semantic error(s):")
                lines.extend(str(e) for e in errors)
            if warnings:
                lines.append(f"\nFound {len(warnings)} semantic warning(s):")
                lines.extend(str(e) for e in warnings)

        if not self.has_errors and not self.has_warnings:
            if self.backend == "local-semantic":
                lines.append("Local checks passed. Compiler acceptance has not been established.")
            else:
                lines.append("✅ Code is valid! No syntax or semantic errors found.")

        return "\n".join(lines)
