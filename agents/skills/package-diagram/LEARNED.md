# Learned Fixes — SysML v2 Package Diagram

This file accumulates fix patterns discovered by the SysML agent during
generation and validation.  Each entry records a validator error and the
corresponding fix, so future sessions can learn from past mistakes.

Format:
```
## Session YYYY-MM-DD
- Error: `<verbatim validator error message>`
  Fix: `<what was changed and why>`
- Error: `<verbatim validator error>`
  STATUS: UNRESOLVED — `<why it couldn't be fixed>`
```

The agent reads this file in Phase A (alongside SKILL.md) and appends
new entries in Phase C after completing all files.

## Session 2026-11-06
- Error: `Syntax error at line 6, column 6: Unexpected 'depends', expecting one of: default, :=, ;...` — the `sysml_syntax_reference_tool` documentation for "dependency relationships" advertises `<element> depends on <target>;`, but the parser rejects that form.
  Fix: Declare dependencies at package level as `dependency <Name> from <SourcePkg> to <TargetPkg>;` (both ends may be qualified, e.g. `to PhysicalPhase::BlockDefinitionDiagram`). Never emit `<element> depends on <target>;` and never subtype a dependency (`dependency B :> A;` is rejected too).
- Error: `Unexpected '{', expecting one of: about, locale, REGULAR_COMMENT` at `comment <name> { doc /* ... */ }`
  Fix: A named comment takes its body directly: `comment <name> /* text */` (or `comment <name> about <element> /* text */`). The brace form only exists for anonymous `comment { ... }`; combine `doc /* ... */` inside each `package` instead.
- Error: `Couldn't resolve reference to Namespace '<OtherPackage>'.` when a package diagram declares `private import <OtherPackage>::*;` for a namespace that lives in another file
  Fix: Do not add cross-file imports (or empty stand-in packages) to a package view — each accepted view file is a self-contained validation unit with inline declarations. Model cross-file relations with `dependency ... from ... to ...`, and document in the header why no imports are present.

## Session 2026-10-07
- No new fixes.

## Session 2026-10-04
- No new fixes.

## Session 2026-09-22
- No new fixes.

## Session 2025-06-16
- No new fixes.

## Session 2026-11-07
- No new fixes. Validated with `dependency <Name> from <SourcePkg> to <TargetPkg>;` for cross-view relations in a self-contained package view (no cross-file imports). Sidecar backend PASS on first attempt.
