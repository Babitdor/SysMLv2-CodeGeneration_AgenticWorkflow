# Learned Fixes — SysML v2 Analysis Case Diagram

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

## Session 2026-08-03
- No new fixes.

## Session 2026-08-02
- Error: `Extra token '{' found, expecting one of: behavior, crosses, feature...` at `doc {` — the multi-line `doc { ... }` block form is rejected by the validator.
  Fix: Use the single-line `doc /* ... */` form for all doc comments in analysis def bodies. The `doc { ... }` block syntax is not supported by this validator.

## Session 2026-08-01
- Error: `Invalid syntax near 'case' — no valid parsing rule matches` for `analysis case def StaticLoadAnalysis` and `analysis case staticLoad : StaticLoadAnalysis`.
  Fix: The validator's grammar uses the single keyword `analysis def <Name>` and `analysis <name> : <Def>` — NOT the two-word `analysis case`. Replace `analysis case def` with `analysis def` and `analysis case <name>` with `analysis <name>`. The `subject`/`objective`/`return`/`constraint` body shape is unchanged. (Note: the SKILL.md documents `analysis case def`, but the validator rejects it; use `analysis def` for zero-error output.)
