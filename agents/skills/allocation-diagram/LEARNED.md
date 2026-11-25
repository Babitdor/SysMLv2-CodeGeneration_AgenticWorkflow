# Learned Fixes — SysML v2 Allocation Diagram

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

## Session 2026-08-01
- No new fixes.

## Session 2026-07-10
- Error: `WARNING: Couldn't resolve reference to Feature 'ActionUsage'` / `'PartUsage'` — the `end function : ActionUsage; end component : PartUsage;` typed ends produce semantic warnings when the validator does not bundle the SysML v2 standard library types.
  Fix: No code change needed — these are warnings, not errors. The file still validates with zero syntax and zero semantic errors. Keep the typed ends as specified by the skill; the warnings are benign standalone-validation artifacts.

## Session 2025-06-16
- Error: `Couldn't resolve reference to Feature '<name>'` — when validating an allocation file standalone, `public import` of external packages does not resolve the referenced action defs and part defs because the validator checks each file in isolation.
  Fix: Include forward declaration stubs (`action def <Name>;` and `part def <Name>;`) in the same package as the `allocate` statements. These stubs match the names used in the source packages but are resolvable locally. In a full model environment the imports would work; the stubs are a workaround for standalone validation.
