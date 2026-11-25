# Learned Fixes — SysML v2 Verification Case Diagram

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

## Session 2026-10-07
- No new fixes.
- Error: `Duplicate definition 'testMethod': already defined as attribute_usage` / `Duplicate definition 'passCriteria': already defined as attribute_usage` — the validator hoists attributes from all `objective` blocks of a `verification def` into one scope, so reusing the same attribute names (e.g. `testMethod`, `passCriteria`) across objectives collides.
  Fix: Give each objective's attributes unique names (`testMethodBraking`, `passCriteriaDetection`, ...) while keeping the per-requirement values unchanged. The two-word `verification case def` form was rejected again ("Invalid syntax near 'case'"); the single-token `verification def` / `verification <name> : <Def>` form validated cleanly as previously recorded.

## Session 2026-08-04
- Error: `WARNING: Subject 's' is a builtin. Subjects typically reference part or item definitions.` when the requirement/verification subject is named `s`.
  Fix: Rename the subject from `s` to a non-builtin name (e.g. `sys`). The single-token `verification def` / `verification ... : Def` keyword form (from the 2026-08-01 entry) and `VerificationCases::*` import both validated cleanly.

## Session 2026-08-03
- No new fixes.

## Session 2026-08-02
- No new fixes.

## Session 2026-08-01
- Error: `Invalid syntax near 'case' — no valid parsing rule matches` at `verification case def Name { ... }` and `verification case Name : Def { ... }`.
  Fix: The validator accepts the single-token keyword `verification def Name { ... }` and usage `verification Name : Def { subject = ...; }` — NOT the two-word `verification case`. Use `verification def` / `verification` in the emitted file.
- Error: `Unexpected 'then', expecting one of: ;,` on `first a then b then c;` — the validator rejects a multi-`then` chain in one `first` clause.
  Fix: Split into one `first X then Y;` clause per adjacent pair: `first a then b; first b then c; first c then d;`.
