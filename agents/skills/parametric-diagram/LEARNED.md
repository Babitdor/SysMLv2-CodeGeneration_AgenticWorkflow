# Learned Fixes — SysML v2 Parametric Diagram

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

## Session 2026-11-07
- Note (no error): rounding rules (ceiling/floor of a sourced quotient) validate cleanly as a two-sided
  Integer inequality against a Real quotient, e.g.
  `p_count >= p_value / p_divisor and p_count < p_value / p_divisor + 1.0` (ceiling) and
  `p_count * p_pitch <= p_length and (p_count + 1) * p_pitch > p_length` (floor).
  Use this instead of `ceil()`/`floor()` calls, which are not needed and would need library support.
  Mixing `Integer` parameters with `Real` parameters in the same constraint body produces no
  type warnings on the `sidecar` backend.
- Note: cross-part binding inside a system-level part def works with dot navigation on a nested part
  usage: `assert constraint cGate : SomeDef { in p_actual = mls.someAttribute; }` where
  `part mls : 'Sizing'[1];`. Same constraint-usage name (e.g. `cUsableRackHeight`) may be reused in
  different part defs without a duplicate-definition error.

## Session 2026-11-06
- Error: `Unexpected '(', expecting one of: ;,` at `assert constraint MyConstraint(` — the skill's `assert constraint <Def>(arg = value, ...)` named-argument form is rejected by both backends. Call count 36+ assertions is fine with the alternative.
  Fix: Use the constraint-USAGE form with a body of `in` bindings: `assert constraint someName : MyConstraintDef { in formalParam = actualAttribute; ... };`. This preserves named binding, keeps constraint usages named/inspectable, and validates with zero errors. The earlier recommendation of anonymous inline constraints (`constraint { expr }`) still works but loses the parameter binding; prefer the usage form whenever all bound features live on the owning part.
- Note: the older "Duplicate definition" error only occurs when a binding's formal name equals an attribute name in the same part def (name shadowing). Namespacing formals and attributes apart (`anzahl_gassen` vs `gassenbreite`) makes the usage form safe.
- Note: `ebenen == (hoehe - ebene) / pitch` style integer-vs-Real equality is accepted by `sidecar` without type warnings, so derived-value constraints can be written directly instead of encoding `floor()` as a two-sided inequality.

## Session 2026-06-20
- Error: `Couldn't resolve reference to Feature 'LengthValue'` (40 warnings) — `LengthValue` and `MassValue` from `ISQ::*` are not available in this validator's library bundle, even with `public import ISQ::*;`. These types are documented in the SysML v2 spec but not bundled with the validator.
  Fix: Replace `LengthValue` with `Real` and `MassValue` with `Real` for all attribute types and constraint parameter types. Units can still be specified with `[mm]` and `[kg]` on `Real` values. This is a validator limitation — in a full SysML v2 toolchain, `LengthValue` and `MassValue` would resolve correctly.

## Session 2026-06-18
- No new fixes.

## Session 2025-06-16
- Error: `Unexpected '(', expecting one of: ;,` — `assert constraint <Name>(arg1 = val1, arg2 = val2)` with parenthesized named argument syntax is not supported by this validator. The skill examples show this syntax but it fails validation.
  Fix: Use anonymous inline constraints (`constraint { expr }`) inside the part def body instead. For named constraint defs, use `constraint <name> : <ConstraintDef> { in <param> = <value>; }` syntax. When this also fails with "Duplicate definition" errors (because the `in` parameter names shadow the constraint def's parameters), fall back to anonymous inline constraints.
- Error: `Duplicate definition 'totalPositions': already defined as attribute_usage` — when using `constraint <name> : <ConstraintDef> { in <param> = <value>; }` inside a part def, the `in` parameter names in the constraint usage body cause duplicate definition errors if they match the constraint def's formal parameter names.
  Fix: Use anonymous inline constraints (`constraint { expr }`) instead. These work correctly and avoid the duplicate definition issue.
