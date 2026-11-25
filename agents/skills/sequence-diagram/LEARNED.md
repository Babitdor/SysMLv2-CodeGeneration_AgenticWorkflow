# Learned Fixes — SysML v2 Sequence Diagram (succession + flow)

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
- No new fixes.

## Session 2026-10-09
- Note: The 2026-06-20 / 2025-06-16 entries about `perform action` do NOT
  generalise to the plain `action` keyword. Verified with a full `sidecar`
  compiler run: `action <name> : <ActionDef>;` inside a `part def` body, ordered
  by `first <name> then <other>;` in the SAME part def body, compiles with zero
  syntax and zero semantic errors, as does the same construct inside a composite
  `action def` (including nested composite action defs, `accept signal : T;`
  bodies, and `succession flow ... from a.port to b.port` between sibling parts).
  Prefer `action <name> : <ActionDef>;` + `first/then` at participle level for
  sequence diagrams built on `succession flow`; the `perform` PREFIX remains the
  only rejected form.
- Note: `succession flow <name> of <ItemDef> from a.<outPort> to b.<inPort>;` is
  accepted by `sidecar` for every direction combination (in-to-in, out-to-out,
  inout-to-inout, inout-to-in, out-to-signal-port) and for item types that do not
  match the target port's declared item. Direction/type mismatches are therefore
  not validator errors — model the correct directions and item types anyway for
  readability, since the validator will not catch them.

## Session 2026-07-12
- No new fixes.

## Session 2026-07-11
- No new fixes.

## Session 2026-07-10
- No new fixes.

## Session 2026-06-20
- Error: `Couldn't resolve reference to Feature '<actionUsageName>'` — `perform action <name> : <ActionDef>;` inside a `part def` causes "Couldn't resolve reference to Feature" errors. The `perform action` keyword is not supported by this validator inside `part def` bodies (though the skill examples show it).
  Fix: Use `action <name> : <ActionDef>;` instead of `perform action <name> : <ActionDef>;` inside `part def` bodies. The `action` keyword creates a valid action usage without the `perform` prefix, and resolves correctly in succession/flow contexts.

## Session 2026-06-18
- Error: `Extra token '~' found, expecting one of: behavior, crosses, feature...` — conjugate port syntax `port def InPort :> ~OutPort` is rejected by the validator, even though the skill examples show this pattern.
  Fix: Use `port def InPort :> OutPort` without the `~` conjugate operator. Port direction must be managed through naming convention (e.g., BinInPort, BinOutPort) rather than syntactic conjugation.

## Session 2025-06-16
- Error: `Couldn't resolve reference to Feature '<actionUsageName>'` — `perform action <name> : <ActionDef>` usage names cannot be resolved by `first <name> then <other>` or `succession first <name> then <other>` inside a `part def` body. The validator cannot resolve `perform action` usage names in succession statements.
  Fix: Move sequencing logic into composite `action def` blocks with internal `first A then B; first B then C;` statements (which work correctly). Use `part def` only for port declarations and `succession flow` for cross-part item transfers. Each `first/then` pair can only have one `then` — chain with `first A then B; first B then C;` not `first A then B then C;`.
- Error: `Unexpected 'then', expecting one of: ;,` — `first A then B then C` is not valid. The `first/then` construct only supports a single `then` per statement.
  Fix: Use multiple `first/then` pairs: `first A then B; first B then C;`
