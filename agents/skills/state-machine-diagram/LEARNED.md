# Learned Fixes — SysML v2 State Machine Diagram

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
- Error: `Couldn't resolve reference to Feature '<stateName>'.` on EVERY transition `then` target (plus a paired `Must have at least two related elements` at the same offset) when the substates of a `state def` machine are themselves declared as `state def <name>;`.
  Fix: Declare substates as `state <name>;` / `state <name> { do action X; }` USAGES inside the machine body and keep only the top-level machine as `state def`. Verified by A/B probe on the same backend: `state` substates + `first/then` transitions → PASS; `state def` substates + identical transitions → 4 errors (2 unresolved-Feature, 2 two-related-elements). This REVERSES the 2026-09-22 entry for the current sidecar backend — when both forms are seen in a batch, prefer `state` usages and treat the 2026-09-22 advice as backend-version-specific.
- Note: state-level `do action` (max one per state) and guarded/effected transitions (`first … accept e : Evt if e.field do action A then …`) both validate cleanly with `state`-usage substates; no transition-level `doc` member (still per 2026-11-06).

## Session 2026-11-06
- Error: `Invalid syntax near 'doc' — no valid parsing rule matches` (and the follow-on cascade `Unexpected 'accept', expecting one of: then`, `Unexpected 'if'`, `Unexpected 'then'`, `Extra token 'transition' found`) on a multi-line `transition <name>` body that began with a `doc /* ... */` line.
  Fix: A `transition` body does NOT accept a `doc` member — valid members are only `first`/`accept`/`if`/`do action`/`then`. Put traceability text in a plain `/* ... */` comment BEFORE the transition declaration (`/* ACTF-009 ... */` then `transition name first ... then ...;`). `doc /* ... */` remains valid on `state def`, `state <usage>` bodies, `item def` and `action def`. Also note the file must fit in one tool call: a `doc`-free transition body is required for the sidecar backend to accept the whole file.

## Session 2026-10-08
- No new fixes.

## Session 2026-09-22
- Error: `Couldn't resolve reference to Feature 'on'.` (and every other state name) on `transition first <src> then <tgt>` — the validator cannot resolve bare `state <name>;` usage substates as transition `then` targets, regardless of scope (`state def` body, `exhibit state` body, `:>` specialization, `: StateDef` typing). This is the same failure class previously logged as UNRESOLVED in Session 2025-06-16.
  Fix: Declare the substates as `state def <name>;` (keyword-form definitions) instead of bare `state <name>;` usages inside the `state def` body, then reference them in transitions exactly as before (`first off then on`). With `state def` substates the transitions resolve cleanly and the file validates with zero errors. Verified working for plain transitions, `accept e : Event` transitions, and the `exhibit state operationalStates :> OperationalStates;` binding on the part. This supersedes the 2025-06-16 UNRESOLVED entry.

## Session 2026-06-24
- Error: `A state may have at most one do action.` — multiple `do action` declarations in a single state body are rejected by the validator.
  Fix: Consolidate multiple behaviors into a single representative `do action` per state. Use `entry action` for the primary behavior and `do action` for the continuous monitoring/background behavior. Composite action defs with sub-actions (`first step1; then step2; ...`) can be used but sub-actions must be declared as bare `action name;` inside the action def body (not `:>` specializations) to avoid reference resolution errors.

## Session 2025-06-16
- Error: `Couldn't resolve reference to Feature '<stateName>'` — transitions using `first <src> then <tgt>` syntax fail semantically because the validator cannot resolve state names in the `then` clause, even when both states are declared in the same scope (whether `state def` body or `exhibit state` body). This is a validator limitation affecting all transition syntax variants tested (`first src then tgt`, `first src accept e : Event then tgt`, `first src then tgt` with `:>` specialization).
  STATUS: UNRESOLVED — The validator cannot resolve state names in transition `then` targets. Produce state machine diagrams with `state def`, `state` usages, and `entry action` behaviors, but omit explicit transitions. Document intended transitions in `doc` comments.
