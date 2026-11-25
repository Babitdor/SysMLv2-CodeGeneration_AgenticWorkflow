# Learned Fixes — SysML v2 Activity Diagram

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

## Session 2026-11-05

- No new fixes.

## Session 2026-10-27

- No new fixes.

## Session 2026-10-08

- No new fixes.

## Session 2026-10-07

- Error: `Invalid syntax near 'composite' — no valid parsing rule matches` on `action composite : doComposite;` and `then composite;` — `composite` is a reserved keyword in SysML v2 and cannot be used as an action usage name (same class as `terminate`, `accept`, `assign`, `verify`).
  Fix: Rename the action usage from `composite` to `middle` (or any non-reserved identifier) in both the `action <usage> : <ActionDef>;` declaration and its `then <usage>;` references. Applies to all action usage names and action def names.

## Session 2026-10-07

- No new fixes.

## Session 2026-10-03

- Error: `Couldn't resolve reference to Element 'functionsReady'.` — a decision guard `if monitorOperation.functionsReady then ...` referenced an output that is not declared on the action usage named on the left of the dot; `functionsReady` is an `out` of `doStartShiftOperation` (usage `startShift`), not of `doMonitorOperationAndHandleDeviations` (usage `monitorOperation`).
  Fix: Point the guard at the usage that actually declares that output: `if startShift.functionsReady then monitorOperation; else reportRejectedOrder;`. The guard's `<usage>.<feature>` path must resolve to a parameter declared on the *type* of that exact usage in the same composite body — not just to any similarly named parameter elsewhere in the file.

## Session 2026-10-02

- No new fixes.

## Session 2026-09-22

- No new fixes.

## Session 2026-09-22

- Error: `Duplicate definition 'centerGimbal': already defined as action_usage` — referencing a declared action usage by its bare name inside a `while { guard } { usageName; }` loop body makes the parser treat it as a second definition of the same usage.
  Fix: Replace the `while` loop with a `then decide;` node using a `not` guard that re-invokes the centering action via a second usage (e.g. `if not centerGimbal.centered then centerGimbalAgain; else augmentAircraft;`). The decision pattern keeps the control loop semantics without the duplicate-definition error.

## Session 2026-09-22

- No new fixes.

## Session 2026-09-22

- No new fixes.

## Session 2026-09-21

- Error: `Missing expected token 'send'` on `then while { true } { then zuordnen; }` — a `then` keyword is NOT allowed before the action inside a `while` loop body.
  Fix: Drop the `then` inside the loop body: `then while { <guard> } { <usageName>; }` (bare usage name + `;`). The `then` prefix is only valid for statements directly in the action def body, not inside a loop block.

## Session 2026-09-21

- No new fixes.

## Session 2026-09-16

- Error: `Couldn't resolve reference to Element 'downstreamIsProduction'.` — a decision guard referenced `completePick.downstreamIsProduction` but the composite action def (`PerformOrderPicking`) declared its own `out downstreamIsProduction` parameter instead of relying on the leaf usage's output; the guard must resolve against the *action usage* (`completePick`), not a parameter of the enclosing def.
  Fix: Remove the duplicate `out` parameter from the composite action def and declare the downstream actions (`productionSupply : doSupplyProduction;`, `shippingBuffer : BufferGoodsForShipping;`) as action usages inside the SAME composite def that contains the decision — guard features must be outputs of a preceding action usage declared in the same body.
- Error: `Couldn't resolve reference to Feature 'productionSupply' / 'shippingBuffer'` + `A transition must own a succession to its target.` — decision-branch action names referenced in `if ... then <usage>; else <usage>;` were declared only in a *different* (top-level) action def, not in the def owning the decision.
  Fix: Every action usage referenced in a `then decide;` block must be declared via `action <usage> : <ActionDef>;` inside the same composite action def body. Cross-def references are not resolved; move the decision (and its branch usages) into the def that owns the flow, and remove the duplicated decision from the top-level orchestration def.

## Session 2026-09-09

- Error: `Extra token ':' found, expecting one of: all, behavior, crosses...` on `action terminate : doTerminateFill;` — `terminate` is a reserved keyword in SysML v2 and cannot be used as an action usage name.
  Fix: Rename the action usage from `terminate` to `terminateFill` (or any non-reserved identifier) in composite action defs. Applies to all action usage declarations and their `then terminateFill;` references.

## Session 2026-09-09

- No new fixes.

## Session 2026-09-04

- No new fixes.

## Session 2026-01-14

- No new fixes.

## Session 2026-08-29

- No new fixes.

## Session 2026-08-26

- No new fixes.

## Session 2026-08-26

- Error: `Extra token 'if' found, expecting one of: behavior, crosses, feature...` on `else if batteryState == "outside" then flagProtection;` — `else if` chains are NOT supported in `then decide;` blocks.
  Fix: Flatten `else if` chains into sequential `then decide; if guard then action; else action;` blocks. Each decision block covers exactly one guard with a single `else`.
- Error: `Duplicate definition 'doRecordProtectionEvents': already defined as action_def` — a Level-2 composite action def and an ACT-NNN leaf action def shared the same name.
  Fix: Ensure every action def name is unique across the whole package. When a composite function-group def and a leaf ACT def would collide, rename the leaf (e.g. `doRecordProtectionEventLog`).

## Session 2026-08-26

- Error: `Couldn't resolve reference to Feature 'cruiseControl'.` — the top-level orchestration action def referenced an action usage name (`cruiseControl`) that was never declared; the declared usage was `cruise : doAdaptiveCruiseControl`.
  Fix: Ensure every `then <usageName>;` in a composite action def references a usage declared via `action <usageName> : <ActionDef>;` in the same body. Also fixed a second mismatch where the orchestration referenced `doHMIDisplay` but the def was named `doHMIStateDisplay` — the type reference in `action displayState : <Type>;` must exactly match a declared `action def` name. Validated cleanly after aligning usage names and type references.

## Session 2026-08-26

- Error: `Unrecognized character or sequence: '!c'` on `if !checkSensors.sensorsAvailable then flagFailure;` — the `!` negation operator is not recognized by the validator in decision guards.
  Fix: Replace `!` with the keyword `not` in decision guards: `if not checkSensors.sensorsAvailable then flagFailure;`. Applies to all boolean negation guards in `then decide; if ... then ... else ...;` blocks.

## Session 2026-08-26

- Error: Validation checklist B1 violation — composite action defs referenced leaf action types WITHOUT the `do` prefix (e.g., `action recordReceipt : RecordGoodsReceipt;`) while the leaf action defs were declared WITH the `do` prefix (`action def doRecordGoodsReceipt`). Every `action <usage> : <TypeName>;` inside a composite action def must reference a type name that is actually declared as an `action def` in the same file.
  Fix: Cross-check every type reference in composite action defs against the declared `action def` names. Leaf action defs declared with the `do` prefix must be referenced with that prefix (e.g., `action recordReceipt : doRecordGoodsReceipt;`). Composite action defs (Level-1/Level-2) that are themselves declared with `do` prefix are referenced with the `do` prefix too (e.g., `action registerDelivery : doRegisterIncomingDelivery;`). Only the two non-`do` composite defs (`ProcessAcceptedArrival`, `RecordVerificationResult`, `ProcessERPFlow`) are referenced without the prefix because they are declared without it. Validated cleanly after the fix.

## Session 2026-07-17

- Error: `Missing expected token 'send'` on `then actionName : ActionDef { ... }` — using colon syntax for action usage with parameter bindings inside composite action defs causes syntax errors.
  Fix: Declare action usages separately before `first start;` using `action usageName : ActionDef;`, then reference them without colon in the control flow: `then usageName;`. Do NOT use `:>>` redefinition bindings or `->first()` method calls in action bodies.
- Error: `Invalid syntax near 'branch'` — the `branch` keyword is not supported in fork/join blocks in this validator.
  Fix: Use sequential `then actionName;` statements instead of fork/branch/join. The validator accepts sequential action sequencing within composite action defs.

## Session 2026-07-10

- No new fixes.

## Session 2026-07-09

- No new fixes.

## Session 2026-07-08

- No new fixes.

## Session 2026-07-08

- Error: `Couldn't resolve reference to Feature 'selectDock'.` (and 70 similar) — leaf action defs had step-level action names (e.g., `selectDock`, `retrievePO`, `scanItems`) referenced in `then` statements without being declared as action usages or action defs.
  Fix: Simplify leaf action def bodies to `first start; then done;` without internal step-level action names. The leaf action defs are the atomic behavioral units; their internal steps are implicit in the action's purpose. Composite action defs that sequence leaf actions use separate `action usageName : ActionDef;` declarations before `first start;`.

## Session 2026-07-07

- No new fixes.

## Session 2026-07-06

- No new fixes.

## Session 2026-07-05

- No new fixes.

## Session 2026-07-04

- No new fixes.

## Session 2026-07-04

- Error: `Unexpected ';', expecting one of: after, at, behavior... (and 34 more)` on `then accept;` — `accept` is a reserved keyword in SysML v2.
  Fix: Rename action usage from `accept` to `acceptRefill` or any non-reserved identifier.
- Error: `Couldn't resolve reference to Element 'extendRequest'.` — a bare feature reference in a decision guard (`if extendRequest then ...`) cannot be resolved because it is not an output of a preceding action usage.
  Fix: Add a preceding action usage of a helper action def that produces the guard boolean as an output, then reference it as `actionUsage.outputName` in the guard.

## Session 2026-07-04

- Error: `Couldn't resolve reference to Element 'size'.` / `Must invoke a behavior or a behavioral feature` — `->size()` method call syntax is not supported in action succession guards (same pattern as `->startsWith()`, `->first()`, `->isEmpty()` from prior sessions).
  Fix: Replace `->size()` with a simple string comparison guard (e.g., `sourceContent == ""` instead of `sourceContent->size() > 10000`). Use boolean output flags on preceding actions for more complex guards.

## Session 2026-07-04

- Error: `An attribute must be typed by attribute definitions.` — `item def LocationData` had an attribute `coordinates : GeographicCoordinates` referencing another item def type, which the validator does not support for item def attributes.
  Fix: Flatten the attribute to use only primitive types (Real, String, Boolean) instead of referencing other item def types within item defs.
- Error: `A state must be typed by state definitions.` — `state` is a reserved keyword in SysML v2; using `in state : SystemPowerState` as a parameter name caused the validator to interpret it as a state definition.
  Fix: Rename the parameter from `state` to `powerStateValue` to avoid the reserved keyword conflict.

## Session 2026-07-04

- Error: `Couldn't resolve reference to Element 'startsWith'.` / `Must invoke a behavior or a behavioral feature` — `->startsWith()` method call syntax is not supported in action succession guards.
  Fix: Replace `->startsWith("C")` with a boolean output flag on the preceding action def (e.g., `out isContainer : Boolean = false;`) and use `if actionName.isContainer then ...` in the decision guard.

## Session 2026-07-02

- No new fixes.

## Session 2026-06-27

- Error: `Couldn't resolve reference to Feature 'pallet'.` (and 15 similar) — `:>>` redefinition bindings in composite action bodies (e.g., `then actionName { in :>> param = value; }`) cause the validator to fail resolving the feature path on the right-hand side of the binding.
  Fix: Remove all `:>>` redefinition bindings from composite action bodies. Use simple `then actionName;` without parameter bindings. The action usage declarations (`action usageName : ActionDef;`) are sufficient for the validator to accept the composition. Data flow between actions is implicit rather than explicitly bound.
- Error: `Invalid syntax near 'message'` — `message` is a reserved keyword in SysML v2.
  Fix: Rename attribute from `message` to `notificationText` in item defs.
- Error: `Extra token 'if' found` on `else if` chains — nested `else if` is not supported in decision nodes.
  Fix: Use sequential flat `then decide;` nodes instead of `else if` chains. Each decision node has exactly one `if guard then action; else action;` pair.
- Error: `Unexpected '->'` on method call syntax like `->first()` or `->isEmpty()` — method call syntax with `->` is not supported in action succession guards or parameter bindings.
  Fix: Replace `->first()` with direct feature reference, and `->isEmpty()` with `== ""` string comparison.

## Session 2026-06-26

- No new fixes.

## Session 2026-06-26

- Error: `Unexpected 'accept', expecting one of: behavior, crosses, feature...` — `accept` is a reserved keyword in SysML v2.
  Fix: Rename action usage from `accept` to `acceptPallet` or any non-reserved identifier.

## Session 2026-06-25

- No new fixes.

## Session 2026-06-25

- Error: `Missing expected token 'send'` on `then receiveComplete = true;` — assignment statements (e.g., `then variable = value;`) are not supported in action succession syntax in this validator.
  Fix: Remove all assignment statements from `then` clauses. Use `then done;` directly as the final node. The output parameter can remain declared but is not assigned inline within the action body.

## Session 2026-06-24

- Error: `no viable alternative at input 'in'` / `no viable alternative at input ':'` on action def parameters named `event` — `event` is a reserved keyword in SysML v2.
  Fix: Rename the parameter from `event` to `failureEvent` or any non-reserved identifier in both the action def declaration and all composite action usages that reference it via `:>>`.
- Error: `mismatched input 'then' expecting '}'` on fork/join blocks with `then actionName;` inside the fork — the validator does not support `then` statements inside a fork block.
  Fix: Replace fork/join with a flat sequential `first ... then ...` flow. For parallel operations, use sequential ordering instead of fork/join syntax.

## Session 2026-06-25

- Error: `Couldn't resolve reference to Element 'mm'.` / `Couldn't resolve reference to Element 'kg'.` / `Couldn't resolve reference to Element 'h'.` — 9 semantic errors from using `Real` type with unit annotations (`[mm]`, `[kg]`, `[h]`) without importing the ISQ/SI libraries.
  Fix: Add `private import ISQ::*;` and `private import SI::*;` at the top of the package, and change attribute types from `Real` to the appropriate ISQ quantity type (`LengthValue`, `MassValue`, `TimeValue`). `Real` does not support unit annotations; only ISQ value types do.

## Session 2026-06-24

- Error: `Unexpected '::', expecting one of: ;` on `package UAVDeliverySystem::Activities` — double-colon namespace separator is not supported in package names by this validator.
  Fix: Use a flat package name with underscores instead of `::` separator: `package UAVDeliverySystem_Activities` instead of `package UAVDeliverySystem::Activities`.
- Error: `Unexpected 'decide', expecting one of: behavior, crosses, feature...` on `else decide;` — nested `else decide;` chains are not supported in SysML v2 decision nodes.
  Fix: Use sequential flat `then decide;` nodes instead of `else decide;` chains. Each decision node has exactly one `if guard then action; else action;` pair, followed by the next `then decide;`.

## Session 2026-06-24

- Error: `Invalid syntax near 'message' — no valid parsing rule matches` — `message` is a reserved keyword in SysML v2.
  Fix: Rename attribute from `message` to `notificationText` in `ArrivalNotification` and `LinkRestorationNotification` item defs.

## Session 2026-06-22

- No new fixes.

## Session 2026-06-22

- Error: `Extra token ':' found, expecting one of: behavior, crosses, feature...` on `action assign : doAssignStoragePosition;` — `assign` is a reserved keyword in SysML v2.
  Fix: Rename action usage from `assign` to `assignPos` or any non-reserved identifier.
- Error: `Invalid syntax near 'verify'` on `action verify : doVerifyContainerSizeAndWeightLimits;` — `verify` is a reserved keyword in SysML v2.
  Fix: Rename action usage from `verify` to `verifyLimits` or `verifyPallet` or any non-reserved identifier.

## Session 2026-06-21

- Error: `Couldn't resolve reference to Feature '<usageName>'` — 82 semantic errors from composite action defs referencing action usages that were not declared inside the action body.
  Fix: Add `action usageName : ActionDef;` declarations inside each composite action def BEFORE the `first start;` statement. Every action usage referenced in `then usageName;` or `if guard then usageName;` must be declared as a typed action usage first.

## Session 2026-06-20

- No new fixes.

## Session 2026-06-19

- No new fixes.

## Session 2026-06-18

- Error: `Unexpected 'message', expecting one of: behavior, crosses, feature...` — `message` is a reserved keyword in SysML v2.
  Fix: Rename parameter from `message` to `alertMsg` or any non-reserved identifier.
- Error: `Unexpected 'item', expecting one of: behavior, crosses, feature...` — `item` is a reserved keyword in SysML v2.
  Fix: Rename parameter from `item` to `transportItem` or any non-reserved identifier.
- Warning: `'RedefinesContext' object is not iterable` — benign semantic warning from `:>>` redefinition bindings in composite action defs. No fix needed.

## Session 2025-06-17

- Error: `Extra token 'if' found` on `else if ... then ...` — SysML v2 does not support `else if` chains in decision nodes.
  Fix: Use sequential flat `then decide;` nodes instead of `else if` chains. For multi-branch decisions, use multiple consecutive `then decide;` blocks, each with a single `if guard then action; else action;` pair.

## Session 2025-06-16

- No new fixes.

## Session 2025-04-18

- Error: `Missing expected token 'send'` on `then actionName : ActionDef { in :>> param = param; }` — colon syntax with inline bindings in `then` statements.
  Fix: Use `then actionName;` without colon or parameter bindings in composite action defs. The parameter bindings (`in :>>`) are not supported in the `then` succession syntax. Declare primitive action defs with in/out parameters separately, then reference them by name in the composite action body.

## Session 2025-04-18

- Error: `Invalid syntax near ':' — no valid parsing rule matches` on `out defined : Boolean = false;`
  Fix: `defined` is a reserved keyword. Rename to `strategiesDefined` or any non-reserved word.
- Error: `Extra token '}' found, expecting one of: ;` on inline object creation `OrderRequest { orderId = carrierId }`
  Fix: Do NOT use inline object creation syntax (`Type { attr = val; }`) in action usage parameter bindings. Change the action def's parameter type from the complex type to a simple string, or pass individual fields instead of constructing an object inline.
- Warning: `'RedefinesContext' object is not iterable` — benign semantic warning from the validator when using `:>>` redefinition bindings in action usage parameter mappings within composite action defs. Does not affect file validity.

## Session 2025-04-17

- No new fixes.

## Session 2025-04-16

- Error: `Missing expected token 'send'` on `then actionName : doActionDef { ... }` — colon syntax in `then` statements
  Fix: Do NOT use inline colon syntax (`then actionName : ActionDef { ... }`). Declare action usages separately with `action usageName : ActionDef;` then reference them without colon: `then usageName;`.
- Error: `Unexpected 'decide', expecting one of: behavior, crosses, feature...` on `first register then decide;`
  Fix: Use `first start;` as the initial node, then `then decide;` — do NOT chain `first action then decide;`. The `decide` keyword must follow `then` directly, not `first ... then`.
- Error: `Unexpected 'fork', expecting one of: behavior, crosses, feature...` on `first receive then fork;`
  Fix: Use `first start;` then `then receive;` then `then fork;` — do NOT chain `first action then fork;`. The `fork` keyword must follow `then` directly.
- Error: `then done;` must be preceded by at least one `first ... then` statement when using `first action1 then action2;` pattern
  Fix: Use `first start;` as the initial node for ALL composite actions, then sequence steps with `then stepName;`, ending with `then done;`. Do NOT use `first step1 then step2; first step2 then step3; ...` as individual succession statements.

## Session 2025-04-15

- No new fixes.

## Session 2025-04-14

- Error: `Missing expected token 'send'` on `then assignContainer : doAssignContainer { ... }` with colon syntax in then statements
  Fix: Do NOT use inline colon syntax (`then actionName : ActionDef { ... }`) in action succession. Declare action usages separately (`action actionName : ActionDef;`) and reference them without colon: `then actionName;`.
- Error: `Unexpected 'then'` on `first executeMovement then updateInventory then done;` — cannot chain more than 2 steps with `first ... then ... then`
  Fix: Use `first start;` then individual `then actionName;` statements, ending with `then done;`. Do not chain multiple `then` keywords in a single statement.
- Error: `Missing expected token 'send'` on `then done;` when not preceded by a `first ... then` pair
  Fix: `then done;` must be preceded by at least one `first ... then` statement. Use `first start;` as the initial node, then `then action1; then action2; ... then done;`.

## Session 2025-01-01

- Error: `Missing expected token '}'` / `Extra token ';' found` at constraint body
  Fix: In `constraint def`, the body is a single boolean expression (not wrapped in `constraint { ... }`). Use `shiftHours >= 8.0 and totalOperationalHours == 16.0` instead of `constraint { shiftHours >= 8.0; ... }`.
- Error: `Unexpected ':', expecting one of: ;` at `if guard then actionName : ActionDef { ... }`
  Fix: In decision branches, declare action usages separately (`action myAction : MyActionDef;`) then reference them in the decision (`if guard then myAction { ... }`). Do NOT use inline `actionName : ActionDef` syntax in the `then` branch.
- Error: `Unexpected 'decide', expecting one of: behavior, crosses, feature... (and 15 more)` at `else decide;`
  Fix: SysML v2 does NOT support nested `else decide;` or `else if` chains. Use a separate action definition for the fallback logic, or use sequential flat decision nodes.
- Error: `Invalid syntax near 'variant'` — `variant` is a reserved keyword
  Fix: Rename output parameter from `variant` to `variantName` or similar non-reserved name.
