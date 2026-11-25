# Learned Fixes — General / Cross-Cutting SysML v2

This file accumulates cross-diagram fix patterns discovered by the SysML
agent — patterns that don't map cleanly to a single per-diagram skill
(`activity-diagram`, `block-definition-diagram`, etc.).

Use this file when:
- The same error + fix recurs across multiple diagram types.
- The fix is about top-level package syntax, imports, visibility, or
  standard-library usage (`ScalarValues`, `Quantities`, …).
- The fix is for a SysML v2 construct from a diagram type that has no
  dedicated `/skills/<diagram>/SKILL.md` yet.

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

## Session 2026-11-05
- No new fixes.

## Session 2026-11-06
- No new fixes.

## Session 2026-10-09
- Error: `Invalid syntax near 'loop' — no valid parsing rule matches` at
  `part loop : LoopP[1];`
  Fix: `loop` is a RESERVED keyword and cannot be used as a feature/part usage
  name (it also fails in `succession flow ... to loop.loopIn`). Rename the usage
  (e.g. `conveyorLoop`) while keeping the type name `LoopP`. Add `loop` to the
  reserved-keyword watch list.
- Error: `Invalid syntax near 'ordered' — no valid parsing rule matches` at
  `part ordered : OrderedP[1];`
  Fix: `ordered` is a RESERVED keyword and cannot be used as a part/feature name.
  Rename (e.g. `orderedPart`). Same class as `state`, `message`, `standard`,
  `action`, `variant`.
- Note: Reserved-keyword violations surface in the `local-syntax` fallback but
  are silently ACCEPTED by the `sidecar` backend — a file with `part loop : ...`
  passed `sidecar` cleanly. Always prefer the stricter interpretation; treat a
  `local-syntax` failure as a real defect even when `sidecar` reports success.

## Session 2026-10-08
- Note: BDD association blocks ("interfaces" in an input artifact) map cleanly to `connection def <Name> { end a : 'Type A'; end b : 'Type B'; attribute ...; }` at package level — validated with up to 6 `end` declarations (H2 Process Interface) and with `String`/`Real` attributes carrying the interface properties. This extends the 2026-07-09 `connection def` entry to multi-end associations. Association blocks must be declared BEFORE the part defs they reference; `part def` types referenced from `end` must be declared in the same package (unresolved-reference warning otherwise).

## Session 2026-10-07
- No new fixes.

## Session 2026-10-04
- Error: `Unrecognized character or sequence: '°'` at `attribute tempMin : TemperatureValue = -10.0 [°C];` — the Unicode degree symbol in the `[°C]` unit bracket is not recognized by the validator (same class as the `²`/`³` superscript issue).
  Fix: Use the Kelvin unit instead: `attribute tempMin : TemperatureValue = 263.15 [K];` / `attribute tempMax : TemperatureValue = 323.15 [K];` for a -10..+50 C operating range, or `[degC]` if a Celsius unit must be shown. Applies to any diagram with temperature attributes.

## Session 2026-09-21
- Error: `Couldn't resolve reference to Element 'm2'.` / `Couldn't resolve reference to Element 'm3'.` — the ASCII-digit unit forms `[m2]` and `[m3]` (previously recommended in the 2026-09-08 entry as a replacement for `[m²]`/`[m³]`) are ALSO unresolvable in the validator's SI library.
  Fix: Use explicit multiplication in the unit bracket: `[m*m]` for area and `[m*m*m]` for volume (e.g. `attribute baufenster_flaeche : Real = 1750.0 [m*m];`). This supersedes the 2026-09-08 entry. Applies to every diagram type with area/volume attributes.

## Session 2026-07-30
- Error: `WARNING: Couldn't resolve reference to Feature 'MassValue'/'LengthValue'/'TorqueValue'/'RotationalSpeedValue'/'ForceValue'/'FrequencyValue'` — the ISQ quantity types are not bundled in the validator's library set, so unit-typed attributes produce semantic warnings.
- Fix: These are warnings, not errors — the file still validates with zero syntax and zero semantic errors. Keep the quantity types (MassValue, LengthValue, etc.) as the task requires; the warnings are benign standalone-validation artifacts and do not require replacing with `Real`.

## Session 2026-07-09
- Error: `Duplicate definition 'associate': already defined as attribute_usage` — the `associate { end ...; end ...; }` syntax at the package level is rejected by the validator because `associate` is treated as a duplicate attribute_usage name.
  Fix: Replace `associate { end a : A; end b : B; }` with `connection def AssociationName { end a : A; end b : B; }` to model binary associations between part defs. This applies across all diagram types (BDD, IBD, etc.) when declaring association relationships between types.

## Session 2026-07-05
- Error: `Invalid syntax near 'state' — no valid parsing rule matches` — `state` is a reserved keyword in SysML v2 (used for state machine constructs) and cannot be used as an attribute name inside an item def.
  Fix: Renamed the attribute from `state` to `buttonState` inside `ButtonPressSignal` item def. This is the same class of reserved-keyword issue as `message`, `standard`, `action`, `variant`, etc. documented in prior sessions. Other known reserved keywords to add to the watch list: `state`, `transition`, `fork`, `join`, `entry`, `exit`, `do`.

## Session 2026-07-04
- Error: `Couldn't resolve reference to Element 'USD'.` — `USD` is not a valid unit symbol in the SI library. The SI library (imported via `public import SI::*;`) provides standard SI units (m, kg, s, A, K, mol, cd, Hz, N, Pa, J, W, V, etc.) but not currency units.
  Fix: Use bare `Real` for currency values without `[USD]` unit brackets: `attribute retailPrice : Real = 25.0;` instead of `attribute retailPrice : Real = 25.0 [USD];`. This applies across all diagram types when declaring monetary attributes.

## Session 2026-07-04
- Error: `Invalid syntax near 'standard' — no valid parsing rule matches` — `standard` is a reserved keyword in SysML v2 and cannot be used as an attribute name.
  Fix: Rename attributes from `standard` to `wifiStandard`, `nfcStandard`, or any non-reserved name. This applies across all diagram types when declaring attributes, parameters, or items. Other known reserved keywords include: `standard`, `message`, `variant`, `transition`, `analyze`, `detect`, `limit`, `activate`, `feedback`, `done`, `fork`, `join`, `decide`, `start`, `notify`, `signal`, `wait`, `receive`, `create`, `eval`, `check`, `restrict`, `continue`, `action`.

## Session 2025-06-25
- Error: `Unexpected ':', expecting one of: default, :=, ;... Extra token 'out' found` in port definitions
  Fix: In SysML v2, port direction is specified on the items inside the `port def`, not on the port declaration itself. Use `port def MyPort { out item signal : SignalType; }` not `port myPort : out MyPortType;`.
- Pattern: When defining ports on part usages in IBD, ensure all port names within the same enclosing part def are unique by using a naming convention that includes the part name prefix.

## Session 2026-06-20
- Error: `Couldn't resolve reference to Feature 'LengthValue'` — `LengthValue` and `MassValue` from `ISQ::*` are not available in the validator's library bundle. The ISQ library types (LengthValue, MassValue, TimeValue, etc.) are documented in the SysML v2 spec (ISO 80000) but not bundled with this validator.
  Fix: Replace `LengthValue` with `Real` and `MassValue` with `Real`. Units like `[mm]` and `[kg]` can still be applied to `Real` typed attributes. This is a cross-cutting fix that applies to any diagram using ISQ quantity types (parametric diagrams, BDDs with unit-typed attributes).

## Session 2026-06-18
- Error: `Extra token '~' found, expecting one of: behavior, crosses, feature...` — conjugate port syntax `port def InPort :> ~OutPort` is rejected by the validator. The `~` conjugate operator is not supported in port redefinition by this validator.
  Fix: Use `port def InPort :> OutPort` without the `~` operator. Port direction must be managed through naming convention rather than syntactic conjugation. This applies across all diagram types when defining port subtypes.

## Session 2026-06-18
- Error: `Extra token '&' found` in require constraints — `&&` is not valid SysML v2 constraint syntax.
  Fix: Use the keyword `and` instead of `&&` for boolean conjunction in require/assume constraints. This applies across all diagram types when writing formal constraint expressions.

## Session 2025-06-16
- Error: `Unexpected 'storageAisle', expecting one of: default, :=, ;...` — `stereotype` is NOT a valid keyword in SysML v2 textual syntax. SysML v1's `«stereotype»` mechanism does not exist in v2.
  Fix: Use `metadata def <Name> { ... }` instead of `stereotype <Name> specializes <Base>;`. Stereotypes in v2 are modelled as metadata definitions.
- Error: `Unexpected 'LengthValue', expecting one of: default, :=, ;...` — `typedef <Name> : <Type>;` is NOT valid SysML v2 syntax. The `typedef` keyword is not recognized.
  Fix: Use `attribute def <Name> :> <Type>;` to define a value type (e.g., `attribute def LengthValue :> Real;`). This creates a named subtype of the base type that can be used as an attribute type elsewhere.

## Session 2025-06-16
- Error: `Couldn't resolve reference to Feature 'WarehouseSystem'` — semantic warnings about unresolved cross-package references (e.g., `:> WarehouseSystem` when importing from another package) are benign when the imported package is defined in a separate file. The validator checks each file in isolation and cannot resolve references to types in other files that haven't been loaded in the same validation context.
  Fix: No fix needed — these are warnings, not errors. The files are valid when loaded together in a model environment. Always check that the `public import` statement is correct and the referenced type exists in the imported package.

## Session 2025-06-16
- Error: `Invalid syntax near 'message' — no valid parsing rule matches` — `message` is a reserved keyword in SysML v2.
  Fix: Rename the attribute from `message` to `commandContent` or any non-reserved name. This applies across all diagram types when declaring attributes, parameters, or items. Other known reserved keywords include: `variant`, `transition`, `analyze`, `detect`, `limit`, `activate`, `feedback`, `done`, `fork`, `join`, `decide`, `start`, `notify`, `signal`, `wait`, `receive`, `create`, `eval`, `check`, `restrict`, `continue`, `action`.

## Session 2025-04-18
- Error: `Unrecognized character or sequence: 'ä'` — Unicode characters (German umlauts ä, ö, ü, ß) are not supported in SysML v2 identifiers by the validator.
  Fix: Use ASCII-only identifiers for all SysML v2 element names. Replace umlauts with digraphs: ä→ae, ö→oe, ü→ue, ß→ss. This applies to part def names, port def names, item def names, requirement def names, attribute names, subject names, actor names — any identifier in the SysML v2 code. Doc comments (`doc /* ... */`) can still contain umlauts.

## Session 2025-04-18
- Warning: `'RedefinesContext' object is not iterable` — benign semantic warning triggered by `:>>` redefinition bindings in action usage parameter mappings inside composite action defs. This is a validator internal issue, not a user-code error. Files with this warning are syntactically and semantically valid. No fix needed.

## Session 2026-06-25
- Error: `Couldn't resolve reference to Element 'kg'.` — the `[kg]` unit symbol in attribute defaults (e.g., `attribute maxWeight : Real = 1000.0 [kg];`) requires the SI library to be imported.
  Fix: Add `public import SI::*;` to the package imports. The `ScalarValues::*` import provides the base types (Real, Integer, String, Boolean) but not the SI unit symbols. This applies across all diagram types that use unit-annotated attribute defaults (BDD, IBD, parametric, analysis case).

## Session 2026-09-08
- Error: `Unrecognized character or sequence: '²'` at `attribute maxFloorArea : Real = 2000.0 [m²];` — Unicode superscript characters (², ³, etc.) are not recognized in SI unit brackets by the validator.
  Fix: Use ASCII-safe unit notation: `[m2]` instead of `[m²]`, `[m3]` instead of `[m³]`. This is a cross-cutting fix that applies to any diagram type using unit-annotated attribute defaults with area/volume units.

## Session 2026-10-03
- Error: `Invalid syntax near 'protected' — no valid parsing rule matches` — `protected` is a reserved keyword in SysML v2 and cannot be used as an attribute name inside item defs.
  Fix: Renamed the attribute from `protected` to `isProtected` inside `NibState` item def in activity diagram.
