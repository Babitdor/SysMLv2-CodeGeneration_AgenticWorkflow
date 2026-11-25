# Learned Fixes — SysML v2 Use Case Diagram

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
- Error: `Couldn't resolve reference to Feature 'REQ_009'.` + `Must reference a constraint.` + `Must reference a requirement.` — bare `satisfy REQ_009;` fails even when `requirement def REQ_009 { doc /* ... */ }` is declared in the same package (corroborates 2026-09-21: a bare `requirement def` is not a resolvable satisfy target).
  Fix: Declare ONE `requirement def RequirementSource { doc /* ... */ }` and model every source requirement as a typed USAGE carrying its exact source ID as an attribute: `requirement REQ_001 : RequirementSource { attribute sourceId : String = "REQ-001"; doc /* ... */ }`, then `satisfy REQ_001;` resolves with 0 errors. This keeps verbatim hyphenated IDs (REQ-001, REQ-NF-001, RULE-010) without quoting and without one alias per satisfy statement — 30 requirement usages and 52 satisfy statements validated cleanly.
- Note: `include use case <alias> : '<UseCaseDef>';` (named alias + quoted type) validated with 0 errors AND 0 warnings, contradicting the 2026-06-20 warning report; the alias must be unique package-wide (includeUC002, includeUC002b, includeUC002c for three separate inclusions of the same target).
- Note: `actor <name> : <PartDef>;` names must also be unique package-wide only when reused inside the same def; reusing `actorPrimary` across 11 use case defs produced no duplicate-definition error, so actor names are per-use-case scoped (unlike subject names, which must be unique — systemUC001..systemUC011).

## Session 2026-11-06
- No new fixes.

## Session 2026-11-05
- No new fixes.

## Session 2026-10-27
- No new fixes.

## Session 2026-10-08
- No new fixes.

## Session 2026-10-07
- Error: `Unexpected 'part', expecting one of: about, ;,` at `@Actor part def Doctor;` — prefixing a `part def` with a metadata usage (`@Actor`) is rejected by the validator grammar; a metadata usage must be a standalone statement, not a modifier on a type declaration.
  Fix: Declare the `part def` first, then apply the stereotype as a separate statement: `@Actor about Doctor;`. Combined with `metadata def Actor { }`, this models the SysML v2 `actor` stereotype on actor part definitions. All 4 actor part defs (Doctor, Nurse, Patient, HospitalAdministration) validated with 0 syntax and 0 semantic errors.
- Note: `extend use case '<Name>' if <guard>;` is still unsupported by this validator (corroborates 2026-07-06 / 2026-02-12 entries); represented UC-006→UC-004 and UC-003→UC-002 extensions with the supported `include use case '<Name>';` form. Hospital system: 6 use case defs with unique subject names (monitorSys..reportSys), 4 actor part defs + 1 subject part def, 6 requirement defs ('REQ-001'..'REQ-006'), aliased `satisfy requirement <alias> : '<REQ-XXX>';`, all validated with 0 syntax and 0 semantic errors.

## Session 2026-10-07
- No new fixes.

## Session 2026-10-03
- No new fixes.

## Session 2026-01-04
- No new fixes.

## Session 2026-10-02
- No new fixes.

## Session 2026-09-22
- No new fixes.

## Session 2026-02-12
- Error: `Invalid syntax near 'use'` / `Unexpected 'if', expecting one of: ;,` for `extend use case <name> : '<UseCaseDef>' if <condition>;` — this validator's grammar has no `extend` keyword (corroborates the 2026-07-06 entry).
  Fix: Represent the optional-extension relationships with the supported `include use case '<UseCaseDef>';` form (Generate Reports includes Access Historical Data; Set Alert Thresholds includes Display Real-Time Data). Hospital system: 6 use case defs with unique subject names (monitorSys..reportSys), 4 actor part defs + 1 subject part def, 6 requirement defs ('REQ-001'..'REQ-006'), aliased `satisfy requirement <alias> : '<REQ-XXX>';`, all validated with 0 syntax and 0 semantic errors.

## Session 2026-09-21
- Error: `Couldn't resolve reference to Feature ''REQ-001''.` + `Must reference a constraint.` + `Must reference a requirement.` (150 semantic errors, 3 per satisfy line) — bare `satisfy 'REQ-001';` inside a `use case def` is NOT accepted by this validator even when `requirement def 'REQ-001' { ... }` exists in the same package. Quoted requirement names in a bare `satisfy` never resolve.
  Fix: Always use the aliased form `satisfy requirement <uniqueAlias> : 'REQ-001';` with a distinct alias per statement (e.g. `uc001Req001`, `uc004ReqNf001`). Hyphenated names must stay quoted on the right-hand side of the colon; the alias must be a plain unquoted ASCII identifier. 9 use case defs (UC-001..UC-009) + 5 actor part defs + 1 subject part def + 23 requirement defs validated with 0 syntax and 0 semantic errors.
- Note: `include use case 'UC-005 Ladungsträger puffern und sequenzieren';` (no alias, no `: Type`) resolves cleanly — confirms the 2026-07-07 / 2026-06-20 entries.
- Note: German umlauts (ä, ö, ü) inside SINGLE-QUOTED element names (e.g. `'UC-002 Behälter einlagern'`, `'Fördertechnik'`) are accepted by this validator, unlike the bare-identifier case recorded in general/LEARNED.md 2025-04-18. Only doc comments and quoted names were used here.

## Session 2026-09-21
- No new fixes.

## Session 2026-09-16
- No new fixes.

## Session 2026-09-09
- No new fixes.

## Session 2026-09-08
- No new fixes.

## Session 2026-01-15
- No new fixes.

## Session 2026-09-04
- No new fixes.

## Session 2026-01-15
- No new fixes.

## Session 2026-08-29
- No new fixes.

## Session 2026-08-26
- Error: `Unexpected '-', expecting one of: ;,` (10 errors) — hyphenated requirement names in `requirement def REQ-001 { ... }` declarations are parsed as subtraction, not just in `satisfy` statements.
  Fix: Quote hyphenated requirement names in the declaration too: `requirement def 'REQ-001' { ... }`. Final file: 10 use case defs (UC-001..UC-010) with unique subject names (uc001Subject..uc010Subject), 6 actor/subject part defs (UtilitySCADA, GridOperator, BatteryManagementSystem, ProtectionSystem, EventRecorder, BatteryEnergyStorageSystem), 10 requirement defs for satisfy tracing ('REQ-001'..'REQ-010'), include use case relationships (UC-005/006/007 include UC-010), objective doc blocks, all validated with 0 syntax errors and 0 semantic errors.

## Session 2026-08-26
- Error: `Couldn't resolve reference to Feature ''REQ_001''` (13 errors) — quoting requirement names in `satisfy 'REQ_001';` produced unresolved-reference errors because the requirement defs were declared as identifiers (REQ_001), not escaped names.
  Fix: Use unquoted `satisfy REQ_001;` when the requirement def name is a plain identifier. Quoting is only needed when the requirement name contains hyphens or spaces (e.g. `REQ-001`). Final file: 10 use case defs (UC-001..UC-010) with unique subject names (uc001Subject..uc010Subject), 5 actor part defs (GridSCADASystem, GridOperatorDispatch, BMSOperator, MaintenanceTechnician, ProtectionSystem), 1 system subject part def (BatteryEnergyStorageSystem), 10 requirement defs for satisfy tracing (REQ_001..REQ_010), objective doc blocks, all validated with 0 syntax errors and 0 semantic errors.

## Session 2026-08-25
- No new fixes.

## Session 2026-07-17
- No new fixes.

## Session 2026-07-16
- No new fixes.

## Session 2026-07-15
- No new fixes.

## Session 2026-07-08
- No new fixes.

## Session 2026-07-07
- No new fixes.

## Session 2026-07-06
- No new fixes.

## Session 2026-07-06
- Error: `Invalid syntax near 'use' — no valid parsing rule matches` and `Unexpected 'if', expecting one of: ;,` — `extend use case <name> : '<UseCaseDef>' if <condition>;` syntax is not supported by this validator. The `extend` keyword with `if` guard causes parse errors.
  Fix: Removed all `extend use case` statements. The sequencing relationships (UC-002 → UC-003 → UC-004) are documented in the use case descriptions but cannot be expressed as `extend` in this validator. Use only `include use case '<UseCaseDefName>';` (without `: Type` suffix) for inclusion relationships.
- Error: `Couldn't resolve reference to Feature ''Record Inventory Movement''` — using `include use case recordMovement : 'Record Inventory Movement';` (with named alias and `: Type`) caused unresolved reference warnings.
  Fix: Changed to `include use case 'Record Inventory Movement';` — the simpler syntax without a named alias and without `: Type` resolves correctly.

## Session 2026-07-14
- No new fixes.

## Session 2026-07-04
- No new fixes.

## Session 2026-07-12
- No new fixes.

## Session 2026-07-11
- Error: `Couldn't resolve reference to Namespace 'Actors'.` and `Couldn't resolve reference to Namespace 'Requirements'.` — the imported packages `Actors::*` and `Requirements::*` do not exist in the model scope, causing 55 unresolved reference errors for all types and requirements.
  Fix: Removed the non-existent imports (`public import Actors::*;` and `public import Requirements::*;`). Defined all actor/subject types as inline `part def` declarations (LagerSystem, WarehouseManagementSystem, GoodsReceiptOperator, PickingOperator, ConveyorSystem, SystemAdministrator) and all referenced requirements as inline `requirement def` declarations (REQ_003 through REQ_018) within the same package. The `satisfy requirement <alias> : <RequirementDef>;` syntax was already correct in the original file.

## Session 2026-07-11
- Error: `Couldn't resolve reference to Namespace 'Actors'` and `Couldn't resolve reference to Namespace 'Requirements'` — the file imported `Actors::*` and `Requirements::*` but those packages don't exist in scope.
  Fix: Removed the two non-existent imports and defined all actor types as `part def` declarations and all requirement types as `requirement def` declarations locally within the same package.
- Error: `Must reference a requirement` and `Couldn't resolve reference to Feature 'REQ_006'` — using `satisfy REQ_006;` (bare reference) inside use case def causes semantic errors because the validator expects `satisfy requirement <alias> : <RequirementDef>;` syntax.
  Fix: Changed `satisfy REQ_006;` to `satisfy requirement req006 : REQ_006;` with a unique alias per satisfy statement and the requirement def declared in the same package scope.

## Session 2026-07-10
- No new fixes.

## Session 2026-06-28
- Error: `Unexpected 'GoodsReceiptOperator', expecting one of: default, :=, ;...` — `associate <partDef> to '<UseCaseName>';` is not valid SysML v2 syntax. The `associate` keyword is not recognized by the validator for actor-to-use-case relationships.
  Fix: Removed all `associate` statements. Actor-to-use-case relationships are already expressed via the `actor <name> : <PartDef>;` declarations inside each `use case def` block. No separate package-level association syntax is needed.

## Session 2026-06-27
- Error: `Extra token 'actor' found, expecting one of: abstract, action, alias...` — `actor def` is not valid SysML v2 syntax. The validator does not recognize `actor` as a keyword for type definitions.
  Fix: Replaced all `actor def <Name>;` with `part def <Name>;` — actors in SysML v2 are declared as `part def` types, not with a dedicated `actor def` keyword. The `actor` keyword is only used inside `use case def` blocks to declare actor parameters.

## Session 2026-06-26
- No new fixes.

## Session 2026-06-25
- No new fixes.

## Session 2026-06-24
- Error: `Must reference a requirement` and `Couldn't resolve reference to Feature ''REQ-XXX''` — using `satisfy 'REQ-XXX';` inside use case def causes semantic errors because the validator expects `satisfy requirement <alias> : '<RequirementDef>';` syntax.
  Fix: Changed `satisfy 'REQ-XXX';` to `satisfy requirement <alias> : 'REQ-XXX';` with a unique alias per satisfy statement. The requirement def must be declared in the same package scope.

## Session 2026-06-22
- No new fixes.

## Session 2026-06-21
- No new fixes.

## Session 2026-06-20
- No new fixes.

## Session 2026-06-20
- Error: `Duplicate definition 'warehouseSystem': already defined as part_usage at line 36` — subject and actor parameter names that are the same across multiple use case defs cause duplicate definition errors. The validator treats subject/actor names as unique across the entire package scope, not scoped per use case def.
  Fix: Use unique subject/actor names per use case (e.g., `receiveSys`, `pickContainerSys`, `storeSys`, `retrieveSys`, `transportSys`, `resequenceSys`, `scheduleSys`, `configSys`, `maintenanceSys`, `bufferProdSys`, `bufferShipSys` and per-use-case actor suffixes like `conveyorSysUC1`, `wmsUC4`, `storageDeviceUC2`, etc.).

## Session 2026-06-20
- Error: `Couldn't resolve reference to Feature ''<UseCaseName>''` — `include use case <alias> : '<UseCaseName>';` with an alias and string name causes unresolved reference warnings (25 warnings). The alias syntax is not supported by this validator for `include use case`.
  Fix: Use `include use case '<UseCaseName>';` without the alias and colon. The use case name as a quoted string directly references the use case def. This matches the syntax reference: `include use case <includedUseCase>;`.

## Session 2026-06-18
- Error: `Unrecognized character or sequence: 'ä'` — umlauts in part def name `Regalbediengerät` cause syntax errors.
  Fix: Replaced `ä` with `ae` (ASCII-only identifiers) as documented in `/skills/general/LEARNED.md`. Validator does not support Unicode characters in identifiers.
- Error: `Couldn't resolve reference to Feature 'REQ_015'` — requirement references using underscores (`REQ_015`) did not match requirement def names defined with hyphens (`<'REQ-001'>`) in the RequirementDiagram package.
  Fix: Used quoted names with hyphens: `satisfy 'REQ-015';` and declared `requirement def 'REQ-001';` stubs in the same package for the validator to resolve. The `public import RequirementDiagram::*;` is kept for documentation but the stubs are what the validator resolves against.

## Session 2026-06-20
- Error: `Invalid syntax near 'use' — no valid parsing rule matches` for `extend use case performMaintenance : 'Perform Maintenance on Automated Equipment' if equipmentFailure;`
  Fix: Replaced `extend use case` with `include use case` (removed the `if` guard clause) since `extend use case` with an `if` guard causes syntax errors in the validator. The `include use case` pattern `include use case <name> : '<UseCaseDef>';` validates cleanly.

## Session 2026-06-19
- No new fixes.

## Session 2026-06-17
- No new fixes.

## Session 2026-06-16
- No new fixes.

## Session 2025-04-18
- No new fixes.

