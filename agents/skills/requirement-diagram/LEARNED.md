# Learned Fixes — SysML v2 Requirement Diagram

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
- Error: `Syntax error at line 20, column 44: Unexpected '::', expecting one of: by, default, :=...` at `satisfy requirement GruppeFunktional::KapazitaetReq;` — a `::`-qualified path is not a legal satisfy target, so a requirement usage nested inside a group requirement cannot be reached by qualified name.
  Fix: Give each nested usage a plain unique name (`requirement <'REQ-003'> req003Kapazitaet : KapazitaetBehaelterstellplaetze { text = "..."; }`) and reference it unqualified from a part usage: `part blk001Lager : Lager { satisfy requirement req003Kapazitaet; }`. This validated clean on the `sidecar` backend with 26 requirement usages, 2 group requirements, 17 part defs/usages and 30 satisfy statements — the working formal traceability form for a requirement diagram (supersedes the "document satisfaction in comments only" entries above).
- Note: `satisfy requirement <usageName>` resolves inside a `part <usage> : <Type> { ... }` body (nested inside the package), not only inside `part def`; the usage name must be declared earlier in the same package.

## Session 2026-11-06
- Note: The schema-mandated shape `requirement <'REQ-NF-001'> { text = "..."; }` (44 usages, 9 nested group requirements each also carrying `text = "..."`, containment = category -> requirement) validated clean on the `sidecar` backend with zero syntax and zero semantic errors — no duplicate-definition error for either `text` or the quoted IDs. Reinforces the 2026-10-27 entry: nesting under a named group owner is the safe pattern.
- Note: Parenthesised group names (e.g. `requirement LagerfunktionUndLeistungsumfang {`) need no `def`/usage type; a bare `requirement <name> { ... }` at package level is legal and is the reusable container for nested requirement usages.
- Note: German requirement prose is fine inside `text = "..."` (umlauts, `ß`, `ä/ö/ü`, `()`/`[]`, `/`, `->`), but German typographic quotes, `&`, `&&`, `%` and `°` must be avoided even in strings; write `und`/`-&gt;`-free ASCII arrows and keep only unquoted ASCII-safe symbols to stay lexer-clean.

## Session 2026-11-05
- No new fixes.

## Session 2026-10-28
- No new fixes.

## Session 2026-10-27
- Error: none — the schema-mandated `requirement <'ID'> { text = "..."; }` form validated clean on the `sidecar` backend with **34** requirement usages in one package (4 group requirements, each carrying its own `text`, plus 30 nested leaf requirements), i.e. zero `Duplicate definition 'text': already defined as attribute_usage` errors. This contradicts the 2026-10-08 entry, which attributed such duplicates to sibling `text` declarations.
  Fix: No change was needed. The duplicate-`text` errors of 2026-10-08 presumably arose from a different validator state or from sibling usages at the *same* scope level without an enclosing owner; nesting every leaf requirement inside its owning group requirement validated clean. Prefer `text = "..."` when the schema mandates it, and fall back to `doc /* ... */` only if the validator actually reports the duplicate.
- Note: `?` and `.docx`/`.xlsx` file names, commas and parentheses inside a `text = "..."` string are all accepted; only the *identifiers* must stay ASCII and reserved-keyword-free. Verbatim source quotes (including German original snippets) were therefore kept in the text strings with umlauts transliterated (ae/oe/ue/ss) inside names only.

## Session 2026-10-08
- Error: `Duplicate definition 'text': already defined as attribute_usage` ×18 — 18 sibling `requirement <'ID'> { text = "..."; }` usages in one package each declare the inherited `text` attribute, so the validator reports 2(N−1) duplicate-definition errors.
  Fix: Carried the verbatim requirement text in `doc /* ... */` instead of `text = "..."` for every requirement usage. Validated clean with 18 requirement usages (REQ-001..REQ-010, REQ-NF-001..REQ-NF-004, C1..C5) in one package.

## Session 2026-10-08
- Error: `Unexpected ':', expecting one of: ;` / `Extra token '<' found` at `requirement receivingZoneReq : <'REQ-001'> ReceivingZoneAcceptance;` — the angle-bracket identifier form is only legal on the *name* slot of a declaration, not on the type slot after `:` in a requirement usage.
  Fix: Reference the requirement definition by its bare name in the usage type slot: `requirement receivingZoneReq : ReceivingZoneAcceptance;`. Same rule as the 2026-10-02 `specializes` entry — quoted `<...>` identifiers belong on the declaration's own name, never on a referenced type.

## Session 2026-10-08
- No new fixes.

## Session 2026-10-07
- No new fixes.

## Session 2026-10-03
- Error: `Couldn't resolve reference to Element 'writeLegible'` / `Couldn't resolve reference to Element 'hasInkReservoir'` / `Couldn't resolve reference to Element 'nibProtected'` / `Couldn't resolve reference to Element 'refillable'` / `Couldn't resolve reference to Element 'comfortableDuration'` — require constraints referenced attributes not declared on the subject part def.
  Fix: Added the missing attributes (`writeLegible`, `hasInkReservoir`, `nibProtected`, `refillable`, `comfortableDuration`) to the `Pen` part def before the requirement defs that reference them. Always cross-check every attribute in a require constraint against the subject part def's declared attributes.

## Session 2026-10-02
- Error: `Extra token '<' found, expecting one of: behavior, crosses, feature...` / `Extra token '>' found, expecting one of: ;,` at `requirement def <'REQ-002'> B specializes <'REQ-001'> {` — the quoted angle-bracket identifier form is only legal on the *name* slot of the declaration, not on the supertype slot after `specializes`.
  Fix: Reference the supertype by its bare (unquoted) name: `requirement def <'REQ-002'> B specializes A {` where `A` is the plain name of the parent requirement def. Same applies to `:>` — `requirement def <'REQ-003'> D :> A {` validates clean.
- Error: none — `requirement <'REQ-001'> { text = "..."; }` nested inside another `requirement <'REQ-001'> { ... }` body validates with zero syntax and zero semantic errors, and 39 uniquely-identified nested requirement usages in one package produce no duplicate-definition errors.
  Fix: For requirement diagrams the schema-mandated `requirement <'ID'> { text = "..."; }` form is the safest carrier of the ID + verbatim text; use nesting to express the containment/derivation hierarchy the source supports (parent requirement = derived-from root, children nested in its body). No `subject`/`require constraint` needed when the source text is purely natural-language.

## Session 2026-09-22
- Error: `Extra token 'connector' found, expecting one of: behavior, crosses, default...` at `subject connector : ConnectorInterface;` — `connector` is a reserved keyword in SysML v2 and cannot be used as a subject (or any feature) name.
  Fix: Renamed the subject feature to `connectorIface` (and all its `require constraint` references) wherever a subject was typed as `ConnectorInterface`. Reserved-keyword subject names must be avoided; use a non-reserved alias like `connectorIface`.

## Session 2026-09-22
- No new fixes.

## Session 2026-09-21
- No new fixes.

## Session 2026-09-16
- No new fixes.

## Session 2026-09-09
- No new fixes.

## Session 2026-09-09
- No new fixes.

## Session 2026-09-09
- Error: `ERROR: Duplicate definition 'text': already defined as attribute_usage` (same for `classification`, `verifyMethod`, `gap`) — N requirement usages each declaring same-named attributes yield 2(N−1) duplicate-definition semantic errors; a single requirement with `text=` + attributes is valid, which masks the pattern.
  Fix: carry verbatim requirement text in `doc /* ... */` instead of `text = "..."`, and make every auxiliary attribute name unique per requirement by suffixing the ID (`classificationReq001`, `verifyMethodReq008`, `gapReqNF003`); validated clean with 18 requirement usages in one package.
- Error: `ERROR: Duplicate definition ''REQ-005'': already defined as requirement_usage` — declaring the same requirement ID both top-level AND nested inside another requirement duplicates the definition.
  Fix: keep all requirement usages top-level; document containment/process-flow relationships (REQ-005/REQ-006 feeding REQ-007) in a comment instead of nesting duplicate IDs.
- Error: `Invalid syntax near 'verification'` — `verification` is not a legal attribute name inside a requirement usage.
  Fix: renamed the attribute to `verifyMethod` (made unique per requirement as above).

## Session 2026-09-04
- No new fixes.

## Session 2026-08-29
- No new fixes.

## Session 2026-08-26
- No new fixes.

## Session 2026-08-26
- Error: `Extra token '%' found, expecting one of: all, behavior, crosses...` at `attribute batteryCharge : Real = 100.0 [%];` — the `%` symbol is not a recognized SI unit in the validator.
  Fix: Use dimensionless `Real` for percentage-valued attributes (battery charge, reserve threshold, detection reliability) without `[%]` unit brackets. Only SI unit symbols (m, kg, s, Hz, min, ms, m/s) are accepted.

## Session 2026-08-26
- Error: `require constraint { system.accessBasedAccessControl == true }` referenced an attribute not declared on the subject part def (typo: `accessBasedAccessControl` vs declared `roleBasedAccessControl`).
  Fix: Corrected the require-constraint attribute reference to `system.roleBasedAccessControl == true` to match the declared attribute on the WarehouseSystem part def. Always cross-check every attribute referenced in a require constraint against the subject part def's declared attributes.

## Session 2026-08-25
- No new fixes.

## Session 2026-07-17
- Error: `Couldn't resolve reference to Element 'inboundPalletDimensionsMm'.` — the `require constraint` referenced an attribute that was not declared on the subject part def.
  Fix: Added the missing attribute `inboundPalletDimensionsMm : Real = 1200.0;` to the `WarehouseSystem` part def before the requirement defs that reference it.

## Session 2026-07-10
- No new fixes.

## Session 2026-07-09
- Error: `Unexpected 'REQ_004', expecting one of: default, :=, ;...` at `contain REQ_004 to REQ_001;` — `contain` and `trace` are not valid keywords at package level in SysML v2 textual notation.
  Fix: Remove `contain` and `trace` statements entirely. Document containment hierarchy and trace relationships using comments (`// Contain: REQ-004 contains REQ-001`). SysML v2 does not support standalone `contain` or `trace` keywords at package scope.

## Session 2026-07-08
- Error: `Unexpected 'require', expecting one of:` at `require constraint { ... }` inside `constraint def` — `require constraint` is only valid inside `requirement def`, not `constraint def`.
  Fix: Use `in <param> : <type>;` parameter declaration followed by a bare boolean expression (no `require constraint` wrapper) inside `constraint def`. E.g., `constraint def C1 { in system : WarehouseSystem; system.floorArea <= 5000.0 }`.

## Session 2026-07-07
- No new fixes.

## Session 2026-07-06
- No new fixes.

## Session 2026-07-05
- Error: `Unexpected 'C1', expecting one of: default, :=, ;...` at `refine C1 by REQ-NF-004;` — `refine` is not a valid keyword in SysML v2 textual notation for package-level relationships.
  Fix: Remove `refine` statements entirely. Document the refinement relationship between constraints and requirements using comments (`// C1 and C2 constrain the spatial envelope that enables hazmat segregation per REQ-NF-004`). SysML v2 does not support a standalone `refine` keyword at package scope.

## Session 2026-07-05
- Error: `Couldn't resolve reference to Element 'degC'.` — `degC` is not a recognized unit symbol in the SI library imported via `public import SI::*;`.
  Fix: Use Kelvin `[K]` instead of `[degC]` for temperature values in require constraints. Convert Celsius values to Kelvin (e.g., 45 degC → 318.15 K, 0 degC → 273.15 K, 40 degC → 313.15 K). This applies across all diagram types when declaring temperature constraints with SI units.

## Session 2026-07-04
- No new fixes.

## Session 2026-07-02
- No new fixes.

## Session 2026-06-27
- No new fixes.

## Session 2026-06-26
- No new fixes.

## Session 2026-06-25
- No new fixes.

## Session 2026-06-26
- Error: 80× `Couldn't resolve reference to Element '<attributeName>'` — empty `part def` stubs had no attribute declarations, so `require constraint { part.attributeName ... }` could not resolve the feature reference.
  Fix: Added all referenced attributes (with correct types: Real, Integer, Boolean) to each `part def` body. Every attribute name used in a `require constraint` predicate must be declared on the subject's part def. The full set of attributes was collected by scanning all `require constraint` expressions across the 57 requirement defs.

## Session 2026-06-26
- Error: `Unexpected '::', expecting one of: ;` at `package UAVDeliverySystem::Requirements` — the `::` separator in package names is not valid in SysML v2 textual notation.
  Fix: Use underscores instead of `::` in package names: `package UAVDeliverySystem_Requirements` instead of `package UAVDeliverySystem::Requirements`. Package names must be simple identifiers without namespace separators.

## Session 2026-06-24
- No new fixes.

## Session 2026-06-24
- Error: `Invalid syntax near '-' — no valid parsing rule matches` at `requirement def <FR-001>` — the hyphen in `FR-001` inside angle brackets without quotes causes the parser to interpret `-` as an operator.
  Fix: Use single-quoted angle-bracket identifiers: `requirement def <'FR-001'>` instead of `requirement def <FR-001>`. The quotes tell the parser the hyphen is part of the identifier, not a subtraction operator. This applies to all requirement IDs containing hyphens (FR-*, NFR-*, REQ-*, etc.).

## Session 2026-06-22
- No new fixes.

## Session 2026-09-08
- Error: `Unrecognized character or sequence: '²'` at `attribute maxFloorArea : Real = 2000.0 [m²];` and `attribute minHazmatArea : Real = 200.0 [m²];` — Unicode superscript characters (², ³, etc.) are not recognized in SI unit brackets by the validator.
  Fix: Use ASCII-safe unit notation: `[m2]` instead of `[m²]`, `[m3]` instead of `[m³]`. This applies to any attribute default using SI units with superscript characters in require/assume constraints or attribute declarations.

## Session 2026-06-20
- No new fixes.

## Session 2026-06-20
- Error: `Duplicate definition 'GoodsReceiving': already defined as part_def at line 41 (95:4)` — nested package names collided with `part def` names declared at the same package scope.
  Fix: Renamed the nested packages to avoid name collisions (e.g., `package GoodsReceivingReqs` instead of `package GoodsReceiving`). Package names and part def names share the same namespace and must be unique within a package.

## Session 2026-06-18
- No new fixes.

## Session 2026-06-18
- Error: `Extra token 'loop' found, expecting one of: behavior, crosses, default... (and 33 more)` — `loop` is a reserved keyword in SysML v2 and cannot be used as a subject identifier.
  Fix: Renamed the subject from `loop` to `conveyorLoop` in requirement def REQ-003. The subject name must avoid all reserved keywords (e.g., `loop`, `message`, `transition`, `variant`, etc.).

## Session 2026-06-17
- No new fixes.

## Session 2026-06-16
- No new fixes.

## Session 2025-04-18
- Error: `Unrecognized character or sequence: 'ä'` — Unicode characters (ä, ö, ü, ß) are not supported in SysML v2 identifiers.
  Fix: Replace all umlauts with ASCII equivalents (ae, oe, ue, ss) in identifiers (part def names, subject names, requirement def names). The doc block comments can still contain umlauts since they are string content.
- Error: `Extra token 'lterlager' found` after `Behälterlager` — the parser splits on umlaut characters.
  Fix: Use ASCII-only identifiers: `Behaelterlager` instead of `Behälterlager`.

## Session 2025-04-17
- No new fixes.

## Session 2025-04-16
- No new fixes.

## Session 2025-04-15
- No new fixes.

## Session 2025-04-14
- No new fixes.

## Session 2025-01-30
- No new fixes.
