# Learned Fixes — SysML v2 Block Definition Diagram

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
- Note: Package-level `connection def` with multiplicities on BOTH `end`s (`end a : 'BLK-110 Lagergeraet Mini-RBG'[0..2]; end b : 'BLK-300 Lagerbehaelter Tablar'[1..*];`) validates cleanly on `sidecar` for all multiplicity shapes used here ([1], [0..2], [1..2], [1..*]). Quoted part-def names in `end` types resolve when the part defs are declared later in the same package, so the "declare associations first" ordering is a readability choice, not a resolution requirement — but keep it, it matches the earlier note.
- Note: An excluded scope boundary declared as `abstract part def 'BND-001 ...'` with `attribute scopeAusgeschlossen : Boolean = true;` and the unfulfilled requirement values as plain attributes validates cleanly; omitting the corresponding `part` usage in the root def is how multiplicity 0 / "boundary only" is expressed (no `[0]` part usage needed).
- Note: Unsourced values ("nicht belegt") can be kept type-safe as placeholder defaults (`attribute pufferkapazitaet : Integer = 0;`) with a `//` comment naming the gap ID and stating the value is an unset placeholder — the validator accepts it and the gap is not silently invented.

## Session 2026-11-06
- Note: A 28-`part def` / 149-port hierarchy built entirely from quoted ASCII-transliterated names (`part def 'LB-2.1 Mehrebenenshuttle'`) validates cleanly on the sidecar backend: generalization to a quoted abstract type (`part def 'LB-2.1.3 Teleskoptisch' :> 'LB-G Lastaufnahmemittel'`), composition to quoted types with derived multiplicities (`part fach : 'LB-1.2 Fach - Behaelterstellplatz'[10560];`, `[0..10560]`), `inout item` inside a `port def`, and one `port def` per (direction, payload) pair so no port usage carries direction. Reserved-keyword collisions to watch when transliterating German block names into part/attribute usage names: `loop` (`p_ls_vom_loop_in` is fine, `part loop : ...` is not).
- Note: Non-ASCII `ß` in an *attribute* name (`oberes_anfahrmaß`) is a hard lexer error (`Unrecognized character or sequence: 'ß'`) — same class as the 2025-04-18 umlaut entry, which only mentioned port def names; it applies to every identifier.

## Session 2026-11-05
- Error: `ERROR: Couldn't resolve reference to Element 'mm'.` (and similarly for `h`, `kg`) on unit-annotated attributes like `[mm]`, `[h]`, `[kg]`, `[1/h]`.
  Fix: Added `public import SI::*;` alongside `public import ScalarValues::*;` so SI unit symbols resolve in the validator.

## Session 2026-10-27
- Error: `ERROR: Cannot override a binding feature value (37:95)` — redefining an inherited attribute that already carries a default with `attribute x :>> Base::x = 1.0 [m/s];` is rejected; the base default is a binding that cannot be overridden.
  Fix: Do not use `:>>` for attribute values that the supertype already binds. Either declare a distinctly named local attribute (e.g. `performanceDoubleCyclesMin`/`Max` on both the generic and the specialised block) or simply omit the redefinition and rely on inheritance. Re-declaring a port that has no default is fine: `port cmdIn : CommandPort;` inside `part def OneLevelShuttle :> StorageRetrievalDevice`.
- Note: `abstract part def <Name>` (for the abstract blocks an input BDD declares) validates cleanly at package level; specialisations use the normal `part def <Sub> :> <Super>`. Port direction (`in port p : P;` / `out port p : P;`) on a port usage validates cleanly — direction may be written on the usage as well as on the `item` inside the `port def`.
- Note: `ref part supportedLoadCarriers : LoadCarrier[1..*];` is the accepted shape for the reference-typed / many-valued properties of an input BDD (e.g. "supportedLoadCarriers: LoadCarrierType[1..*]", "mountedOn: Reference RackStructure", "assignedLevel: Reference RackLevel", "registeredEntities: Reference ..."). No default value needed.

## Session 2026-10-08
- Error: `Extra token '%' found` at `attribute pct : Real = 99.0 [%];` — the `[%]` unit symbol is not parseable by the validator (same class as the `[m2]`/`[m3]` unit-symbol issue).
  Fix: Drop the unit bracket for percentage attributes — use a bare `Real = 99.0;` (value already carries the percent semantics in the attribute name). Applies to any diagram with `%` attributes (availability, duty cycle, humidity).
- Note: `[degC]`, `[kW*h]`, `[kW]`, `[W]`, `[Hz]`, `[A]`, `[V]`, `[s]`, `[mm]`, `[m]`, `[kg]` all resolve cleanly — no conversion to base units needed. `[mA]` still does NOT resolve (use `[A]`, e.g. `0.006 [A]` for a 6 mA DC trip threshold).

## Session 2026-10-07
- Note: `String[0..*]` collection attributes with a scalar base type (e.g. `attribute constellations : String[0..*];`) validate cleanly — extends the earlier `T[0..*]` note, which only covered attribute-def-typed lists.
- Note: Compound/derived unit symbols `[kW*h]`, `[W*h/kg]`, `[mm/h]`, `[min]` all resolve cleanly in the validator (no conversion to base units needed).
- Note: Inline `constraint name { expr }` bodies with unit-annotated comparisons (e.g. `breach_response_s <= 1.0 [s]`, `takeoff_mass_kg <= 25.0 [kg]`) validate cleanly.
- Note: `ref part operator : DeliveryOperator[1];` for an external actor typed by a `part def` validates cleanly and needs no default value — same shape as the item-def `ref part` fix but for occurrence-def actors.

## Session 2026-10-07
- No new fixes.

## Session 2026-10-03
- No new fixes.

## Session 2026-10-03
- Error: `ERROR: An attribute must be typed by attribute definitions.` — an `attribute` typed by an `item def` (`attribute releasedSequence : RetrievalOrder[0..*];`, `attribute carrierFormat : LoadCarrier;`) is rejected, even though `item def` types are fine inside `port def` bodies.
  Fix: Declare item-typed features as `ref part <name> : <ItemDef>[mult];` instead of `attribute`. `ref part` accepts `item def` types and needs no default value; `attribute` only accepts `attribute def` / scalar types.
- Error: `Syntax error ... Unexpected 'movements', expecting one of: ]` and `Unexpected '/'` — unit strings containing spaces or slashes (`[bin movements/h]`, `[Doppelspiele/h]`) are not parseable unit expressions.
  Fix: Use a bare `[1/h]` for every rate/count-per-time attribute and move the domain wording into the attribute name (e.g. `performancePerRobot : Real = 80.0 [1/h];`). Same class: `[orders/h]` → `[1/h]`.
- Error: `Syntax error ... Unexpected 'of', expecting one of: default, :=, ;` — collection syntax `attribute x : List of T;` and `attribute x : Map (A, B);` is not supported, nor is an inline enumeration `attribute x : Enumeration { a, b };` or an inline record literal `attribute x : {Length_mm = 52000, ...};`.
  Fix: Use `T[0..*]` for lists (typed by an `attribute def`), split record literals into one scalar attribute per field, and declare every enumeration as a package-level `enum def` referenced by name. Map-typed properties were reduced to a `String` description of the mapping.
- Note: `as` is a reserved keyword — `attribute as : AvailabilityState;` fails with `Invalid syntax near 'as'`; rename to `availabilityState`. Same watch list as `state`, `action`, `item`, `message`, `standard`, `variant`, `frame`.

## Session 2026-10-03
- Error: `A port must be typed by port definitions.` — declaring a port directly with an `item def` (or an `enum def`) as its type, e.g. `port storageAccess : LoadCarrier;` or `port stateOut : DeviceState;`, is rejected by the validator. The skill's Example 3 (`port inboundDock : MaterialFlowPort;`) is the only accepted shape.
  Fix: Declare one `port def` per interface direction/payload — `port def CarrierInOutPort { inout item carrier : LoadCarrier; }` — and type every port usage with that port def (`port storageAccess : CarrierInOutPort;`). Direction belongs on the `item` inside the `port def`, never on the port usage. This applies to every BDD that lists ports in its input artifact.
- Error: `An occurrence, item or part must be typed by occurrence definitions.` — using an `enum def` as the payload type of an `item` inside a `port def` (e.g. `out item state : OccupancyState;`).
  Fix: Wrap the enumeration in an `item def` (e.g. `item def OccupancyReport { attribute occupancyState : OccupancyState = OccupancyState::free; }`), or better, type the port with a `port def` whose item is that wrapper item def.

## Session 2026-10-02
- No new fixes.

## Session 2026-09-22
- No new fixes.

## Session 2026-09-22
- Note: `[degC]` and `[ohm]` unit symbols resolve cleanly in the validator (zero errors) — no need to convert Celsius to Kelvin or use `[Ω]`. Prior entries (2026-07-05) recommending `[K]` conversion may be stale for this validator version. Kept `[degC]` for temperature and `[ohm]` for resistance attributes.
- Note: AC charging-point BDD with 11 top-level parts (7 required components + 4 distinct certified protection/safety parts), nested sub-parts (AuthorisationManager, SessionManager, FaultCoordinator, LoadManagement, PwmGenerator, PilotStateMachine, ConnectorLock, ProximityPilotSensor, RcdTypeA, RdcDd), and Real-typed attributes with SI units validates with zero errors on first attempt.

## Session 2026-09-21
- Error: `Couldn't resolve reference to Element 'm2'.` / `Couldn't resolve reference to Element 'm3'.` — area/volume unit symbols written as `[m2]` and `[m3]` are not recognised by the validator's SI library.
  Fix: Write area as `[m*m]` and volume as `[m*m*m]` (e.g. `attribute baufenster_flaeche : Real = 1750.0 [m*m];`). This supersedes the earlier `[m2]`/`[m3]` advice in `/skills/general/LEARNED.md` — the ASCII-digit form is also unresolvable.
- Note: `[1/h]`, `[1/d]`, `[m/s]`, `[mm]`, `[m]`, `[kg]`, `[h]` all resolve cleanly; no conversion needed for rate/duration/speed attributes.
- Note: Block names containing spaces, hyphens, umlauts or parentheses (e.g. `LB-210 Behaelterregal (OLS)`) must be (a) transliterated to ASCII and (b) wrapped in single quotes: `part def 'LB-210 Behaelterregal OLS' { ... }`. A 41-block hierarchy with quoted names, nested `part <name> : 'Type'[1];` composition and inline `constraint name { expr }` bodies validates with zero errors.

## Session 2026-09-16
- No new fixes.

## Session 2026-09-09
- Error: `Syntax error at line 179, column 16: Unexpected '-', expecting one of: ;` — a `part def` name containing a hyphen and space (`HRS-700 Station`) is not parsed as a valid identifier.
  Fix: Wrap the identifier in single quotes: `part def 'HRS-700 Station' { ... }`. Any element name (part def, part usage, attribute) containing spaces, hyphens, or other non-identifier characters must be quoted.

## Session 2026-09-09
- No new fixes.

## Session 2026-09-08
- No new fixes.

## Session 2026-09-04
- No new fixes.

## Session 2026-08-30
- Error: `Invalid syntax near 'action' — no valid parsing rule matches` — `action` is a reserved keyword in SysML v2 and cannot be used as an attribute name.
  Fix: Renamed the attribute from `action` to `requestAction` (e.g., `attribute requestAction : String = "unknown";`). Add `action` to the reserved keyword watch list alongside `state`, `message`, `standard`, `variant`, `transition`, `analyze`, `detect`, `limit`, `activate`, `feedback`, `done`, `fork`, `join`, `decide`, `start`, `notify`, `signal`, `wait`, `receive`, `create`, `eval`, `check`, `restrict`, `continue`.
- Pattern: Semantic warnings about "Couldn't resolve reference to Feature 'XXX'" when sub-block part types are referenced but not declared as `part def`.
  Fix: Ensure every type referenced in a `part <name> : Type[mult]` declaration has a corresponding `part def Type { ... }` in the same package. All 29 sub-block types (DockManager, GoodsReceiptRecorder, ASRSController, RackStructure, etc.) must be declared before they are referenced.

## Session 2026-08-29
- No new fixes.

## Session 2026-08-26
- No new fixes.

## Session 2026-08-26
- Error: `Couldn't resolve reference to Feature 'acceptProgram'.` (49 occurrences) — `perform action <name>;` and `perform action <name> : <Type>;` both fail semantic resolution in the validator.
  Fix: Use `action <name>;` for untyped operations and `action <name> : <ActionDef>;` for parameterised operations (with a matching `action def <Name> { in ...; }` declared at package level). The `perform action` keyword form is not accepted by this validator for operation declarations inside part defs.

## Session 2026-08-26
- Error: `Invalid syntax near 'item' — no valid parsing rule matches` — `item` is a reserved keyword in SysML v2 and cannot be used as an attribute name.
  Fix: Renamed the attribute from `item` to `itemId` (e.g., `attribute itemId : String = "ITEM-0000";`). This is the same class of reserved-keyword issue as `message`, `standard`, `state`, `action`, `variant` documented in prior sessions.

## Session 2026-08-25
- No new fixes.

## Session 2026-08-25
- Error: `Unexpected ',', expecting one of: default, :=, ;...` at enum literal — comma-separated enum literals (`hazardous, nonHazardous`) are rejected by the validator.
  Fix: Use `enum def Name { enum lit1; enum lit2; }` with the `enum` keyword before each literal and semicolons, not commas.

## Session 2026-07-17
- Error: `An occurrence, item or part must be typed by occurrence definitions.` — using `String` directly as the item type inside a port def (e.g., `in item dataIn : String;`) causes a semantic error because `String` is not an occurrence definition.
  Fix: Create an `item def DataSignal { attribute content : String = "unknown"; }` and use `in item dataIn : DataSignal;` inside the port def instead of referencing `String` directly.

## Session 2026-07-09
- No new fixes.

## Session 2026-07-08
- No new fixes.

## Session 2026-07-07
- No new fixes.

## Session 2026-07-06
- No new fixes.

## Session 2026-07-05
- No new fixes.

## Session 2026-07-07
- Error: `Couldn't resolve reference to Element 'mA'.` — `mA` (milliampere) and `ms` (millisecond) are not recognised SI unit symbols in the validator's library.
  Fix: Converted `3000.0 [mA]` to `3.0 [A]` and `10.0 [ms]` to `0.01 [s]` using base SI units (A for amperes, s for seconds). The validator only supports base SI unit symbols (m, kg, s, A, K, mol, cd, Hz, N, Pa, J, W, V, etc.) and their standard SI-prefixed forms may not all be available.

## Session 2026-07-07
- Error: `Invalid syntax near 'frame' — no valid parsing rule matches` — `frame` is a reserved keyword in SysML v2 and cannot be used as a part usage name.
  Fix: Renamed part usage from `frame` to `chassisFrame` to avoid keyword conflict.
- Error: `Couldn't resolve reference to Element 'inch'.` — `inch` is not a recognised SI unit symbol in the validator's library.
  Fix: Replaced `6.4 [inch]` with `162.56 [mm]` (converted inches to millimetres, a recognised SI unit).

## Session 2026-07-07
- Error: `Couldn't resolve reference to Element 'inch'.` — `inch` is not a recognised SI unit symbol in the validator's library.
  Fix: Replaced `6.5 [inch]` with `165.1 [mm]` (converted inches to millimetres, a recognised SI unit).
- Error: `Couldn't resolve reference to Element 'GHz'.` — `GHz` is not a recognised SI unit symbol (the validator only supports base SI units like `Hz`, `m`, `s`, `kg`, `K`, etc., not SI-prefixed compound units).
  Fix: Replaced `3.6 [GHz]` with `3.6e9 [Hz]` (expressed gigahertz as scientific notation in base hertz).
- Error: `Couldn't resolve reference to Element 'degC'.` — `degC` (degrees Celsius) is not a recognised SI unit symbol in the validator.
  Fix: Replaced `85.0 [degC]` with `358.15 [K]` (converted Celsius to Kelvin, the recognised SI base unit for temperature).

## Session 2026-07-04
- Error: `Invalid syntax near 'standard' — no valid parsing rule matches` — `standard` is a reserved keyword in SysML v2 and cannot be used as an attribute name.
  Fix: Renamed attributes from `standard` to `wifiStandard` and `nfcStandard` to avoid keyword conflict.

## Session 2026-07-02
- No new fixes.

## Session 2026-06-27
- No new fixes.

## Session 2026-06-26
- No new fixes.

## Session 2026-06-25
- No new fixes.

## Session 2026-06-25
- Error: `no viable alternative at input 'merge'` and `no viable alternative at input 'fork'` — `merge` and `fork` are reserved keywords in SysML v2 and cannot be used as enum literal names.
  Fix: Renamed enum literals from `merge` to `mergePoint` and from `fork` to `forkType` to avoid keyword conflicts.
- No new fixes.

## Session 2026-06-24
- No new fixes.

## Session 2026-06-24
- Error: `Unexpected '(', expecting one of: ;` — `perform action name()` with parentheses is invalid syntax. `perform action` does not use parentheses or return types.
  Fix: Removed all `()` and return types from `perform action` declarations. Use `perform action name;` (no parentheses, no return type).
- Error: `Couldn't resolve reference to Feature 'actuate'` — `perform action` requires a reference to an existing action definition. In a pure structural BDD without action definitions, `perform action` lines cause unresolved reference errors.
  Fix: Removed all `perform action` lines from the BDD. Operations on part definitions are behavioral and belong in activity diagrams, not structural BDDs. The BDD should only contain `part def`, `attribute`, `part` (composition), and `port` declarations.

## Session 2026-06-22
- No new fixes.

## Session 2026-06-21
- No new fixes.

## Session 2026-06-21
- Error: `Extra token '[' found, expecting one of: all, behavior, crosses...` — Array type syntax `Real[3]` and array literals `[0.6, 0.4, 0.32]` are not supported in SysML v2.
  Fix: Replaced array attributes with individual scalar attributes (e.g., `maxContainerWidthM`, `maxContainerDepthM`, `maxContainerHeightM` instead of `maxContainerDimensionsM : Real[3]`).

## Session 2026-06-20
- No new fixes.

## Session 2026-06-19
- No new fixes.

## Session 2025-06-17
- No new fixes.

## Session 2025-06-16
- Error: `Invalid syntax near 'message' — no valid parsing rule matches` — `message` is a reserved keyword in SysML v2.
  Fix: Rename the attribute from `message` to `commandContent` or any non-reserved name. Reserved keywords include: `message`, `variant`, `transition`, `analyze`, `detect`, `limit`, `activate`, `feedback`, `done`, `fork`, `join`, `decide`, `start`, `notify`, `signal`, `wait`, `receive`, `create`, `eval`, `check`, `restrict`, `continue`.

## Session 2025-04-18
- Error: `Duplicate definition 'lastaufnahmemittel': already defined as attribute_usage` — same name used for both an attribute and a part in the same part def.
  Fix: Use distinct names for attributes and parts within the same part def scope. Rename the attribute (e.g., `lamTyp` instead of `lastaufnahmemittel`) to avoid conflict with the part usage `lastaufnahmemittel : Lastaufnahmemittel[1]`.
- Error: `Unrecognized character or sequence: 'ä'` — Unicode characters in port def names.
  Fix: Use ASCII-only identifiers for all SysML v2 element names (part def, port def, item def, attribute names). Replace umlauts: ä→ae, ö→oe, ü→ue, ß→ss.

## Session 2025-04-17
- No new fixes.

## Session 2025-04-16
- No new fixes.

## Session 2025-04-15
- No new fixes.

## Session 2025-04-14
- No new fixes.
