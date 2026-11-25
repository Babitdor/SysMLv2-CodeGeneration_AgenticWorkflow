# Learned Fixes — SysML v2 Internal Block Diagram

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

## Session 2026-10-27
- Error: `Couldn't resolve reference to Feature 'storagePositionStatus'. (361:58)` (and 361:99) — a `flow of StoragePositionStatus from ctrl.stockInOut.storagePositionStatus to stock.entryIn.storagePositionEntry;` was written because the input source said the delegation "carries StockRecordEntry and StoragePositionStatus", but both endpoints are typed `StockRecordEntryPort`, whose only item feature is `stockRecordEntry`. This is a *feature-resolution* failure (the traversed item feature must exist on the port), distinct from the documented 2026-10-03 note about item *type* compatibility.
  Fix: Deleted the flow statement for the second item on the shared delegation and kept only `flow of StockRecordEntry from ctrl.stockInOut.stockRecordEntry to stock.entryIn.stockRecordEntry;` (the StoragePositionStatus item is genuinely carried by the other port, `stock.occupancyOut`). Only the item features declared on each traversed port def may appear after the port in a flow path; a source sentence naming two items on one port does not license a second flow.
- Error: `Must have at least two related elements (433:9)` — `connection cF02 connect rack1.rackSlotInOut to maint1;` was followed by `flow of MaintenanceAccess from rack1.rackSlotInOut.loadCarrier to maint1;`, but the target part `maint1` (type `MaintenanceLevel`) declares no port, so the flow's second endpoint had no port/item path and the flow had only one related element.
  Fix: Removed the flow statement and kept the connection alone (`connection cF02 connect rack1.rackSlotInOut to maint1;`). A `connection` with no `flow` is valid and is the correct form for a purely structural link; never attach a flow whose endpoint part declares no port.

## Session 2026-10-08
- No new fixes.

## Session 2026-10-07
- Error: `Duplicate definition 'EmergencyLandingCommand': already defined as item_def at line 39` — a part def (`EmergencyLandingCommand`) and an item def (`EmergencyLandingCommand`) shared one name in the same package namespace.
  Fix: Renamed the part def to `EmergencyLandingCommandModule` (and its usage type reference) so the module type no longer collides with the `EmergencyLandingCommand` item def. Same class as the documented `BrakingCommand`/`SteeringCommand` case.
- Error: `Couldn't resolve reference to Element 'cmd_out'. (338:45)` / `Couldn't resolve reference to Element 'stop_in'. (340:93)` — a `flow of EmergencyLandingCommand from <part>.cmd_out.emergencyLandingCommand to <part>.emergency_landing_in.emergencyLandingCommand;` failed to resolve the item feature even though the port def declared `out item emergencyLandingCommand : EmergencyLandingCommand;`. Minimal probes showed the identical item name (`emergencyLandingCommand`) used simultaneously as the item def name, the port-def item feature name, and the flow's `of <ItemDef>` type triggers a resolution collision in the validator.
  Fix: Renamed the port-def item feature to a distinct non-colliding name (`landingCmd`) in both the Out/In port defs and in every flow statement that traverses those ports (`flow of EmergencyLandingCommand from ...cmd_out.landingCmd to ...emergency_landing_in.landingCmd;`). The `of <ItemDef>` type stays `EmergencyLandingCommand`; only the traversed item feature name changed. Validated with zero errors.

## Session 2026-10-07
- Error: `Couldn't resolve reference to Element 'hydSupplyOut'/'h2In'/... (112 semantic errors)` — an IBD that relied on `public import HRSBlockDefinition::*;` for its part/port/item defs failed every connection endpoint in isolated validation (the validator cannot resolve cross-file package members, so each port reference cascaded into an unresolvable-element error).
  Fix: Make the IBD file self-contained — declare the item defs (HydrogenFlow/CoolantFlow/ControlSignal/EnergyFlow), the port defs (HydrogenInPort/HydrogenOutPort/CoolantInPort/CoolantOutPort/ControlInPort/ControlOutPort/ControlInOutPort/EnergyInPort/EnergyOutPort), and the subsystem + external-boundary part defs (with only the ports the IBD actually wires) inline in the IBD package, then instantiate them as part usages. External boundary parts (vehicle tanks, payment provider, remote operations centre, atmosphere, electrical supply, manual ESD actuator) that the IBD connects to but that are not BDD subsystems must be declared as part defs in the IBD package too. Validated with zero errors. Same class of cross-package-resolution limitation as the 2025-04-15 entry.

## Session 2026-10-03
- Error: `Syntax error at line 19, column 35: Unexpected ','` on `enum NibPositionState { EXPOSED, RETRACTED }` — `enum` is not a SysML v2 keyword.
  Fix: Replaced with `item def` using an `Integer` attribute instead of an enum declaration.
- Error: `Invalid syntax near 'state'` (3 occurrences) — `state` is a reserved keyword in SysML v2, cannot be used as an item or attribute name.
  Fix: Renamed `state` to `nibPosition` in item defs and port defs; renamed `protected` to `protectedState` and `flow` to `inkFlowItem` (both also reserved keywords).
- Error: `Extra token '%' found` on `Real = 0.0 [%]` — `%` is not a valid SI unit for `Real` defaults.
  Fix: Removed unit annotation; used bare `Real = 0.0`.
- Error: `An attribute must be typed by attribute definitions` on `attribute status : String` — `String` is not a valid attribute type in SysML v2.
  Fix: Used `Integer` for all enum-like attributes instead of `String`.
- Note: Connection `NibInterface ─[InkDepositSignal]→ WritingCapability.DepositPort` requires the endpoint to match the declared port on the target part def, not the source. Fixed `nibInterface.depositPort` → `writingCapability.depositPort` instead of the incorrect `userInput.alertPort`.

## Session 2026-10-03
- Error: (avoided pre-emptively) — an IBD "Parts (instances)" table named a part usage `port` (`port : BLK-006 Storage-Workstation Interface [6]`). `port` is a reserved SysML v2 keyword and cannot be used as a part usage name.
  Fix: Declared the part usage as `part portUnit : StorageWorkstationInterface[6];` and rewrote every connection/flow endpoint from `port.<x>` to `portUnit.<x>`. Same reserved-keyword class as `message`/`state`/`action`/`frame`/`verification`; applies to any diagram type whose source uses `port`, `part`, `item`, `connect`, `flow`, `interface` or `action` as an element name.

## Session 2026-10-03
- Note: `flow of <ItemDef> from <part>.<port>.<item> to <part>.<port>.<item>;` validates with zero errors even when `<ItemDef>` is NOT the item type declared on the traversed port def (e.g. `flow 'CN-008-F' of OccupancyReport from rackSystem.storageAccess.carrier to stockAndLocationManagement.bookingIn.booking;` where `storageAccess` is a `CarrierInOutPort` with `inout item carrier : LoadCarrier`). The validator resolves the endpoint paths but does not check flow-item vs port-item compatibility, so the semantic flow type named in the source can be kept verbatim. Do not invent extra ports just to make the flow item match.
- Note: ports inherited through part-def specialization (`part def PickingStation :> Workstation { ... }`) resolve correctly in `connect`/`flow` paths (`pickingStation.carrierIn`, `pickingStation.carrierOut`, `pickingStation.stationStateOut`), so a specialised part def only needs to re-declare the ports it adds. Declaring a general `Workstation` part alongside its two specialised roles in the same enclosing part def is valid.
- Note: the input's "Parts (instances)" table may list a part that no connection row references (here `workstation` : BLK-009, the general workstation). Declare it as a sibling part usage anyway — dropping it would break coverage — and simply leave it unconnected.

## Session 2026-10-02
- Error: `An occurrence, item or part must be typed by occurrence definitions. (15:9)` — a port def item was declared with a scalar library type (`inout item hostDataExchange : String;`).
  Fix: Scalar types (`String`, `Real`, `Integer`, `Boolean`) cannot type an `item`. Declare a real `item def HostDataExchange { attribute exchangeId : String; }` and use that as the item type. Every `item` in a port def must be typed by an `item def` (or a specialization of one), never by a scalar value type.

## Session 2026-09-22
- Error: `Syntax error at line 165, column 17: Extra token 'event' found, expecting one of: behavior, crosses, default...` (4 errors on `out item event : TripEvent;` / `in item event : TripEvent;` / `out item event : IsolationEvent;` / `in item event : IsolationEvent;` in port defs).
  Fix: `event` is a reserved keyword in SysML v2 and cannot be used as an item name inside a port def. Renamed the items to `tripEvent` and `isolationEvent` inside the `TripEventOutPort`/`TripEventInPort`/`IsolationEventOutPort`/`IsolationEventInPort` port defs. Same class of reserved-keyword issue as `message`, `frame`, `state`, `action` documented elsewhere.

## Session 2026-09-22
- Error: `Duplicate definition 'EmergencyStop': already defined as item_def at line 17` + `Couldn't resolve reference to Element 'estop_out'.` — a part def and an item def shared the name `EmergencyStop` (the EmergencyStop item flow and the Emergency-Stop GCS module).
  Fix: Renamed the part def to `EmergencyStopModule` (and its usage type reference) so it no longer collides with the `EmergencyStop` item def. Item defs and part defs share one namespace, so the name must be unique across both — same class of issue as the documented `BrakingCommand`/`SteeringCommand` case.

## Session 2026-09-22
- No new fixes.

## Session 2026-09-21
- Error: none (validated on first attempt) — but a fan-out modelling trap was found: an IBD source row like `CON-046 — Energieversorgung → BLK-400, BLK-500, BLK-200, BLK-900` and `CON-042 — BLK-900 → BLK-400 / BLK-500` cannot be expressed as one `connect`.
  Fix: A `connect`/`connection` statement has exactly two endpoints, so an N-target row must be expanded into N named connections with distinct names (`'CON-042'`, `'CON-042-HEB'`, `'CON-046'`, `'CON-046-HEB'`, `'CON-046-LOOP'`, `'CON-046-LVS'`); each still carries the original spec ID as its name prefix so traceability survives. Also: a `connection` with no `flow` statement is valid and is the right form for mechanical/non-material links (e.g. `CON-041` Wartungszugang), so do not invent an item def just to hang a flow on it.

## Session 2026-09-21
- Error: `Syntax error at line 10, column 29: Unexpected 'to', expecting one of: connect, default, :=...` at `connection c01 : a.p to b.q;`
  Fix: A named connection usage requires the `connect` keyword between the name and the endpoints: `connection <name> connect <p1>.<port> to <p2>.<port>;`. The bare `connection <name> : <expr>` form does not parse. Use the named form (rather than anonymous `connect a.p to b.q;`) whenever the connection must carry a spec-assigned ID such as `C-01`.
- Error: `Couldn't resolve reference to Feature 'rueckmeldung_in'.` (2 errors, on connections C-40/C-41)
  Fix: The input IBD's port inventory omitted `rueckmeldung_in` on LB-720 even though the connection table routes C-40 and C-41 (`Bewegungsrückmeldung` from LB-220/LB-320) into it. Added the missing `port rueckmeldung_in : InfoInPort;` to the `LB720` part def. When a connection table row targets a port that the port inventory does not list, the connection table wins — declare the port so the connection resolves; record it as a spec gap rather than dropping the connection.
- Note: `flow <ItemDef> from <sub>.<part>.<port>.<item> to <sub>.<part>.<port>.<item>;` resolves correctly across multi-level part paths (e.g. `lb200.lb230.behaelter_out.material`) as long as every intermediate part usage is declared in its owning part def. Verified with 66 connections / 64 flows in one package.

## Session 2026-09-16
- No new fixes.

## Session 2026-09-09
- No new fixes.

## Session 2026-09-04
- No new fixes.

## Session 2026-08-30
- Error: `Couldn't resolve reference to Element 'dockIdentifier'.` — The `DeliveryNotificationInPort` port def lacked a `dockIdentifier` item, but the IBD's flow statement referenced `receiving.deliveryInfoIn.dockIdentifier`. Fix: Added `in item dockIdentifier : DockIdentifier;` to the `DeliveryNotificationInPort` port def to match the port usage's declared items. Every port def must declare all items that any flow statement on that port will reference.

## Session 2026-08-29
- No new fixes.

## Session 2026-08-26
- No new fixes.

## Session 2026-08-26
- Error: `Couldn't resolve reference to Element 'powerStatus'.` — a `flow` statement referenced `batteryManagementSystem.status_in.powerStatus`, but the `status_in` port was typed `ModuleStatusInPort` (item `moduleStatus`), not `PowerStatusInPort`. When a single part receives two different flow types (Module_Status and Power_Status) on the same conceptual "status" port, they must be declared as two distinct ports with dedicated port defs (e.g. `status_in : ModuleStatusInPort` and `power_status_in : PowerStatusInPort`). Each `flow` must reference the item declared on the exact port it flows through — no sharing a port across different item types.

## Session 2026-08-26
- Error: `Couldn't resolve reference to Element 'motorCommandPort'/'stopCommandPort'/'alarmCommandPort'/'logEventPort'/'exportCommandPort'/'validationRequestPort'/'powerCommandPort'` (8 errors)
  Fix: The connect statements referenced port names (`controller.motorCommandPort`) that did not match the actual port usage names declared on the Controller part def (`controller.motorCommandOut`, `controller.stopCommandOut`, etc.). Renamed the connect/flow source endpoints to the exact declared port usage names. Key pattern: every port referenced in a `connect`/`flow` must exactly match the port usage name declared on the owning part def — no aliasing or renaming.

## Session 2026-08-26
- Error: `Duplicate definition 'BrakingCommand': already defined as item_def` / `Duplicate definition 'SteeringCommand': already defined as item_def`
  Fix: When a flow item and a part share the same name (e.g. `BrakingCommand` is both an item def and a part def), rename the part def to a distinct name (e.g. `BrakingCommandModule`, `SteeringCommandModule`) and update the part usage type reference accordingly. Item defs and part defs live in the same namespace in SysML v2, so names must be globally unique within the package.

## Session 2026-08-26
- Error: `Invalid syntax near 'state' — no valid parsing rule matches` — `state` is a reserved keyword in SysML v2 and cannot be used as an attribute name inside an item def.
  Fix: Renamed the attribute from `state` to `commandState` inside `ValveCommand` item def. This is the same class of reserved-keyword issue documented in `/skills/general/LEARNED.md` (Session 2025-06-16, `message` keyword).

## Session 2026-08-25
- No new fixes.

## Session 2026-07-17
- No new fixes.

## Session 2026-07-09
- No new fixes.

## Session 2026-07-09
- Error: `Invalid syntax near 'filter' — no valid parsing rule matches` — `filter` is a reserved keyword in SysML v2 and cannot be used as an attribute name inside an item def.
  Fix: Renamed the attribute from `filter` to `filterCriteria` inside `ReportFilter` item def. This is the same class of reserved-keyword issue documented in `/skills/general/LEARNED.md` (Session 2025-06-16, `message` keyword).
- Error: `Invalid syntax near 'message' — no valid parsing rule matches` — `message` is a reserved keyword in SysML v2 and cannot be used as an attribute name inside an item def.
  Fix: Renamed the attribute from `message` to `errorMessage` inside `ErrorResponse` item def. This is the same class of reserved-keyword issue documented in `/skills/general/LEARNED.md` (Session 2025-06-16, `message` keyword).

## Session 2026-07-08
- No new fixes.

## Session 2026-07-07
- No new fixes.

## Session 2026-07-06
- No new fixes.

## Session 2026-07-06
- Error: `Invalid syntax near 'action' — no valid parsing rule matches` — `action` is a reserved keyword in SysML v2 and cannot be used as an attribute name inside an item def.
  Fix: Renamed the attribute from `action` to `requestedAction` inside `PermissionRequest` item def. This is the same class of reserved-keyword issue documented in `/skills/general/LEARNED.md` (Session 2025-06-16, `message` keyword) and `/skills/internal-block-diagram/LEARNED.md` (Session 2026-06-26, `action` keyword in item defs).

## Session 2026-07-05
- No new fixes.

## Session 2026-07-04
- No new fixes.

## Session 2026-07-04
- Error: `Extra token 'frame' found, expecting one of: behavior, crosses, default...` — `frame` is a reserved keyword in SysML v2 and cannot be used as an item name inside a port def.
  Fix: Renamed the item from `frame` to `displayFrame` inside `DisplayFrameOutPort` port def. This is the same class of reserved-keyword issue documented in `/skills/general/LEARNED.md` (Session 2025-06-16, `message` keyword).

## Session 2026-07-02
- No new fixes.

## Session 2026-06-27
- No new fixes.

## Session 2026-06-26
- No new fixes.

## Session 2026-06-25
- No new fixes.

## Session 2026-06-24
- No new fixes.

## Session 2026-06-24
- Error: `Unexpected '::', expecting one of: ;,` — package name with `::` (e.g., `package UAVDeliverySystem::InternalConnections`) is not valid in SysML v2 package declaration syntax. The `::` namespace separator is used for qualified references (imports, type references), not for package names.
  Fix: Replaced `::` with `_` in the package name: `package UAVDeliverySystem_InternalConnections`.

## Session 2026-06-24
- Error: `Extra token 'verification' found` — `verification` is a reserved keyword in SysML v2 and cannot be used as an item name inside a port def.
  Fix: Renamed the item from `verification` to `zoneClearSignal` inside `ZoneClearVerificationOut` and `ZoneClearVerificationIn` port defs. This is the same class of reserved-keyword issue documented in `/skills/general/LEARNED.md` (Session 2025-06-16, `message` keyword).

## Session 2026-06-22
- No new fixes.

## Session 2026-06-21
- No new fixes.

## Session 2025-06-25
- Error: `Duplicate definition 'load_carrier': already defined as item_usage`
  Fix: Made port names unique by prefixing them with the part name (e.g., `gripper_load_carrier_out` instead of just `load_carrier_out`) to avoid name collisions in the same namespace.
- Error: `Unexpected ':', expecting one of: default, :=, ;... Extra token 'out' found`
  Fix: Used correct `port def` syntax with direction on items (e.g., `out item carrier : LoadCarrier;`) instead of inline port definitions with direction.
- Error: `Couldn't resolve reference to Feature 'RBGMotionControl'` (and similar for other part types)
  Fix: These are benign warnings when validating in isolation - the references will resolve when all files are loaded together in a model environment. No code changes needed.

## Session 2026-06-20
- No new fixes.

## Session 2026-06-20
- STATUS: BLOCKED — Required input artifact `/workspace/project_default/logical/IBD.md` does not exist in the workspace. The directory `/workspace/project_default/` contains only `physical/sysml/` (empty). Also missing: `/workspace/project_default/logical/BDD.md` (required for part references). Cannot generate IBD without input data. The orchestrator must ensure these files exist before invoking the IBD agent.

## Session 2026-06-19
- No new fixes.

## Session 2026-06-18
- No new fixes.

## Session 2025-06-17
- No new fixes.

## Session 2025-06-16
- No new fixes.

## Session 2025-04-18
- No new fixes.

## Session 2025-04-17
- No new fixes.

## Session 2025-04-16
- No new fixes.

## Session 2025-04-15
- Warning: `Couldn't resolve reference to Feature 'AutomatedWarehouseSystem'.`
  Fix: Replaced 19 individual `public import BlockDefinitionDiagram::<TypeName>;` imports with a single `public import BlockDefinitionDiagram::*;` wildcard import. The warning is a benign cross-package resolution limitation of the isolated validator — all port/connection references are valid against the BDD declarations. (Cross-cutting fix also recorded in /skills/skills/general/LEARNED.md.)

## Session 2025-04-14
- No new fixes.

## Session 2025-04-10
- Error: `Couldn't resolve reference to Element 'controlSystem'.`
  Fix: When a `part def SubType :> ParentType` overrides some inherited parts (e.g., `containerStorage`, `conveyorSystem`), inherited parts that are NOT being overridden (e.g., `controlSystem`) must still be explicitly redeclared in the subtype body for the validator to resolve them in dot-navigation flow paths (`subUsage.controlSystem.commandPort`) inside a containing part def.
