---
name: block-definition-diagram
description: Generate SysML v2 Block Definition Diagrams (part defs with attributes, ports, generalization, composition) from USS functions and requirements. Use when the user wants to declare structural types and their relationships, producing USS logical_blocks for IBD and Layout agents.
license: proprietary
---

## Purpose

A Block Definition Diagram (BDD) declares the *types* in the system — `part def`, `attribute def`, `port def`, `interface def`, `item def` — and the generalization / composition relationships between them. In SysML v2 the v1 `block` keyword has been replaced by `part def`. The BDD answers "what kinds of things exist and how do they classify?", while the IBD answers "how do instances of those parts wire together inside a context?".

## Syntax Rules

- Use `part def <Name> { ... }` — NOT v1 `block <Name>`
- Use `:>` (specializes) for generalization between part definitions: `part def Truck :> Vehicle`
- Use `:>>` for redefinition of an inherited feature: `attribute mass :>> Vehicle::mass = 4500.0 [kg];`
- Composition uses `part <name> : <PartDef>[multiplicity];` inside a parent `part def`
- Multiplicities: `[1]`, `[0..1]`, `[*]`, `[1..*]`, `[N]` — declared in square brackets after the type
- Every attribute MUST have type AND default value: `attribute capacity : Integer = 1000;`
- Use SI units inside square brackets: `attribute height : LengthValue = 12.0 [m];`
- Imports MUST have visibility: `public import ScalarValues::*;` `public import ISQ::*;` `public import SI::*;`
- Constraints inside a part def reference *only* features owned by that part def or its ancestors (scope rule)
- Use `enum def Name { lit1; lit2; }` for enumerations — reference via `Name::lit1`
- Naming convention: PascalCase for definitions (`StorageZone`), camelCase for usages (`storageZone`)
- NEVER use `class`, `block`, `redefines`, ellipsis, or untyped attributes

## Few-Shot Examples

### Example 1: Part Definitions with Composition and Generalization
**Source:** FAPS-archive/bdd-v2.sysml
```sysml
package WarehouseBlocks {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    part def Zone {
        attribute name : String = "Generic Zone";
        attribute area : AreaValue = 0.0 [m * m];
    }

    part def ReceivingZone :> Zone {
        attribute maxTrucks : Integer = 10;
        attribute berthLength : LengthValue = 12.5 [m];
    }

    part def StorageZone :> Zone {
        attribute rackCount : Integer = 100;
        attribute aisleWidth : LengthValue = 2.5 [m];
    }

    part def LoadingZone :> Zone {
        attribute dockCount : Integer = 8;
        attribute dockWidth : LengthValue = 4.5 [m];
    }

    part def Warehouse {
        attribute name : String = "Main Warehouse";
        attribute totalArea : AreaValue = 5000.0 [m * m];
        attribute capacity : Integer = 10000;

        part receiving : ReceivingZone[1];
        part storage   : StorageZone[1..*];
        part loading   : LoadingZone[1];
    }
}
```

### Example 2: Part Def with Owned Constraint
**Source:** FAPS-archive/bdd-constraint.sysml
```sysml
package WarehouseBlocks {
    public import ScalarValues::*;

    part def Warehouse {
        attribute maxCapacity : Integer = 10000;
        attribute currentOccupancy : Integer = 0;

        constraint capacityRule {
            currentOccupancy <= maxCapacity
        }
    }
}
```

### Example 3: Part Def with Ports Declared on Boundary
**Source:** FAPS-archive/bdd-ports.sysml
```sysml
package WarehouseBlocks {
    public import ScalarValues::*;
    public import ISQ::MassValue;

    item def Pallet {
        attribute weight : MassValue = 25.0 [kg];
    }

    port def MaterialFlowPort {
        out item shippedPallet : Pallet;
        in  item receivedPallet : Pallet;
    }

    part def StorageZone {
        attribute rackCount : Integer = 100;

        port inboundDock  : MaterialFlowPort;
        port outboundDock : ~MaterialFlowPort;
    }
}
```

## Error Patterns to Avoid

- **Using `block`** — v1 keyword, removed in v2; always `part def`
- **Using `redefines`** — v1 keyword, replaced by `:>>` in v2
- **Missing attribute defaults** — `attribute x : Real;` is incomplete; must be `attribute x : Real = 0.0;`
- **Forgetting visibility on imports** — bare `import` is parsed but flagged; always prefix `public` or `private`
- **Reference before declaration** — features used in `constraint` bodies must be declared earlier in the same scope
- **Wrong unit syntax** — use `5.0 [m]` (square brackets), NOT `5.0 m` or `5.0 "m"`
- **Multiplicity in wrong position** — multiplicity follows the type, not the name: `part p : T[*];` NOT `part p[*] : T;`
- **Mixing parts and attributes confusingly** — `part` for structural composition, `attribute` for value properties; don't model a measurement as a part

## USS Bindings

- **Consumes:**
  - `functions` — `id`, `name`, `input_params`, `output_params` (each function maps to ≥1 part def or constraint)
  - `requirements` — `id`, `subject_type` (drives which part def is generated)
- **Produces:**
  - `logical_blocks` — `name`, `description`, `block_type`, `attributes`, `ports`, `inputs`, `outputs`, `requirement_ids`, `function_ids`
  - `artifacts` — persisted SysML v2 source, AST, validation report

## Learned Error Patterns

<!-- Error patterns recorded by skill_improver_tool during validation fixes.
     Each entry has a Pattern (what went wrong) and a Fix (what resolved it).
     These help future runs avoid repeating the same mistakes. -->


### Pattern
Used doc "string literal" syntax for documentation - SysML v2 requires doc /* block comment */ syntax

### Fix
Changed all doc statements from `doc "description"` to `doc /* description */` using block comment syntax instead of string literals


### Pattern
Used 'doc "..."' inside requirement/constraint/part definitions - invalid SysML v2 syntax

### Fix
Use // comments for documentation instead of 'doc "..."' keyword inside definitions. For requirements, use 'subject <name> : <Type>; require constraint { <expression> }' syntax. For constraints, use 'in <param> : <Type>; <expression>' syntax.


### Pattern
Using ISQ types (MassValue, LengthValue) caused unresolved reference warnings in validator

### Fix
Use Real type with SI units in square brackets instead of ISQ quantity types. This avoids import resolution issues while still providing unit semantics through SI::* import.


### Pattern
Used constraint name { expression } syntax inside part definitions - invalid SysML v2 syntax

### Fix
Use inline constraint syntax with double curly braces: constraint {{ expression }} instead of constraint name { expression }


### Pattern
Used [%] as a unit in attribute declaration - caused syntax error

### Fix
Remove the [%] unit and use just the numeric value: attribute pickingAccuracy : Real = 99.5; instead of attribute pickingAccuracy : Real = 99.5 [%];


### Pattern
Using BDD with part definitions, item definitions, port definitions, and composite parts with nested sub-blocks works correctly

### Fix
Generated package with item def (Container, Pallet), port def (MaterialPort, ControlPort), part def for each subsystem (ShuttleSubsystem, LiftSubsystem, RackStorageSubsystem, ConveyorInterfaceSubsystem, WarehouseControlSubsystem with nested parts, PowerDistributionSubsystem), and top-level WarehouseSystem composition. All attributes have type and value, ports declared on part boundaries, SI units in square brackets.


### Pattern
Using ISQ types like LengthValue and MassValue directly causes semantic warnings about unresolved references

### Fix
Use Real type with SI unit annotations (e.g., `attribute width : Real = 2500.0 [mm];`) instead of ISQ types. The unit annotations in square brackets provide the quantity information while avoiding type resolution issues.


### Pattern
Adding layout generation dimensions to BDD parts

### Fix
Added posX, posY, posZ (position coordinates) and sizeX, sizeY, sizeZ (dimensions in meters) attributes to each part definition for layout generation. WarehouseSystem also includes building dimensions as sizeX, sizeY, sizeZ.


### Pattern
Using enum def with lit keyword causes syntax error in SysML v2

### Fix
Use enum def with literal names directly without 'lit' keyword: enum def BrakeSystemState { operational; degraded; failSafe; }


### Pattern
BDD with item defs, port defs, interface defs, enum defs, part defs with composition and constraints passes validation

### Fix
Generated package with item def (BrakeFluid, ElectricalSignal, WheelSpeedData), port def (HydraulicPort, SignalPort, SensorPort), interface def (BrakeLineInterface, SensorInterface), enum def (BrakeSystemState), part def for each subsystem (BrakePedal, BrakeMasterCylinder, HydraulicLine, BrakeCaliper, BrakeDisc, BrakePad, WheelSpeedSensor, ABSController, WheelAssembly with nested parts), and top-level BrakeSystem composition. All attributes have type and value, ports declared on part boundaries, SI units in square brackets, constraints use inline syntax.


### Pattern
Enum definition with comma-separated or semicolon-separated literals caused syntax errors

### Fix
Use 'enum def Name { enum Literal1; enum Literal2; }' syntax with 'enum' keyword before each literal name and semicolons
