---
name: use-case-diagram
description: Generate SysML v2 Use Case Diagrams (use case defs with subject, typed actor parameters, include/extend relations, perform action, satisfy requirement traces) from USS requirements. Use when the user wants to model what the system does for whom and bind behaviour to actions.
license: proprietary
---

## MINIMAL WORKING TEMPLATE
Copy this EXACT structure and modify only the names:

```sysml
package <YourPackageName> { 
    public import ScalarValues::*;
    
    // Actor/Subject type definitions
    part def ActorType;
    part def SubjectType;
    
    // Use case definition - SIMPLE VERSION (no optional parts)
    use case def 'Use Case Name' {
        subject subjectName : SubjectType;
        actor actorName : ActorType;
        satisfy R1;
    }
    
    // Requirement definition for tracing
    requirement def R1 { doc /* Requirement text */ }
}
```

**CRITICAL SYNTAX RULES:**
1. ALWAYS use STRAIGHT single quotes `'` NOT fancy quotes `'` `
2. Subject and actor MUST be typed: `subject name : Type;` NOT just `subject name;`
3. If using objective, use format: `objective { doc /* text */ }` (without title string)
4. Use `satisfy R1;` NOT `satisfy requirement R1;`

## Purpose

Use case diagrams define _what the system does for whom_. SysML v2 elevates use cases from loose UML stereotypes to first-class behavioral elements with an explicit `subject` (the system under consideration), typed `actor` parameters (external participants), and machine-checkable `objective` predicates. Each use case scenarios out to actions in an Activity diagram and traces back to one or more requirements.

## Syntax Rules

- Use `use case def <Name> { ... }` for the type; `use case <name> : <UseCaseDef>` for the usage
- MUST declare exactly one `subject <name> : <PartDef>;` — the system targeted by the use case
- Declare external participants with `actor <name> : <PartDef>;` (one or more); actors are typed parts
- Express the "why" with `objective '<title>' { doc /* ... */ }` — natural-language goal of the use case
- Reuse base behaviour with `include use case <name> : <BaseUseCaseDef>;` (mandatory inclusion)
- Add optional behaviour conditionally with `extend use case <name> : <ExtUseCaseDef> if <guard>;`
- Use `perform action <name> : <ActionDef>;` to bind the use case to a behavioral action chain
- Quote names with spaces in single quotes: `use case def 'Purchase Ticket' { ... }`
- Trace to requirements: `subject :>> <name> = <part>;` and reference requirements via `satisfy`
- NEVER use UML stereotypes like `<<actor>>`, `<<include>>`, `<<extend>>` — they are v1 syntax

## Few-Shot Examples

### Example 1: Use Case Definition with Subject, Actors, and Objective

**Source:** FAPS-archive/use-case-v1.sysml

```sysml
package WarehouseUseCases {
    public import ScalarValues::*;
    public import WarehouseBlocks::*;

    use case def 'Receive Shipment' {
        subject warehouse : Warehouse;
        actor   driver    : TruckDriver;
        actor   clerk     : ReceivingClerk;

        objective 'Inbound shipment validated and stored' {
            doc /* The driver delivers a shipment; the clerk verifies it
                   against the purchase order and stores accepted items
                   in the storage zone within 30 minutes of arrival. */
        }
    }

    part def TruckDriver;
    part def ReceivingClerk;
}
```

### Example 2: Include and Extend Relationships

**Source:** FAPS-archive/use-case-include-extend.sysml

```sysml
package WarehouseUseCases {
    public import ScalarValues::*;
    public import WarehouseBlocks::*;

    use case def 'Process Order' {
        subject warehouse : Warehouse;
        actor   customer  : Customer;

        objective { doc /* Customer order is fulfilled and shipped. */ }

        include use case validate : 'Validate Inventory';
        include use case pick     : 'Pick Items';
        include use case ship     : 'Ship Order';

        extend use case backorder : 'Backorder Item' if anyOutOfStock;

        attribute anyOutOfStock : Boolean = false;
    }

    use case def 'Validate Inventory' {
        subject warehouse : Warehouse;
    }

    use case def 'Pick Items' {
        subject warehouse : Warehouse;
        actor   picker    : WarehouseWorker;
    }

    use case def 'Ship Order' {
        subject warehouse : Warehouse;
        actor   carrier   : ShippingCarrier;
    }

    use case def 'Backorder Item' {
        subject warehouse : Warehouse;
        actor   buyer     : ProcurementOfficer;
    }

    part def Customer;
    part def WarehouseWorker;
    part def ShippingCarrier;
    part def ProcurementOfficer;
}
```

### Example 3: Use Case Bound to an Action and Traced to Requirement

**Source:** FAPS-archive/use-case-bound.sysml

```sysml
package WarehouseUseCases {
    public import WarehouseBlocks::*;
    public import WarehouseActivities::*;
    public import WarehouseRequirements::*;

    use case def 'Real-time Inventory Query' {
        subject warehouse : Warehouse;
        actor   operator  : WarehouseOperator;

        objective { doc /* Operator obtains current stock level under 5 s. */ }

        perform action query : doQueryInventory {
            in :>> warehouseRef = warehouse;
            in :>> operatorId   = operator.id;
        }

        satisfy requirement realtime : RealtimeInventoryVisibility;
    }

    part def WarehouseOperator { attribute id : String; }
    action def doQueryInventory {
        in warehouseRef : Warehouse;
        in operatorId   : String;
        out latency     : Real = 0.0;
    }
}
```

## Error Patterns to Avoid

- **No `subject`** — every use case def MUST declare its subject; without it the trace to a part def is missing
- **Floating actors** — actors are _part usages typed by part defs_, not loose strings; declare the part def somewhere
- **UML v1 stereotypes** — `<<include>>`, `<<extend>>`, `<<actor>>` are not v2; use `include use case`, `extend use case`, `actor`
- **`extend` without guard** — `extend use case x : Y;` is incomplete; always specify `if <condition>`
- **Missing import for actor types** — actor types must be defined or imported in scope (`public import ...`)
- **Use case names with spaces unquoted** — `use case def Process Order` is a parse error; use `'Process Order'`
- **Action without binding** — `perform action a : ADef` referenced inside a use case must come from the Activity diagram package
- **Forgetting the `satisfy` link** — every use case should trace to ≥1 requirement via `satisfy requirement`

## USS Bindings

- **Consumes:**
  - `requirements` — `id`, `text`, `subject_type` (use cases realize requirements; the trace is mandatory)
- **Produces:**
  - `use_cases` — `name`, `description` (from objective doc), `subject_type`, `actors` (list of types), `included_use_cases`, `extended_use_cases`, `bound_action_id`, `requirement_ids`, `scenarios` (sequence of perform-action chains)
  - `artifacts` — persisted SysML v2 source, AST, validation report

## Learned Error Patterns

<!-- Error patterns recorded by skill_improver_tool during validation fixes.
     Each entry has a Pattern (what went wrong) and a Fix (what resolved it).
     These help future runs avoid repeating the same mistakes. -->

### Pattern

Duplicate subject definition 'warehouseSystem' - same subject name used across multiple use case definitions causes duplicate definition error

### Fix

Use unique subject names for each use case definition. Each use case def must have a unique subject identifier (e.g., 'warehouseSystem', 'throughputSystem', 'continuousOpsSystem', etc.) to avoid duplicate definition conflicts.


### Pattern
Used incorrect syntax for doc strings (doc "") and use case definitions with subject/actor/objective inside def. Also used incorrect action def syntax with 'in input' and 'out output'.

### Fix
Use '<' for doc strings: doc < "text". Use case defs should NOT contain subject/actor/objective - those belong in use case USAGES. Action defs use 'in' and 'out' directly followed by parameter names, not 'in input' or 'out output'.


### Pattern
Duplicate definition errors for subject names like 'brakeSystem' used in multiple use case definitions

### Fix
Use unique feature names for each subject in use case definitions, or reference existing features from the package scope using qualified names like `brakeSystem = brakeSystem` or use unique aliases per use case.


### Pattern
Duplicate subject names across use case definitions cause 'Duplicate definition' errors

### Fix
Use unique subject names in each use case definition (e.g., theBrakeSystem, theABSBrakeSystem, theSafeBrakeSystem, brakingSystem1, etc.) instead of reusing the same name across multiple use case definitions.


### Pattern
JSON metadata leaked into code field and incorrect use case syntax

### Fix
The 'code' field must contain ONLY pure SysML v2 source. Use proper syntax: use case def 'Name' with single quotes for names with spaces, objective { doc /* ... */ } for goals, perform action name : 'ActionDef'; for behavior binding, and satisfy R4; for requirement tracing. Use unique subject names in each use case to avoid duplicate definitions.


### Pattern
Duplicate definition errors for subject names across multiple use case definitions. Reusing the same subject name (e.g., 'brkSys') in multiple use case defs causes "already defined as part_usage" errors.

### Fix
Use unique subject names for each use case definition within the same package. For example: brakeSubsystemA, brakeSubsystemB, brakeSubsystemC, etc. Each use case def needs its own unique subject identifier.


### Pattern
JSON metadata leaked into code field and missing requirement definitions

### Fix
The 'code' field must contain ONLY pure SysML v2 source. When using satisfy R1; or similar requirement references, the requirement definitions (requirement def R1 { ... }) must be included in the same package or imported. Never include JSON metadata, validated flags, or source_entity_ids inside the SysML code.


### Pattern
Used 'satisfy R1 by useCaseName' syntax which is invalid in SysML v2

### Fix
Use inline satisfy statements inside the use case definition block: use case 'Name' : UseCaseDef { ... satisfy R1; } instead of separate satisfy clauses


### Pattern
Duplicate definition errors for subject names like 'theBrakeSystem', 'absBrakeSystem', 'safeBrakeSystem', 'feedbackBrakeSystem' used in both use case def and use case usage

### Fix
Use unique subject names for each use case definition. Each use case def must have a unique subject identifier (e.g., 'brakeSys', 'absSys', 'safeSys', 'feedbackSys'). Do not create separate use case usages with the same subject names - just use use case definitions directly. Each use case def should have its own unique subject name.


### Pattern
Couldn't resolve reference to Feature 'detectFailure' when using perform action syntax inside use case def

### Fix
Remove perform action bindings from use case definitions when the action definitions are not properly imported or defined in scope. Use only satisfy requirement for tracing to requirements.


### Pattern
Use case names with spaces were incorrectly quoted with single quotes, causing parser errors. The validator expected proper include/extend syntax with named usage references.

### Fix
Use single quotes for use case definition names with spaces (e.g., `use case def 'Apply Braking'`). For include/extend relationships, use the pattern `include use case <name> : '<Use Case Def Name>';` where the name is a local reference and the type is the quoted use case definition name. Use `:>>` for subject/actor redefinition in derived use cases to avoid duplicate definition errors.


### Pattern
JSON metadata leaked into SysML code - the output JSON with source_entity_ids was appended inside the SysML code block, causing parser errors on every line

### Fix
Ensure the 'code' field contains ONLY pure SysML v2 source code. Never append JSON metadata, source_entity_ids arrays, or validated flags inside the SysML code. Return those as separate JSON fields only.


### Pattern
Using 'perform action' inside use case def caused semantic errors - couldn't resolve references to action features

### Fix
Use 'action <name> : <ActionDef>;' inside use case def to reference action definitions, not 'perform action'. The 'perform action' is used in part definitions to instantiate actions.


### Pattern
perform action statements caused "Couldn't resolve reference to Feature" errors because the action definitions were not in scope

### Fix
Remove perform action statements from use case defs when actions are not defined in the same package. Use only subject, actor, objective, and satisfy requirement to keep the code self-contained and valid.


### Pattern
Using 'objective { doc /* ... */ }' inside requirement definitions caused syntax errors - validator expected 'doc /* ... */' directly in requirement def body

### Fix
Use 'doc /* ... */' directly inside requirement def body without wrapping in 'objective { }'. The objective keyword is only valid inside use case definitions, not requirement definitions.


### Pattern
Using named perform action syntax like 'perform action receive : ReceiveGoodsAction;' caused reference errors

### Fix
Use anonymous perform action syntax without name: 'perform action : ReceiveGoodsAction;' — the name is optional in use case definitions


### Pattern
Using unique subject names per use case definition (warehouseSys, warehouseSys2, warehouseSys3, warehouseSys4) avoids duplicate definition errors

### Fix
Each use case definition must have a unique subject identifier to avoid duplicate definition conflicts. Use numbered or suffixed subject names like warehouseSys, warehouseSys2, etc.


### Pattern
Using use case def with subject, actor, objective, and satisfy requirement syntax works correctly

### Fix
Define use case def with subject typed by part def, actor typed by part def, objective with doc comment, and satisfy requirement statements for requirement tracing. Use unique subject names per use case definition to avoid duplicate definition errors.
