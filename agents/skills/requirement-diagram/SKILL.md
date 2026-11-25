---
name: requirement-diagram
description: Generate SysML v2 Requirement Diagrams (requirement defs with subject, formal require/assume constraints, doc text, priority, traceability) from Fragenkatalog input. Use as the first phase of the MBSE pipeline to produce USS requirements consumed by all downstream agents.
license: proprietary
---

## Purpose

Requirement diagrams are the *first phase* of an MBSE pipeline. They formalize stakeholder needs (Fragenkatalog items) as machine-checkable constraints, each with a subject, an explicit `require` predicate, and traceability to use cases that realize them. SysML v2 elevates requirements from documentation strings to first-class constraint definitions whose satisfaction can be verified automatically.

## Syntax Rules

- Use `requirement def` (NOT v1 `requirement` stereotype) — definitions are reusable templates
- Every requirement def MUST declare a `subject` (the part the requirement applies to)
- Use `doc /* ... */` for the natural-language SHALL statement — complete sentence with measurable criteria
- Use `require constraint { <expr> }` for the formal predicate that must hold in any valid solution
- Use `assume constraint { <expr> }` for preconditions that scope when the requirement applies
- Identifiers go in angle brackets: `requirement def <REQ-001> Name { ... }` — use `REQ-XX` format, sequential
- Priority via attribute (no built-in keyword): `attribute priority : Priority = must;` with enum `must|should|could|wont`
- Source attribute references the Fragenkatalog item: `attribute source : String = "Fragenkatalog 4.1.2";`
- For specialization use `specializes <ParentReq>`, redefine the subject with `subject redefines <name>`
- Subrequirements nest as `requirement : <ChildReqDef>;` inside a requirement def body
- NEVER use ellipsis (`...`), `TBD`, vague modal verbs ("should be good"), or undefined units

## Few-Shot Examples

### Example 1: Requirement Definition with Subject and Formal Constraint
**Source:** FAPS-archive/requirements-v2.sysml
```sysml
package WarehouseRequirements {
    public import ScalarValues::*;
    public import ISQ::*;

    enum def Priority { must; should; could; wont; }

    requirement def <'REQ-001'> RealtimeInventoryVisibility {
        subject warehouse : Warehouse;
        doc /* The system SHALL provide real-time inventory visibility across
               all warehouse zones with latency less than 5 seconds. */

        attribute priority : Priority = Priority::must;
        attribute source : String = "Fragenkatalog 3.2.1";
        attribute maxLatency : DurationValue = 5.0 [s];

        require constraint { warehouse.inventoryUpdateLatency <= maxLatency }
    }
}
```

### Example 2: Subrequirement Decomposition with Specialization
**Source:** FAPS-archive/requirements-decomposition.sysml
```sysml
package WarehouseRequirements {
    public import ISQ::LengthValue;

    requirement def <'REQ-002'> ReceivingZoneCapacity {
        subject zone : ReceivingZone;
        doc /* The receiving zone SHALL accommodate at least 10 trucks
               simultaneously with minimum berth length of 12.5 m. */

        attribute priority : Priority = Priority::must;
        attribute source : String = "Fragenkatalog 4.1.2";
        attribute minTrucks : Integer = 10;
        attribute minBerthLength : LengthValue = 12.5 [m];

        requirement <'REQ-002.1'> truckCount : TruckCountConstraint;
        requirement <'REQ-002.2'> berthGeometry : BerthGeometryConstraint;
    }

    requirement def TruckCountConstraint {
        subject zone : ReceivingZone;
        require constraint { zone.maxConcurrentTrucks >= 10 }
    }

    requirement def BerthGeometryConstraint {
        subject zone : ReceivingZone;
        require constraint { zone.berthLength >= 12.5 [m] }
    }
}
```

### Example 3: Requirement with Assumption and Use-Case Realization
**Source:** FAPS-archive/requirements-assumption.sysml
```sysml
package WarehouseRequirements {
    public import ISQ::LengthValue;

    requirement def <'REQ-003'> ForkliftAisleClearance {
        subject zone : StorageZone;
        doc /* The storage zone SHALL provide a minimum clearance of 2.5 m
               between rack rows for forklift operation. */

        attribute priority : Priority = Priority::must;
        attribute source : String = "Fragenkatalog 4.2.1";
        attribute minAisle : LengthValue = 2.5 [m];

        assume constraint { zone.usesForklifts == true }
        require constraint { zone.aisleWidth >= minAisle }
    }
}
```

## Error Patterns to Avoid

- **Missing `subject`** — without a subject the requirement cannot be satisfied or verified; ANTLR parser accepts it but downstream `satisfy` relations fail
- **Vague text** — "the system should be reliable" has no measurable criteria; always include a numeric threshold and units
- **No `require constraint`** — natural-language `doc` alone is not machine-checkable; pair every requirement with a formal predicate
- **Wrong priority** — using `must` for nice-to-haves pollutes the priority hierarchy used downstream by Layout / OR-Tools
- **Ellipsis or TBD placeholders** — never use `...`, `TBD`, or `<placeholder>` in requirement text or constraints
- **Forgetting attribute defaults** — every attribute needs a value: `attribute x : Real = 0.0;` not `attribute x : Real;`
- **Using v1 syntax** — do NOT use `requirement <<requirement>>`, `<<satisfy>>` stereotypes, or `requirement_id` strings; use the v2 angle-bracket identifier
- **Subject type undeclared** — the part type referenced as `subject` MUST be importable / declared earlier in the package

## USS Bindings

- **Consumes:** none (requirements are the first artifact in the pipeline)
- **Produces:**
  - `requirements` — `id`, `text` (from `doc`), `priority`, `source`, `subject_type`, `formal_constraint`
  - `use_cases` — `name`, `description`, `actors`, `scenarios`, `requirement_ids` (each requirement should map to ≥1 use case)
  - `artifacts` — the persisted SysML v2 source code, parsed AST, and validation report

## Learned Error Patterns

<!-- Error patterns recorded by skill_improver_tool during validation fixes.
     Each entry has a Pattern (what went wrong) and a Fix (what resolved it).
     These help future runs avoid repeating the same mistakes. -->


### Pattern
Used 'actor def' and 'block def' at package level - these are not valid keywords in SysML v2

### Fix
Use 'part def' for defining types at package level. Actors should be typed as 'actor name : PersonType' inside use cases, and Person types defined as 'part def PersonType'. For blocks, use 'part def' instead of 'block def' at package level.


### Pattern
Used 'derive requirement X from Y' syntax which is invalid in SysML v2

### Fix
Use doc comments inside requirement usages to document derivation from use cases instead. The 'derive' keyword is not valid for requirement-to-use-case relationships in SysML v2 textual notation.


### Pattern
Used 'verify R by test/inspection/analysis' syntax which is invalid standalone

### Fix
Remove standalone verify statements. Verification relationships in SysML v2 are typically shown in diagrams or using 'verify' keyword inside part definitions with verification cases, not as standalone statements.


### Pattern
Using 'satisfy' relationships at package level with use case usages - 'uc1' not defined in accessible scope

### Fix
In SysML v2, 'satisfy' relationships can only be used inside part definitions. Use case definitions can include comments documenting which requirements they satisfy, or use 'satisfy' inside a part definition with properly scoped use case references. For package-level traceability, use comments in use case definitions to document requirement mappings (e.g., // Satisfies: REQ-001, REQ-002).


### Pattern
Used 'activity def' syntax which is invalid in SysML v2; placed satisfy relationships at package level where requirements are not in scope

### Fix
Use 'requirement usage : RequirementDefName' at package level for traceability, then use 'satisfy r1 by A1' at package level. Requirements defined with 'requirement def <'ID'> Name' create definitions, and requirement usages reference them. The 'satisfy' keyword works at package level when referencing requirement usages, not requirement definitions directly.


### Pattern
Used 'satisfy' at package level with requirement usages - 'satisfy R1_usage by HydraulicDiscBrakeUnit;' caused semantic errors

### Fix
In SysML v2, 'satisfy' relationships must be used inside part definitions, not at package level. Declare requirements as part features (e.g., `requirement r1 : R1;`) inside the part definition, then use `satisfy r1;` to establish the satisfaction relationship within that part's scope.


### Pattern
Satisfy requirement scope errors - couldn't resolve feature references

### Fix
Declare requirement usages BEFORE they are referenced in satisfy statements. Each satisfy requirement needs its own requirement usage declared earlier in the package scope.


### Pattern
Requirement usages at package level with 'requirement <ID> name : ReqDefName;' caused semantic warnings about unresolved feature references

### Fix
Remove standalone requirement usages at package level. In SysML v2, requirement definitions (requirement def) are sufficient for the requirement diagram. Only use requirement usages inside part definitions where they are properly scoped. For package-level traceability, use comments documenting use case derivation relationships.


### Pattern
Used ISQ::LengthValue and DurationValue types that could not be resolved in attribute declarations

### Fix
Use ScalarValues::Real for numeric attributes with unit comments (e.g., 'attribute maxBrakingDistance : Real = 40.0; // meters') instead of ISQ::LengthValue or DurationValue, which may not resolve in all SysML v2 environments. Only use ISQ types when they are explicitly imported and confirmed available.


### Pattern
Using 'satisfy r1;' inside part definitions with 'requirement r1 : ApplyBraking;' caused unresolved feature warnings for requirement definitions

### Fix
Satisfy relationships inside part definitions cannot directly reference requirement definitions in this SysML v2 validator. Requirement diagrams should focus on requirement definitions with subjects, doc comments, formal constraints, and use cases. Document satisfaction relationships in comments rather than using satisfy keyword inside part definitions.


### Pattern
Used angle-bracket identifiers like <REQ-001> in requirement def names, causing 'satisfy REQ_001' to fail with unresolved reference errors

### Fix
Remove angle-bracket identifiers from requirement definitions. Use plain names like `requirement def ArticleStorageRequirement` and create separate requirement usages like `requirement r1 : ArticleStorageRequirement;` at package level. Reference these usages in satisfy relationships.
