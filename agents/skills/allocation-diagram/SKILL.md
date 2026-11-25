---
name: allocation-diagram
description: 'Generate SysML v2 Allocation Diagrams — `allocation def` with typed end declarations plus `allocate <source> to <target>` mappings that cross-cut the structural hierarchy. Use when the user wants to express realization links — function-to-component, behavior-to-structure, requirement-to-part, logical-to-physical, or any "X is realized by Y" relation that doesn''t fit inside either side''s containment tree.'
license: proprietary
---

## Purpose

An Allocation Diagram declares **realization mappings** that span two otherwise-independent hierarchies. The classic case is functional → physical: a `CaptureImage` action is *realized by* a `Camera` part; a `PlanPath` action is *realized by* a `NavigationComputer` part. Neither side has to know about the other inside its own definition — the `allocation def` and its `allocate` usages live in a separate namespace that connects the two.

SysML v1 modelled this as an `«allocate»` stereotype on a dependency relationship. SysML v2 promotes it to a first-class **`allocation def`** with typed `end` declarations, and the actual mapping is expressed by **`allocate <source> to <target>`** statements.

## Using FUNCTION_TREE.md (optional input)

The function side of a function→component allocation is exactly the `Function → Sub-Function → Activity (ACT-NNN)` hierarchy in `functional/FUNCTION_TREE.md`. When it's present, use it to choose **what** to allocate and at **what granularity**:

- Allocate at the level that matches the structure: map a whole sub-function to a subsystem part when one part realizes the entire group, or map individual leaf `ACT-NNN` actions to parts when realization is finer-grained. Let the tree's levels guide the choice.
- Avoid redundant allocations — if you allocate a parent function to a part, don't also allocate every child to the same part unless the realization genuinely differs per child.
- Use the tree to spot **coverage gaps**: every leaf activity should be realized by at least one part. A leaf with no allocation is an unallocated function — flag it rather than inventing a target.
- Sources/targets must still be real: every allocated function traces to `ACTIVITIES.md` (via its `ACT-NNN`), and every target part exists in `BDD.md`/`IBD.md`. The tree only organizes the function side; it never introduces new parts.
- If `FUNCTION_TREE.md` is absent, allocate the flat `ACT-NNN` activities from `ACTIVITIES.md` directly.

## Syntax Rules

- Use `allocation def <Name> { end <role1> : <Type1>; end <role2> : <Type2>; }` — exactly two ends, each typed (typically `ActionUsage`, `PartUsage`, `RequirementUsage`, etc.)
- The two ends are NOT direction-sensitive at the definition level — `allocate X to Y` picks the direction at the usage site
- Use `allocate <source> to <target>;` as a standalone statement (typically inside an `allocation def` body, an enclosing `allocation` usage, or a containing package)
- For a named, reusable allocation usage: `allocation <name> : <AllocationDef> { allocate <s> to <t>; }`
- The source / target identifiers are **resolved against the enclosing namespace** — usually you `import` both the functional and physical packages, then write bare `allocate funcName to partName`
- Use `import` rather than fully-qualified names: `import Functions::*; import Physical::*;` then `allocate CaptureImage to camera;`
- Imports MUST have visibility: `public import Functions::*;` `public import Physical::*;`
- Multiple allocations may share one definition — just write more `allocate` statements inside the same `allocation def` or sibling `allocation` usages
- NEVER use v1 stereotype syntax: `«allocate»`, `«allocation»`, `allocateFrom`, `allocateTo` are not v2 keywords
- NEVER omit the `to` keyword — `allocate X Y` is invalid; it must be `allocate X to Y`
- Allocations are PURELY annotational — they do NOT modify the source or target's owned features (no implicit composition, no implicit part-of)

## Few-Shot Examples

### Example 1: Function → Component Allocation (Canonical Pattern)

```sysml
package FunctionalArchitecture {
    public import ScalarValues::*;

    action def CaptureImage;
    action def DetectObstacle;
    action def PlanPath;
}

package PhysicalArchitecture {
    public import ScalarValues::*;

    part def Camera;
    part def ProcessingUnit;
    part def NavigationComputer;

    part def Robot {
        part camera     : Camera;
        part processor  : ProcessingUnit;
        part navigator  : NavigationComputer;
    }
}

package FunctionToPhysicalAllocation {
    public import FunctionalArchitecture::*;
    public import PhysicalArchitecture::*;

    allocation def FunctionToComponent {
        end function  : ActionUsage;
        end component : PartUsage;
    }

    allocate CaptureImage   to Robot::camera;
    allocate DetectObstacle to Robot::processor;
    allocate PlanPath       to Robot::navigator;
}
```

### Example 2: Logical → Physical Allocation with Named Usage

```sysml
package LogicalToPhysical {
    public import ScalarValues::*;
    public import LogicalArchitecture::*;
    public import PhysicalArchitecture::*;

    allocation def LogicalToPhysical {
        end logical  : PartUsage;
        end physical : PartUsage;
    }

    // Named usage groups several related allocations under one identifier
    // that can be referenced from a view or a verification case.
    allocation toLogisticsSystem : LogicalToPhysical {
        allocate ReceivingZone to dockArea;
        allocate StorageZone   to rackArea;
        allocate ShippingZone  to dispatchArea;
    }
}
```

### Example 3: Requirement → Part Allocation (Satisfaction-by-Allocation)

```sysml
package RequirementAllocation {
    public import ScalarValues::*;
    public import VehicleRequirements::*;
    public import VehicleParts::*;

    allocation def RequirementToPart {
        end req  : RequirementUsage;
        end part : PartUsage;
    }

    allocate MaxMassLimit       to chassis;
    allocate FrontalImpactRating to frontCrumpleZone;
    allocate BrakingDistance    to brakeSystem;
}
```

## Common Pitfalls

- **Forgetting the `to` keyword** → validator error `Missing expected token 'to'` — every `allocate` statement is exactly `allocate <source> to <target>;`
- **Source / target not in scope** → validator error `Couldn't resolve reference to Element` — the source and target names must be imported (`import Functions::*;`) or fully qualified (`Functions::CaptureImage`)
- **Using v1 stereotype syntax** (`«allocate»`, `«allocation»`) → these are UML-style stereotype annotations and do not exist in v2's textual notation; rewrite as `allocation def` + `allocate`
- **Wrong end count** → `allocation def` must have EXACTLY two `end` declarations. Three-way mappings get split into multiple allocation defs
- **Treating allocation as composition** → an allocation does NOT make the target a part of the source (or vice versa); it's an annotational link, not a structural change
- **Asymmetric typing** → both ends are typed; if `end function : ActionUsage` is on the left, the source identifier in `allocate <source> to <target>` must resolve to an `ActionUsage`. Type-mismatched allocations may surface as semantic errors

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "Each function is realized by a physical component." | `allocation def FunctionToComponent { end function : ActionUsage; end component : PartUsage; }` + `allocate` per pair |
| "The logical receiving zone is implemented by the dock area part." | `allocation def LogicalToPhysical` between two `PartUsage` ends |
| "REQ-007 is satisfied by the brake-system component." | `allocation def RequirementToPart { end req : RequirementUsage; end part : PartUsage; }` + `allocate REQ-007 to brakeSystem;` (consider also `satisfy` if the link is requirement satisfaction, which has solver semantics) |
| "Group several related allocations under one name for a view." | Named allocation usage: `allocation <name> : <AllocDef> { allocate …; allocate …; }` |
| "Map a behavior (action) onto a part WITHOUT changing either side." | The canonical case — separate package, import both sides, declare the def + allocate statements |
