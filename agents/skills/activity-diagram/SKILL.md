---
name: activity-diagram
description: Generate SysML v2 Activity Diagrams (action defs with control flow) from USS requirements. Use when the user requests behavioral modeling of action sequences, decisions, parallel forks, or loops, and produces USS functions consumed by BDD/IBD agents.
license: proprietary
---

## Purpose

Activity diagrams model *behavior* — the ordered set of actions that transform inputs into outputs to satisfy requirements. In SysML v2, the v1 `activity` keyword is replaced by `action def`, and control flow uses `first ... then ... then done` rather than UML-style activity edges. Each action becomes a USS `function` that the BDD/IBD agents consume to derive structural blocks.

## Using FUNCTION_TREE.md (optional input)

`ACTIVITIES.md` is a flat list of `ACT-NNN` activities. When `functional/FUNCTION_TREE.md` is also present, it gives the `Function → Sub-Function → Activity (ACT-NNN)` hierarchy — use it to **structure** the diagram, not to add content:

- Model each non-leaf node (a high-level function / sub-function) as a composite `action def` whose body sequences the child actions; model each leaf `ACT-NNN` as the concrete `action def` it already maps to in `ACTIVITIES.md`. This turns a flat action soup into a readable decomposition.
- Keep the parent→child nesting exactly as the tree declares it: every `ACT-NNN` sits under the one parent function the tree assigns it.
- The tree is structure only. Every action, parameter, decision, and flow you emit must still trace to `ACTIVITIES.md`. Never invent an action just because the tree names a function with no backing `ACT-NNN`.
- If `FUNCTION_TREE.md` is absent, fall back to a flat sequence of the `ACT-NNN` actions from `ACTIVITIES.md`.

## Syntax Rules

- Use `action def <Name> { ... }` for an action *type*; `action <name> : <ActionDef>` for a usage
- Action names follow `do<Verb>` or `<verb><Object>` convention: `doReceiveShipment`, `verifyShipment`
- Parameters: `in <name> : <Type> = <default>;` and `out <name> : <Type>;` (in/out direction is mandatory)
- Use `first start;` to mark the initial node, `then done;` to mark the final node
- Sequence: `first <action1> then <action2>` — the `then` keyword chains actions in execution order
- Decision: `then decide; if <guard> then <action>; else <action>;` — guards are boolean expressions
- Parallel fork: `then fork;` followed by branches; merge with `then join;`
- Loop: `then while <guard> { <body> }` or use a recursive action call
- Event acceptance: `accept <evt> : <SignalType>` — for asynchronous triggers
- Calc expressions for derived values: `calc def <name>(in x : Real) : Real { x * 2.0 }`
- NEVER use ellipsis, untyped parameters, or unguarded decision branches

## Few-Shot Examples

### Example 1: Sequential Action with Decision
**Source:** FAPS-archive/activity-v2.sysml
```sysml
package WarehouseActivities {
    public import ScalarValues::*;
    public import ISQ::MassValue;

    item def Shipment {
        attribute itemId : String;
        attribute quantity : Integer = 0;
        attribute weight : MassValue = 0.0 [kg];
    }

    action def doReceiveShipment {
        in shipment : Shipment;
        out accepted : Boolean = false;

        first start;
        then verify : doVerifyShipment {
            in :>> shipment = shipment;
        }
        then decide;
            if verify.matches then accept : doStoreItems {
                in :>> shipment = shipment;
            }
            else reject : doFlagDiscrepancy {
                in :>> shipment = shipment;
            }
        then done;
    }

    action def doVerifyShipment {
        in shipment : Shipment;
        out matches : Boolean = false;
    }

    action def doStoreItems { in shipment : Shipment; }
    action def doFlagDiscrepancy { in shipment : Shipment; }
}
```

### Example 2: Parallel Fork / Join
**Source:** FAPS-archive/activity-parallel.sysml
```sysml
package WarehouseActivities {
    public import ScalarValues::*;

    action def doProcessOrder {
        in orderId : String;

        first start;
        then fork;
            branch payment   : doValidatePayment   { in :>> orderId = orderId; }
            branch reserve   : doReserveInventory  { in :>> orderId = orderId; }
            branch notify    : doNotifyWarehouse   { in :>> orderId = orderId; }
        then join;
        then fulfill : doFulfillOrder { in :>> orderId = orderId; }
        then done;
    }

    action def doValidatePayment   { in orderId : String; }
    action def doReserveInventory  { in orderId : String; }
    action def doNotifyWarehouse   { in orderId : String; }
    action def doFulfillOrder      { in orderId : String; }
}
```

### Example 3: Iterative Picking with Loop Guard
**Source:** FAPS-archive/activity-loop.sysml
```sysml
package WarehouseActivities {
    public import ScalarValues::*;

    action def doPickItems {
        in orderId : String;
        in itemList : String[*];

        first start;
        then while { itemList->notEmpty() } {
            pick : doPickNextItem  { in :>> itemId = itemList->first(); }
            then move : doMoveToPacking { in :>> itemId = itemList->first(); }
        }
        then consolidate : doConsolidatePickedItems { in :>> orderId = orderId; }
        then done;
    }

    action def doPickNextItem        { in itemId : String; }
    action def doMoveToPacking       { in itemId : String; }
    action def doConsolidatePickedItems { in orderId : String; }
}
```

## Error Patterns to Avoid

- **Using v1 `activity` keyword** — replaced by `action def` in v2
- **Missing in/out direction on parameters** — every action parameter must be `in`, `out`, or `inout`
- **Untyped parameters** — `in x;` is invalid; must include type and default
- **No `start` / `done`** — every top-level action def must have explicit `first start;` and `then done;`
- **Decision without `else`** — `decide` blocks must cover all guard cases or end with `else`
- **Reference before declaration** — referenced action defs must be declared in the same package before use, or imported
- **Mixing procedural `if` with action flow** — use `then decide; if ... then ... else ...` pattern, NOT a bare procedural `if` block
- **Forgetting `then` between sequenced actions** — actions are *not* implicitly sequenced; every step needs `then`
- **Invalid fork syntax** — use `then fork; branch <name> : <ActionDef> { ... }` then `then join;`, not v1 `fork {} also {} also {}`

## USS Bindings

- **Consumes:**
  - `requirements` — `id`, `text`, `priority`, `subject_type` (each action def should trace to ≥1 requirement)
- **Produces:**
  - `functions` — `name`, `description`, `input_params` (with type/default), `output_params`, `requirement_ids`, `parent_action`, `control_flow_kind` (sequential | parallel | loop | decision)
  - `artifacts` — persisted SysML v2 source, AST, validation report

## Learned Error Patterns

<!-- Error patterns recorded by skill_improver_tool during validation fixes.
     Each entry has a Pattern (what went wrong) and a Fix (what resolved it).
     These help future runs avoid repeating the same mistakes. -->


### Pattern
Action definitions with spaces in names caused syntax errors. Used quoted names like 'Receive Storage/Retrieval Order' which parser didn't recognize.

### Fix
Use valid identifiers without spaces/special characters for action def names. Example: `action def ReceiveStorageRetrievalOrder` instead of `action def 'Receive Storage/Retrieval Order'`


### Pattern
Used 'first start actionName' and 'then actionName then actionName' syntax which caused parsing errors.

### Fix
Use 'first actionName then actionName;' for each succession. Each action-to-action flow needs its own 'first ... then ...' statement. Example: 'first receiveOrder then validateOrder;' followed by 'first validateOrder then dispatchAGV;'


### Pattern
Used `then actionName : ActionDef` syntax for action usage in composite actions

### Fix
Use `action actionName : ActionDef;` to declare the action, then `then actionName;` for control flow. The colon syntax is only for type declarations, not usages.


### Pattern
Used `doc = "..."` syntax for requirement documentation

### Fix
Use multi-line doc comment syntax: `doc /* comment */` or `doc /*\n * comment\n */` for multi-line documentation in requirements.


### Pattern
Used action names that conflict with reserved keywords (analyze, detect, limit, activate, transition, feedback, done)

### Fix
Use simple action names without reserved keywords. Instead of `then analyze : DoAction`, use `then doAction;` and define separate action usages outside the action def with proper input bindings. Also, use single quotes for doc comments `/** text */` and single quotes for string values like `priority: 'must'`.


### Pattern
Used invalid action succession syntax with colon (then action : doActionDef) and reserved keywords as action names

### Fix
Use simple action succession syntax without colon (then doActionName;) and avoid reserved keywords like analyze, engage, limit, display, transition, feedback, signal, wait, receive, create, eval, check, restrict, activate, continue, notify, fork, join, decide, start, done, fork, join as action names


### Pattern
Used single quotes for multi-word identifiers, but SysML v2 requires single quotes only for names with spaces in use case definitions; for item/requirement/action defs, single quotes are valid but need proper require constraint syntax

### Fix
Use single quotes for names with spaces in item/requirement/action defs, and use `require constraint { expression }` syntax for requirement constraints. Avoid reserved keywords like 'transition' as variable names. Use `action 'ActionName' { ... }` for action calls within activities.


### Pattern
Used 'branch doActionName;' syntax inside fork/join - invalid in SysML v2

### Fix
Use 'then doActionName;' directly inside fork/join blocks without the 'branch' keyword


### Pattern
Used inline action usage with colon in decision branches (if actionName : ActionDef) - caused syntax errors

### Fix
Declare action usages separately before first/start, then use them in decision branches without colon: action verify : doVerify; first start; then verify; then decide; if verify.valid then recordAction; else rejectAction;


### Pattern
Used 'then actionName;' syntax which caused 'Missing expected token send' errors

### Fix
Use explicit 'first action1 then action2;' syntax for each succession. Example: 'first receiveGoods then storeGoods;' instead of 'then storeGoods;'


### Pattern
Using colon syntax for action usage inside composite action body (then actionName : ActionDef) causes syntax errors

### Fix
Use 'action actionName : ActionDef;' to declare the action usage, then 'then actionName;' for control flow. Do NOT use colon in the 'then' statement.
