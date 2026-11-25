---
name: parametric-diagram
description: Generate SysML v2 Parametric Diagrams — formal constraint definitions with named parameters bound to part attributes via `assert constraint` usages. Use when the user wants to express mathematical relations, performance budgets, safety envelopes, or any quantitative invariant across parts of the system. Replaces SysML v1's "Constraint Block + Binding Connector" pattern with parameter binding via `=` in the assertion's argument list.
license: proprietary
---

## Purpose

A Parametric Diagram declares **constraint definitions** (Boolean expressions with typed parameters) and **constraint usages** that bind those parameters to concrete attributes elsewhere in the model. The point of the diagram is to make quantitative relationships first-class and machine-checkable — once a constraint is asserted with bound arguments, a solver can evaluate it against any instance of the subject part.

In SysML v1 this was modeled as a "Constraint Block" surrounded by a Parametric Diagram with «equal» Binding Connectors. SysML v2 collapses this into the **`constraint def`** + **`assert constraint`** pair, with binding expressed as positional or named arguments to the assertion.

## Syntax Rules

- Use `constraint def <Name> { in <param> : <Type>; <Boolean expression> }` — parameters are declared with `in` and have a concrete type
- The constraint body is a **single Boolean expression** — no statements, no side effects, no `;` between sub-expressions inside the body
- Use `assert constraint <ConstraintDef>(arg1 = …, arg2 = …)` — named arguments bind formal parameters to actual attributes
- Positional binding also works: `assert constraint <ConstraintDef>(actual1, actual2)` — first formal in declaration order takes `actual1`, etc.
- Anonymous inline constraints are legal: `constraint { vehicle.mass <= 2500 [kg] }` — useful when a one-off constraint doesn't need a name
- All numeric expressions in physical constraints MUST carry units in square brackets: `12.5 [m]`, `2500 [kg]`, `60 [km/h]`
- Imports MUST have visibility: `public import ScalarValues::*;` `public import ISQ::*;` `public import SI::*;`
- Forall / exists quantifiers: `forall p in collection { boolean expr }` / `exists p in collection { boolean expr }`
- Implication: `(condition) implies (consequence)` — both sides must be Boolean
- NEVER use v1 keywords: `constraintBlock`, `«equal»`, `BindingConnector`, `nestedConnectorEnd`
- NEVER nest multiple statements in a constraint body — split into sub-constraints if you need conjunctions of bound expressions
- Asserting a constraint inside a `part def` body scopes the bindings to the owning part's features

## Few-Shot Examples

### Example 1: Mass-Budget Constraint with Parametric Binding

```sysml
package MassBudget {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    constraint def WithinMassBudget {
        in actual  : MassValue;
        in maxMass : MassValue;
        actual <= maxMass
    }

    part def Vehicle {
        attribute mass        : MassValue;
        attribute massLimit   : MassValue = 2500.0 [kg];

        // Bind the formal parameters to local attributes by name.
        assert constraint WithinMassBudget(
            actual  = mass,
            maxMass = massLimit
        );
    }
}
```

### Example 2: Power-Balance Constraint Across Sub-Parts

```sysml
package PowerBalance {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    constraint def PowerBalanced {
        in supplied  : PowerValue;
        in consumed  : PowerValue;
        in headroom  : PowerValue;
        supplied >= consumed + headroom
    }

    part def Battery   { attribute outputPower : PowerValue; }
    part def Motor     { attribute drawPower   : PowerValue; }
    part def Vehicle {
        attribute reservePower : PowerValue = 250.0 [W];

        part battery : Battery;
        part motor   : Motor;

        // Positional binding — formals consumed in declaration order.
        assert constraint PowerBalanced(
            battery.outputPower,
            motor.drawPower,
            reservePower
        );
    }
}
```

### Example 3: Inline / Anonymous Constraint with Quantifier

```sysml
package SafetyEnvelope {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    part def Wheel  { attribute pressure : PressureValue; }
    part def Vehicle {
        part wheels : Wheel[4];

        // Anonymous inline — no name, no parameters, scoped to this part.
        // Use `forall` over the multi-cardinality part to assert per-wheel.
        assert constraint {
            forall w in wheels { w.pressure >= 2.0 [bar] and w.pressure <= 2.8 [bar] }
        }
    }
}
```

## Common Pitfalls

- **Forgetting the `in` on parameters** → validator error `Unexpected parameter declaration` — every constraint-def parameter must use the `in` keyword
- **Multiple Boolean statements in the body** → split them. `constraint def { in x; x > 0; x < 10 }` is invalid. Use `x > 0 and x < 10` instead
- **Missing units on physical quantities** → validator passes but the solver cannot dimensional-check; use `[kg]`, `[m]`, `[s]`, `[W]` etc.
- **Using `=` between bound parameters and target expressions in named binding** is required — `WithinMassBudget(actual = mass)` not `WithinMassBudget(actual mass)`
- **Reaching outside the owning part** in an asserted constraint — the assertion sees features of the part it's declared in (and inherited features). Reaching deeper into nested parts via dot navigation works (`battery.outputPower`); reaching siblings via name lookup does not

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "Total mass must stay under 2500 kg." | `constraint def WithinMassBudget` + assertion on Vehicle |
| "Power supply must cover demand plus reserve." | Three-parameter constraint def + assertion binding three attributes |
| "All four wheel pressures must be in range." | Anonymous inline with `forall` over the multi-cardinality part |
| "If headlights are on, daytime brightness must be low." | Anonymous inline with `implies` |
| "Drag force formula relating velocity, area, and Cd." | `constraint def` over derived expression — assert at the level where all three feeders are visible |
