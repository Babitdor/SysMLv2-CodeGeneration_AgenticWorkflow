---
name: analysis-case-diagram
description: 'Generate SysML v2 Analysis Case Diagrams — `analysis case def` with a subject, one or more typed `objective` declarations, and a `return` expression computing each objective, plus concrete usages that bind the analysis to a specific subject. Use when the user wants to compute model-derived properties (mass rollups, performance metrics, cost estimates, range vs. weight trade-offs) — the formal v2 counterpart of running a calculation against the model.'
license: proprietary
---

## Purpose

An Analysis Case Diagram declares **a calculation that runs against the model** to derive a property or trade-off. It is a kind of `action def` specialised for analysis: it has a `subject` (what is being analysed), one or more `objective` declarations (the quantities to compute), and a `return` expression that produces each objective's value from the subject's features. Analysis cases compose — one analysis can call another, building chained analyses (mass → centre-of-gravity → handling characteristics, etc.).

Where `verification case def` says "prove this requirement holds", `analysis case def` says "compute this property". Both share the subject + objective shape but differ in whether the result is Boolean (verification) or a typed value (analysis).

## Syntax Rules

- Use `analysis case def <Name> { … }` — the keyword is two words (`analysis case`), not `analysisCase`
- Required elements inside the body:
  - `subject <name> : <SubjectPart>;` — exactly ONE subject
  - At least one `objective <name> : <ResultType>;` — the quantities being computed
- Recommended elements:
  - `doc /* what the analysis computes and why */`
  - One `return <objective> = <expression>;` clause per objective — the expression typically navigates the subject's features
- For a concrete usage: `analysis case <name> : <AnalysisCaseDef> { subject = <concretePart>; }`
- Aggregations over multi-cardinality parts use standard collection operations: `sum(veh.allParts.mass)`, `count(veh.passengers)`, `max(wheels->collect(w | w.pressure))`
- Imports MUST have visibility: `public import ScalarValues::*;` `public import ISQ::*;` `public import SI::*;` (almost always needed for typed objectives)
- Multiple objectives in one analysis case is the COMMON case — performance analyses typically compute top speed, acceleration, and fuel economy together
- Optional `actor` and `stakeholder` declarations attach external participants and interested parties (e.g. `stakeholder finance;`)
- Composition: an analysis case may CALL another analysis case via `perform` or by directly referencing the other case's objective in a `return` expression
- NEVER use v1 keywords: `«analysis»`, `«evaluates»`, `Calculation` block — none are v2 keywords
- NEVER conflate objectives with `attribute`s — an `objective` is a typed result of the analysis, not a stored attribute of the subject

## Few-Shot Examples

### Example 1: Mass Roll-up (Single Objective)

```sysml
package MassAnalysis {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;
    public import VehicleModel::*;

    analysis case def MassRollup {
        doc /* Computes total vehicle mass by summing the mass of every owned part. */

        subject veh : Vehicle;
        objective totalMass : MassValue;

        return totalMass = sum(veh.allParts.mass);
    }

    // Concrete usage — runs the analysis against a specific vehicle instance.
    analysis case myCarMass : MassRollup {
        subject = myCar;
    }
}
```

### Example 2: Multi-Objective Performance Analysis

```sysml
package PerformanceAnalysis {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;
    public import VehicleModel::*;

    analysis case def Performance {
        doc /* Powertrain analysis — three objectives derived from the vehicle's design. */

        subject veh : Vehicle;

        objective topSpeed       : SpeedValue;
        objective acceleration0to60 : TimeValue;
        objective fuelEconomy    : FuelEfficiencyValue;

        // Each return clause computes one objective.
        return topSpeed          = veh.powertrain.peakPower / (veh.mass * 0.5);
        return acceleration0to60 = (60.0 [km/h]) / (veh.powertrain.peakPower / veh.mass);
        return fuelEconomy       = veh.powertrain.efficiency * veh.fuelTank.capacity;
    }
}
```

### Example 3: Stakeholder-Driven Cost Estimate

```sysml
package CostAnalysis {
    public import ScalarValues::*;
    public import ISQ::*;
    public import ProjectModel::*;

    analysis case def CostEstimate {
        doc /* Aggregates per-component cost into a project-level total. */

        subject  proj : Project;
        stakeholder finance;
        stakeholder programManager;

        objective totalCost   : MoneyValue;
        objective contingency : MoneyValue;

        return totalCost   = sum(proj.allComponents.unitCost);
        return contingency = totalCost * 0.15;        // 15 % buffer
    }
}
```

## Common Pitfalls

- **Writing `analysisCase def`** (single token) → validator error `Unexpected token` — the keyword is `analysis case` (two tokens, space-separated)
- **Objective without a type** → `objective totalMass;` is invalid; every objective must declare a result type: `objective totalMass : MassValue;`
- **`return` clause naming an undeclared objective** → the `return <name> = …;` clause must name an objective declared earlier in the same body
- **Treating objectives like attributes** → objectives are NOT stored on the subject; they are computed by the analysis case and surfaced via the `return` clauses
- **Forgetting to import quantity types** → `objective topSpeed : SpeedValue;` fails to resolve `SpeedValue` without `public import ISQ::*;`
- **Aggregating without using the right collection operator** — `sum(parts.mass)` requires `parts` to be a multi-cardinality feature whose elements have a `mass` attribute. For nested traversal use `allParts` (when defined) or chain navigations explicitly

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "What is the total mass of the vehicle?" | Single-objective `analysis case def MassRollup` with `return totalMass = sum(veh.allParts.mass)` |
| "Compute top speed, 0-60 time, and fuel economy together." | Multi-objective `analysis case def Performance` with one `return` per objective |
| "Estimate project cost for finance and programme management." | `analysis case def CostEstimate` with two `stakeholder`s and two objectives (total + contingency) |
| "Trade-off between range and battery weight for an EV." | Multi-objective analysis case; the trade-off is the relationship between the two `return` expressions |
| "Run this analysis on a specific design variant." | Usage: `analysis case caseName : AnalysisCaseDef { subject = <variantPart>; }` |
| "I need the analysis to call another analysis." | Reference the other case's objective in a `return` expression, or `perform` it explicitly |
