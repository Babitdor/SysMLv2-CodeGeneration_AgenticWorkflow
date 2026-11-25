---
name: state-machine-diagram
description: Generate SysML v2 State Machine Diagrams — state definitions with entry/exit/do actions, transitions with accept/guard/effect triples, hierarchical (nested) states, and `exhibit state` linking a part to its state machine. Use when modelling modes, lifecycle phases, fault behaviour, control logic, or any system that has discrete configurations with event-driven changes.
license: proprietary
---

## Purpose

A State Machine Diagram declares the **modes** a system or part can occupy, the **transitions** between them, and the **events / guards / effects** that drive transitions. SysML v2 builds the state machine directly into the language with `state def`, `state` usages, `transition`, and the `exhibit` relationship that binds a part to a state machine.

State machines answer: "What discrete configurations is this system in over time, and what makes it change?" — complementing structural diagrams (BDD/IBD) and action-flow diagrams (Activity).

## Syntax Rules

- Use `state def <Name> { … }` for a reusable state machine definition; use bare `state <name>` for a usage inside another state or `exhibit state` block
- A state body may declare `entry action <act>`, `do action <act>`, `exit action <act>` — each is optional, each takes an `action def` reference or an inline action
- Use `exhibit state <name> { state <s1>; state <s2>; transition … }` inside a `part def` body to attach a state machine to that part
- Transitions ALWAYS use `first <source> then <target>` as the source/target keywords — `first` and `then` are mandatory
- Full transition shape: `transition <name> first <src> accept <evt> : <EventType> if <guard> do action <effect> then <tgt>;`
- Hierarchical (composite) states nest naturally: `state running { state accelerating; state cruising; }` — substates inherit their parent's `do` action and exit-from-parent transitions
- Multiplicity / re-entry: a transition into a composite state lands on its initial substate; explicit history is modelled with a designated `history` pseudo-state when needed
- Imports MUST have visibility: `public import ScalarValues::*;` and (for typed events) the event-type package
- Events come from incoming signals (`accept signal : SignalType`), incoming items, time, or attribute changes (`accept e: ChangeOf(attr)`)
- NEVER use v1 keywords: `StateMachine`, `Region`, `Pseudostate`, `«initial»`, `«final»`, `«history»` — v2 uses keyword forms (`first`, `then`, plain `state` declarations)
- NEVER use `else if` chains inside transitions — model each branch as a separate transition with mutually exclusive guards
- NEVER reuse reserved names: `state`, `transition`, `accept`, `entry`, `exit`, `do`, `first`, `then` cannot be used as identifiers

## Few-Shot Examples

### Example 1: Vehicle Power-State Machine

```sysml
package VehiclePowerStates {
    public import ScalarValues::*;
    public import ISQ::*;

    action def TurnOnLights;
    action def TurnOffLights;
    action def MonitorSensors;

    state def VehicleState {
        entry action TurnOnLights;
        do    action MonitorSensors;
        exit  action TurnOffLights;
    }

    part def Vehicle {
        exhibit state vehicleStates {
            state off;
            state on;
            state error;

            transition powerOn  first off   then on;
            transition powerOff first on    then off;
            transition fault    first on    then error;
            transition recover  first error then off;
        }
    }
}
```

### Example 2: Hierarchical Operational Mode with Event + Guard

```sysml
package OperationalModes {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    item def StartCmd;
    item def StopCmd;
    item def TempSensorReading {
        attribute temperature : TemperatureValue;
    }
    action def CooldownProtocol;

    state def Operational {
        state idle;
        state running {
            state accelerating;
            state cruising;
            state braking;

            transition  first accelerating then cruising;
            transition  first cruising     then braking;
        }
        state degraded;

        transition start
            first idle
            accept signal : StartCmd
            then running;

        transition stop
            first running
            accept signal : StopCmd
            then idle;

        // Guarded + effected transition out of the composite state.
        transition overheat
            first running
            accept e : TempSensorReading
            if e.temperature > 100.0 [°C]
            do action CooldownProtocol
            then degraded;
    }
}
```

### Example 3: Flashlight — Minimal Self-Contained State Machine

```sysml
package FlashlightControl {
    public import ScalarValues::*;

    item def PressEvent;

    part def Flashlight {
        exhibit state mode {
            state off;
            state on;

            transition pressOn  first off accept p : PressEvent then on;
            transition pressOff first on  accept p : PressEvent then off;
        }
    }
}
```

## Common Pitfalls

- **Omitting `first` / `then`** → validator error `Missing expected token 'first'` — both keywords are mandatory on every transition
- **`else if` inside a transition** → v2 has no nested-conditional transition form; model each guard as its own transition out of the same source state, with mutually exclusive `if` clauses
- **Naming a state with a reserved word** (`state`, `transition`, `initial`, `final`) → rename to a non-reserved identifier; the v2 keyword list at `language/02-textual-notation-syntax.md:348` is the authoritative list
- **Forgetting `exhibit`** when attaching a state machine to a part → the part has no link to the state def; always wrap `state def` consumption in `exhibit state <name> :> <StateDef> { … }` or inline the usage
- **Guards that reference variables outside the event payload + state context** → the guard scope is the accepted event and the state's owning part; navigate through `e.<field>` for event-carried data
- **Treating substates as independent of their composite parent** → a substate transitioning out triggers the parent's exit; a transition into a composite lands on the composite's initial substate (declared by inline initial transition or by being first declared)

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "The system is either OFF, ON, or ERROR." | Flat state machine with three `state` usages + transitions |
| "When operational, it cycles through accelerating → cruising → braking." | Hierarchical `state def Operational` with nested `state running { … }` |
| "It reacts to a PressEvent by toggling power." | Transition with `accept p : PressEvent` |
| "Above 100°C it goes to degraded mode and runs cooldown." | Transition with guard (`if e.temperature > 100.0 [°C]`) and effect (`do action CooldownProtocol`) |
| "Bind this state machine to the Vehicle part." | `part def Vehicle { exhibit state vehicleStates { … } }` |
