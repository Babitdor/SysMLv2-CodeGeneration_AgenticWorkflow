---
name: sequence-diagram
description: Generate SysML v2 "sequence diagrams" — temporally ordered interactions between parts modelled via `succession`, `flow`, and ordered `action` decomposition. CRITICAL — SysML v2 has NO dedicated `interaction` / `message` / `lifeline` keyword (verified against the v2 keyword catalogue). The v1 UML-style sequence diagram is replaced by sequenced actions, item flows over connections, and `succession flow` for timed transfers. Use when the user wants to express "X happens, then Y, then Z" or "A sends a command to B, then B replies".
license: proprietary
---

## Purpose

A "sequence diagram" in SysML v2 captures **what happens, in what order, between which participants**. Unlike SysML v1 (which adopted UML's interaction frame + lifeline + message-arrow notation), v2 has no separate `interaction` construct — temporal ordering is expressed inside the existing action / flow / state vocabulary:

- **`succession`** sequences two behaviours / actions / states in time
- **`flow`** transfers an item from a source port to a target port
- **`succession flow`** combines them: an item transfer that also imposes a temporal precedence
- **Ordered sub-actions** inside an `action def` body (`first … then … then …`) express a totally ordered chain
- **`accept`** in an action body models receiving an event/signal — the v2 counterpart of UML's "incoming message"

When a user asks for a "sequence diagram", they usually want one of these patterns; the skill maps each common intent to the right v2 construct.

## Syntax Rules

- **No `interaction` / `lifeline` / `message` keyword exists in SysML v2.** Do NOT emit them — the parser will reject them as unknown identifiers
- Use `action def <Name>` for behaviours that can be sequenced. The action's sub-actions (`action sub1; action sub2;`) are ordered by `first sub1 then sub2;` clauses
- Use `succession <name> first <src> then <tgt>;` to impose explicit temporal precedence between two arbitrary actions / states
- Use `flow <name> from <src.port> to <tgt.port>;` to transfer an item between connected parts (NOT temporal on its own — pair with a succession when timing matters)
- Use `succession flow <name> from <src.port> to <tgt.port>;` for "after the source emits, the target receives, and the second action waits for it"
- For receive-side modelling, use `accept signal : SignalType` inside an action body — the action waits for a matching incoming signal before proceeding
- Items being exchanged must be declared as `item def <Name>`; ports must carry the matching item type (`in`/`out` direction)
- All numeric quantities carry units in square brackets: `5.0 [s]`, `100 [m]`, `12.0 [kg]`
- Imports MUST have visibility: `public import ScalarValues::*;` and any item / signal packages used
- NEVER use `interaction def`, `interaction`, `lifeline`, `message`, `«message»`, `«send»`, `«receive»` — none are v2 keywords
- NEVER use UML-style arrow notation (`->`, `-->`) in a `.sysml` file — sequencing is purely textual via `first`/`then`/`succession`

## Few-Shot Examples

### Example 1: Two-Party Command-Response Sequence

```sysml
package CommandResponse {
    public import ScalarValues::*;

    item def StartCmd;
    item def Ack;

    port def CmdOutPort  { out cmd : StartCmd; }
    port def CmdInPort   :> ~CmdOutPort;
    port def AckOutPort  { out ack : Ack; }
    port def AckInPort   :> ~AckOutPort;

    action def IssueCommand;
    action def ReceiveCommand {
        accept signal : StartCmd;
    }
    action def SendAcknowledgement;
    action def ReceiveAcknowledgement {
        accept signal : Ack;
    }

    part def Controller {
        port cmdOut : CmdOutPort;
        port ackIn  : AckInPort;

        perform action sendCmd      : IssueCommand;
        perform action awaitAck     : ReceiveAcknowledgement;

        // Temporal order on the controller side.
        first sendCmd then awaitAck;
    }

    part def Actuator {
        port cmdIn  : CmdInPort;
        port ackOut : AckOutPort;

        perform action receive : ReceiveCommand;
        perform action reply   : SendAcknowledgement;

        first receive then reply;
    }

    part def System {
        part ctrl : Controller;
        part act  : Actuator;

        // Cross-party item transfers — the succession flows tie sender → receiver
        // AND impose temporal precedence between the two actions.
        succession flow cmdTransfer from ctrl.cmdOut to act.cmdIn;
        succession flow ackTransfer from act.ackOut  to ctrl.ackIn;
    }
}
```

### Example 2: Three-Step Pipelined Sequence Inside One Part

```sysml
package ImagePipeline {
    public import ScalarValues::*;

    action def CaptureFrame;
    action def DetectObstacles;
    action def UpdatePathPlan;

    part def AutopilotControl {
        perform action capture : CaptureFrame;
        perform action detect  : DetectObstacles;
        perform action plan    : UpdatePathPlan;

        // Total order of the pipeline — equivalent to a 3-step sequence diagram
        // with a single lifeline in UML terms.
        first capture then detect then plan;
    }
}
```

### Example 3: Timed Producer-Consumer with Item Flow

```sysml
package ProducerConsumer {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    item def Sample {
        attribute value : Real;
    }

    port def SampleOutPort { out sample : Sample; }
    port def SampleInPort  :> ~SampleOutPort;

    part def Sensor {
        port out_ : SampleOutPort;
        perform action measure;
    }
    part def Logger {
        port in_ : SampleInPort;
        perform action record;
    }

    part def Probe {
        part sensor : Sensor;
        part logger : Logger;

        // Sample flows from sensor → logger; "record" can only happen
        // AFTER the corresponding "measure" produces a sample.
        succession flow stream from sensor.out_ to logger.in_;
    }
}
```

## Common Pitfalls

- **Writing `interaction def Foo { … }`** → validator error `Unexpected token 'interaction'` — the keyword does not exist in v2. Replace with `action def Foo { … }` or model the participants explicitly with `part def`s and `succession flow`s
- **Using `lifeline` or `message`** → unknown identifiers; SysML v2 expresses participation through `part def`s and message exchange through item flows
- **Trying to draw arrows (`->`)** in textual notation → not part of v2 grammar; use `succession`, `flow`, or `first … then …` for ordering
- **Omitting `first` in an action body that needs sequencing** → sub-actions inside an action body are NOT implicitly ordered. If order matters, add `first <a> then <b>;`
- **Receiving without `accept`** → if a participant waits for a signal, the receiving action MUST contain `accept signal : SignalType` inside its body — otherwise it's just a free-running action with no event-driven trigger
- **Confusing `flow` with `succession flow`** → bare `flow` transfers items but imposes no temporal order on its own. `succession flow` adds the temporal precedence

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "A sends a command, B receives, B sends a reply." | Two `part def`s with `succession flow`s in both directions + `accept` in receivers |
| "Pipeline: capture → detect → plan, all on one component." | Single `part def` with three `perform action`s linked by `first … then … then …` |
| "Periodic samples from sensor to logger." | `succession flow` between sensor's out port and logger's in port |
| "A timed sequence of operations across multiple parts." | One `succession flow` per inter-part transfer plus per-part `first/then` for intra-part ordering |
| "User clicks button → system shows dialog → user types → system saves." | Model as actions and successions in a `use case` (see use-case-diagram skill) — for fine-grained order, expand into per-step actions inside the use case body |
| "I want a UML-style sequence diagram with lifelines and arrows." | **Reject the framing**. Explain v2 has no lifeline/message construct; offer to model the same intent with parts + succession flows |
