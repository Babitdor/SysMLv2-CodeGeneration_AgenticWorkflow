---
name: internal-block-diagram
description: Generate SysML v2 Internal Block Diagrams (Interconnection Views with port defs, interfaces, item flows, conjugate ports) from USS logical_blocks. Use when the user wants to wire instantiated parts via ports and quantify item exchange between them.
license: proprietary
---

## Purpose

An Internal Block Diagram (IBD) — called the **Interconnection View** in SysML v2 — opens the "white box" of a containing part and shows how its nested part *usages* are wired together via ports, connections, interfaces, and item flows. The BDD declares the *types*; the IBD instantiates them and connects them. Item flows on the connections specify what matter, energy, or data is exchanged at runtime.

## Syntax Rules

- IBD content lives **inside a `part def` body** — the part def is the encapsulating context
- Use `part <name> : <PartDef>;` for nested part usages (NOT `part def` — that would create a new type)
- Use `port def <Name> { in/out/inout item <itemName> : <ItemDef>; }` to declare port types in BDD scope
- Port direction modifiers (`in`, `out`, `inout`) go on the *items* inside the port def, NOT on the port itself
- Conjugate a port with `~` to flip directions: `port reply : ~RequestPort;` makes outs into ins
- Use `interface def <Name> { end <p1> : <Port>; end <p2> : ~<Port>; flow ... }` for reusable connection types
- Connect ports with: `connect <part1>.<port> to <part2>.<port>;` or via interface usage
- Item flows on connections: `flow <item> from <src>.<port>.<feature> to <dst>.<port>.<feature>;`
- Use `item def <Name>` to declare what flows (Pallet, Signal, Energy, etc.)
- Every connected pair of ports MUST have compatible (or conjugate-compatible) types
- Connection multiplicities: `connect [1..*] <p1> to <p2>;` for one-to-many wiring
- NEVER connect to a port that wasn't declared on the part def's boundary

## Few-Shot Examples

### Example 1: Two Parts Connected via Conjugate Ports
**Source:** FAPS-archive/ibd-v1.sysml
```sysml
package WarehouseInterconnections {
    public import ScalarValues::*;
    public import ISQ::MassValue;

    item def Pallet {
        attribute weight : MassValue = 25.0 [kg];
    }

    port def MaterialFlowPort {
        out item shippedPallet : Pallet;
        in  item receivedPallet : Pallet;
    }

    part def Conveyor {
        port outlet : MaterialFlowPort;
    }

    part def StorageRack {
        port inlet : ~MaterialFlowPort;
    }

    part def StorageZoneAssembly {
        part conveyor1 : Conveyor;
        part rack1     : StorageRack;

        connect conveyor1.outlet to rack1.inlet;
    }
}
```

### Example 2: Interface Definition with Item Flow
**Source:** FAPS-archive/ibd-interface.sysml
```sysml
package WarehouseInterconnections {
    public import ScalarValues::*;

    item def OrderRequest  { attribute orderId : String; }
    item def OrderResponse { attribute confirmed : Boolean = false; }

    port def OrderRequestPort {
        out item request  : OrderRequest;
        in  item response : OrderResponse;
    }

    interface def OrderInterface {
        end requester : OrderRequestPort;
        end servicer  : ~OrderRequestPort;

        flow of OrderRequest  from requester.request  to servicer.request;
        flow of OrderResponse from servicer.response  to requester.response;
    }

    part def WarehouseManagementSystem {
        port wmsPort : OrderRequestPort;
    }

    part def Forklift {
        port forkliftPort : ~OrderRequestPort;
    }

    part def WarehouseControlContext {
        part wms      : WarehouseManagementSystem;
        part forklift : Forklift;

        interface : OrderInterface
            connect wms.wmsPort
                 to forklift.forkliftPort;
    }
}
```

### Example 3: Multiplicity and Multiple Item Flows
**Source:** FAPS-archive/ibd-multiplicity.sysml
```sysml
package WarehouseInterconnections {
    public import ISQ::PowerValue;
    public import SI::W;

    item def ElectricalPower { attribute power : PowerValue = 0.0 [W]; }
    item def ControlSignal   { attribute command : String; }

    port def ChargerPort {
        out item supplied : ElectricalPower;
        in  item ack      : ControlSignal;
    }

    part def ChargingStation {
        port supply [1..*] : ChargerPort;
    }

    part def Forklift {
        port battery : ~ChargerPort;
    }

    part def ChargingZoneAssembly {
        part station   : ChargingStation;
        part forklifts : Forklift[1..4];

        connect station.supply to forklifts.battery;
    }
}
```

## Error Patterns to Avoid

- **Using `part def` for nested parts** — that creates a new type; use `part <name> : <Def>` for usages
- **Direction modifier on the port** — `port out p : T;` is wrong; direction is on the items inside the port def
- **Forgetting `~` for the receiving end** — both endpoints with the same orientation cannot connect; conjugate one with `~`
- **Connecting undeclared ports** — every port used in `connect` must be declared on the boundary of its part def
- **Item flows without item defs** — flows reference `item def`s; declare them in scope before use
- **Using v1 stereotypes** — no `<<flowPort>>`, `<<itemFlow>>` stereotypes in v2; use `port def` and `flow` directly
- **Cross-context connections** — you can only connect parts that are siblings inside the same enclosing part def
- **Multiplicity mismatch** — connecting `port[1]` to `port[*]` requires explicit multiplicity on the connection itself
- **Reusing a port name across part defs without import** — port defs must be imported or fully qualified

## USS Bindings

- **Consumes:**
  - `logical_blocks` — `id`, `name`, `ports` (with item types), `parent_block_id`
  - `functions` — `inputs`, `outputs` (drives which item flows are needed between blocks)
- **Produces:**
  - `interconnections` — `source_block_id`, `source_port`, `target_block_id`, `target_port`, `item_type`, `multiplicity`, `interface_def_id`
  - `artifacts` — persisted SysML v2 source, AST, validation report

## Learned Error Patterns

<!-- Error patterns recorded by skill_improver_tool during validation fixes.
     Each entry has a Pattern (what went wrong) and a Fix (what resolved it).
     These help future runs avoid repeating the same mistakes. -->


### Pattern
Used incorrect syntax for connection with interface definition (connect name : InterfaceDef { ... })

### Fix
Use simple direct connection syntax without interface definition: `connect part1.port to part2.port;` The interface definition pattern requires different syntax not supported in basic connections.


### Pattern
Used incorrect connection syntax with interface type: `connect part1.port to part2.port : InterfaceType;`

### Fix
Use `interface <name> : <InterfaceType> connect <port1> to <port2>;` syntax for connections with interface definitions


### Pattern
Used reserved keyword 'message' as attribute name

### Fix
Renamed attribute from 'message' to 'feedbackMessage' - avoid reserved keywords in SysML v2


### Pattern
Incorrect connection syntax with ':> DiagnosticsInterface' after connection statement

### Fix
Removed the type annotation from connection statement - simple connections don't need type annotations: `connect a.port to b.port;`


### Pattern
Using connection usage with nested end part syntax causes syntax errors with dot notation for port references

### Fix
Use interface usage syntax instead: `interface <name> : <InterfaceType> connect <part1>.<port> to <part2>.<port>;` This correctly handles dot notation for referencing nested ports in parts.


### Pattern
Duplicate item names in port definitions (e.g., "in item item : Type")

### Fix
Use unique item names in port definitions: "in item inPallet : EuroPallet;" instead of "in item item : EuroPallet;" to avoid duplicate definition errors


### Pattern
Using ISQ types (MassValue, TimeValue) directly in item definitions causes semantic warnings

### Fix
Use Real type with SI units instead: attribute maxWeight : Real = 50.0 [kg]; attribute processingTime : Real = 60.0 [s];


### Pattern
Interface definition with flow statements causing unresolved reference errors

### Fix
Removed flow statements from interface definitions - interface defs should only declare end ports without flow statements
