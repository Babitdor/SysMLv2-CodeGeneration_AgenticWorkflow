## Part Definition and Usage

### Error: Incorrect Part Definition Syntax
**Category:** Part Definition and Usage  
**Severity:** High  
**Error Code:** PD001

**Problem Description:**
Using SysML v1 block syntax instead of SysML v2 part definition syntax.

**Incorrect Code:**
```sysml
<<block>> Engine {
    values:
        power: Real [kW]
}
```

**Correct Solution:**
```sysml
part def Engine {
    attribute power : ScalarValues::Real;
    attribute weight : MassValue;
}

part engine : Engine;
```

**Key Points:**
- Use `part def` instead of `<<block>>`
- Attributes use `:` separator, not `values:` section
- Include proper type imports

---

### Error: Missing Namespace Imports
**Category:** Part Definition and Usage  
**Severity:** Medium  
**Error Code:** PD002

**Problem Description:**
Using types without proper import statements causes compilation errors.

**Incorrect Code:**
```sysml
part def Battery {
    attribute voltage : Real;
}
```

**Correct Solution:**
```sysml
import ScalarValues::*;

part def Battery {
    attribute voltage : Real;
    attribute capacity : Real;
}
```

**Key Points:**
- Always import required type libraries
- Use `import ScalarValues::*` for basic types
- Import specific types when needed

---

### Error: Incorrect Multiplicity Syntax
**Category:** Part Definition and Usage  
**Severity:** Medium  
**Error Code:** PD003

**Problem Description:**
Using SysML v1 multiplicity notation instead of v2 syntax.

**Incorrect Code:**
```sysml
part wheels : Wheel[4];  // Actually correct in v2
```

**Correct Solutions:**
```sysml
part wheels : Wheel[4];
// OR with explicit multiplicity
part wheels[4] : Wheel;
```

**Key Points:**
- SysML v2 supports both syntactic forms
- Choose consistent style throughout model
- Use explicit multiplicity for clarity

---

## Action Definition

### Error: Missing Action Definition Structure
**Category:** Action Definition  
**Severity:** High  
**Error Code:** AD001

**Problem Description:**
Defining actions without proper input/output parameter structure.

**Incorrect Code:**
```sysml
action StartEngine;
```

**Correct Solution:**
```sysml
action def StartEngine {
    in ignitionSignal : Boolean;
    out engineRunning : Boolean;
    out engineRPM : Real;
}

action startEngine : StartEngine;
```

**Key Points:**
- Define action signatures with `in` and `out` parameters
- Separate definition from usage
- Use descriptive parameter names

---

### Error: Incorrect Flow Connections
**Category:** Action Definition  
**Severity:** High  
**Error Code:** AD002

**Problem Description:**
Using old-style flow connections instead of SysML v2 interface-based connections.

**Incorrect Code:**
```sysml
engine.powerOut --> transmission.powerIn;
```

**Correct Solution:**
```sysml
interface def PowerInterface {
    attribute power : Real;
    attribute torque : Real;
}

connection powerFlow : PowerInterface connect 
    engine.powerOut to transmission.powerIn;
```

**Key Points:**
- Define interfaces for connections
- Use `connection` keyword with proper syntax
- Specify connection types explicitly

---

### Error: Missing Succession Relationships
**Category:** Action Definition  
**Severity:** Medium  
**Error Code:** AD003

**Problem Description:**
Actions without proper temporal sequencing relationships.

**Incorrect Code:**
```sysml
action checkOil : CheckOil;
action startEngine : StartEngine;
```

**Correct Solution:**
```sysml
action checkOil : CheckOil;
action startEngine : StartEngine;

succession checkOil then startEngine;
```

**Key Points:**
- Use `succession` for temporal ordering
- Define clear action sequences
- Consider parallel vs. sequential execution

---

## State Definition

### Error: Incorrect State Definition Syntax
**Category:** State Definition  
**Severity:** High  
**Error Code:** SD001

**Problem Description:**
Using old UML/SysML v1 state syntax instead of SysML v2 format.

**Incorrect Code:**
```sysml
state Running {
    entry / turnOnLED();
}
```

**Correct Solution:**
```sysml
state def RunningState {
    entry action turnOnLED;
    exit action turnOffLED;
    do action maintainOperation;
}

state running : RunningState;
```

**Key Points:**
- Use `state def` for definitions
- Use `action` keyword for behaviors
- Separate definition from instantiation

---

### Error: Missing State Machine Context
**Category:** State Definition  
**Severity:** High  
**Error Code:** SD002

**Problem Description:**
States defined without proper state machine context.

**Incorrect Code:**
```sysml
state idle : IdleState;
state running : RunningState;
```

**Correct Solution:**
```sysml
part def Engine {
    state machine {
        state idle : IdleState;
        state running : RunningState;
        
        transition idle to running
            accept StartSignal;
        
        transition running to idle
            accept StopSignal;
    }
}
```

**Key Points:**
- Embed states within `state machine` blocks
- Define transitions between states
- Use `accept` for event triggers

---

### Error: Incorrect Transition Triggers
**Category:** State Definition  
**Severity:** Medium  
**Error Code:** SD003

**Problem Description:**
Using old-style event triggers instead of proper SysML v2 syntax.

**Incorrect Code:**
```sysml
idle -[startPressed]-> running;
```

**Correct Solution:**
```sysml
transition idle to running
    accept startPressed : StartEvent;

transition running to idle
    accept when (temperature > maxTemp);
```

**Key Points:**
- Use `transition` keyword with `accept`
- Support both event and guard conditions
- Use proper event typing

---

## Constraint Definition

### Error: Missing Constraint Definition
**Category:** Constraint Definition  
**Severity:** High  
**Error Code:** CD001

**Problem Description:**
Using constraints without proper definition structure.

**Incorrect Code:**
```sysml
constraint massBalance;
```

**Correct Solution:**
```sysml
constraint def MassBalance {
    attribute totalMass : MassValue;
    attribute componentMasses : MassValue[*];
    
    inv { totalMass == componentMasses->sum() }
}

constraint massBalance : MassBalance;
```

**Key Points:**
- Define constraint structure first
- Use `inv` for invariant conditions
- Specify constraint attributes clearly

---

### Error: Incorrect Constraint Expression Syntax
**Category:** Constraint Definition  
**Severity:** High  
**Error Code:** CD002

**Problem Description:**
Using mathematical expressions without proper KerML syntax.

**Incorrect Code:**
```sysml
constraint { power = voltage * current }
```

**Correct Solution:**
```sysml
constraint def PowerLaw {
    attribute power : Real;
    attribute voltage : Real;
    attribute current : Real;
    
    inv powerEquation { power == voltage * current }
}
```

**Key Points:**
- Use `==` for equality in constraints, not `=`
- Name constraint expressions for clarity
- Define all referenced attributes

---

### Error: Missing Assertion Context
**Category:** Constraint Definition  
**Severity:** Medium  
**Error Code:** CD003

**Problem Description:**
Constraints not properly bound to specific parts or contexts.

**Incorrect Code:**
```sysml
constraint powerBalance : PowerBalance;
```

**Correct Solution:**
```sysml
part def ElectricalSystem {
    attribute inputPower : Real;
    attribute outputPower : Real;
    attribute efficiency : Real;
    
    constraint powerBalance : PowerBalance {
        :>> inputPower = ElectricalSystem::inputPower;
        :>> outputPower = ElectricalSystem::outputPower;
        :>> efficiency = ElectricalSystem::efficiency;
    }
}
```

**Key Points:**
- Bind constraints to specific contexts
- Use redefinition (`:>>`) for parameter binding
- Ensure constraint scope is clear

---

## Requirement Definition

### Error: Missing Requirement Structure
**Category:** Requirement Definition  
**Severity:** High  
**Error Code:** RD001

**Problem Description:**
Requirements without proper text, subject, and verification structure.

**Incorrect Code:**
```sysml
requirement SafetyReq;
```

**Correct Solution:**
```sysml
requirement def SafetyRequirement {
    subject vehicle : Vehicle;
    
    require constraint {
        doc /* The vehicle shall detect system failures within 50ms */
    }
    
    satisfy by vehicle.diagnosticSystem;
}

requirement safetyReq : SafetyRequirement;
```

**Key Points:**
- Define requirement subjects clearly
- Include requirement text in `doc` comments
- Specify satisfaction relationships

---

### Error: Missing Traceability Relationships
**Category:** Requirement Definition  
**Severity:** Medium  
**Error Code:** RD002

**Problem Description:**
Requirements not connected to design elements through traceability.

**Incorrect Code:**
```sysml
requirement performanceReq : PerformanceRequirement;
part engine : Engine;
```

**Correct Solution:**
```sysml
requirement performanceReq : PerformanceRequirement;
part engine : Engine;

satisfy performanceReq by engine;
```

**Key Points:**
- Use `satisfy` relationships for traceability
- Connect requirements to implementing elements
- Maintain bidirectional traceability

---

### Error: Incorrect Requirement Verification
**Category:** Requirement Definition  
**Severity:** Medium  
**Error Code:** RD003

**Problem Description:**
Missing or incorrectly defined verification methods for requirements.

**Incorrect Code:**
```sysml
requirement speedReq : SpeedRequirement;
```

**Correct Solution:**
```sysml
verification def SpeedTest {
    subject testVehicle : Vehicle;
    
    verify requirement speedReq {
        /* Test procedure to verify speed requirement */
    }
}

verify speedReq by speedTest : SpeedTest;
```

**Key Points:**
- Define verification procedures
- Link verification to specific requirements
- Include test methodology

---

## Interface and Connection

### Error: Missing Interface Definition
**Category:** Interface and Connection  
**Severity:** High  
**Error Code:** IC001

**Problem Description:**
Connecting parts without defining proper interfaces.

**Incorrect Code:**
```sysml
connect engine to transmission;
```

**Correct Solution:**
```sysml
interface def MechanicalInterface {
    attribute torque : Real;
    attribute speed : Real;
}

connection mechanicalLink : MechanicalInterface 
    connect engine.output to transmission.input;
```

**Key Points:**
- Define interfaces before connections
- Specify interface attributes clearly
- Use typed connections

---

### Error: Missing Port Definitions
**Category:** Interface and Connection  
**Severity:** High  
**Error Code:** IC002

**Problem Description:**
Parts missing proper port definitions for connections.

**Incorrect Code:**
```sysml
part def Engine { }
part def Transmission { }
// Missing port definitions
```

**Correct Solution:**
```sysml
part def Engine {
    port output : ~MechanicalInterface;
}

part def Transmission {
    port input : MechanicalInterface;
}
```

**Key Points:**
- Define ports with proper directions
- Use `~` for output ports (conjugated)
- Match port types to interface definitions

---

### Error: Incorrect Connection Binding
**Category:** Interface and Connection  
**Severity:** Medium  
**Error Code:** IC003

**Problem Description:**
Parts not properly connected through their port definitions.

**Incorrect Code:**
```sysml
// Missing proper connection syntax
```

**Correct Solution:**
```sysml
part vehicle : Vehicle {
    part engine : Engine;
    part transmission : Transmission;
    
    connection :>> mechanicalConnection 
        connect engine.output to transmission.input;
}
```

**Key Points:**
- Use binding connectors in assembly contexts
- Reference specific port instances
- Maintain connection type consistency

---

## Type System and Definition

### Error: Attribute Must Be Typed by Attribute Definitions
**Category:** Type System and Definition  
**Severity:** High  
**Error Code:** TS001

**Problem Description:**
Using part definitions or undefined types where attribute definitions are required.

**Incorrect Code:**
```sysml
part def Engine {
    attribute power : Real;
}

part def Vehicle {
    attribute engine : Engine;  // ERROR: Engine is a part def, not an attribute def
}
```

**Correct Solution:**
```sysml
import ScalarValues::*;
import ISQ::*;

part def Engine {
    attribute power : Real;
}

part def Vehicle {
    part engine : Engine;              // Use 'part' for part definitions
    attribute mass : MassValue;        // Use attribute definition
    attribute speed : Real;            // Use standard type
    attribute name : String;           // Use standard type
}
```

**Key Points:**
- Use `part` for structural components
- Use `attribute` for data properties  
- Import standard attribute types
- Distinguish between parts and attributes clearly

---

### Error: Invalid Attribute Definition Structure
**Category:** Type System and Definition  
**Severity:** High  
**Error Code:** TS002

**Problem Description:**
Incorrectly defining custom attribute definitions.

**Incorrect Code:**
```sysml
part def MyAttribute {              // Should be 'attribute def'
    // attribute definition content
}

attribute MyCustomType {            // Missing 'def' keyword
    // definition content  
}
```

**Correct Solution:**
```sysml
attribute def MyCustomType :> String {
    // Custom attribute definition content
}

attribute def TemperatureRange :> Real {
    assert constraint { that >= -40.0 and that <= 150.0 }
}

// Usage in parts
part def Engine {
    attribute operatingTemp : TemperatureRange;
    attribute status : VehicleStatus;
}
```

**Key Points:**
- Use `attribute def` for custom attribute types
- Specialize from appropriate base types
- Include validation constraints when needed

---

## Parser and Syntax

### Error: Missing EOF (End of File)
**Category:** Parser and Syntax  
**Severity:** High  
**Error Code:** PS001

**Problem Description:**
Incomplete model definition causing parser to expect more content.

**Incorrect Code:**
```sysml
part def Vehicle {
    attribute mass : Real;
    part engine : Engine {
        attribute power : Real
    // Missing closing brace for engine
// Missing closing brace for Vehicle
```

**Correct Solution:**
```sysml
part def Vehicle {
    attribute mass : Real;
    part engine : Engine {
        attribute power : Real;
    }
}
```

**Key Points:**
- Ensure all opening braces have closing braces
- Check nested definition structures
- Validate complete syntax before compilation

---

### Error: Mismatched Input - 'entry' Expecting '}'
**Category:** Parser and Syntax  
**Severity:** High  
**Error Code:** PS002

**Problem Description:**
Using state machine keywords in wrong context or missing braces.

**Incorrect Code:**
```sysml
part def Engine {
    attribute temperature : Real;
    entry action startHeating;  // 'entry' not valid here
}
```

**Correct Solution:**
```sysml
part def Engine {
    attribute temperature : Real;
    
    state machine {
        state idle {
            entry action initialize;
        }
        state running {
            entry action startHeating;
            exit action stopHeating;
        }
    }
}
```

**Key Points:**
- Use `entry` only within state definitions
- Ensure proper state machine context
- Check brace matching in complex structures

---

### Error: Invalid Import Syntax
**Category:** Import and Namespace Syntax  
**Severity:** Medium  
**Error Code:** INS001

**Problem Description:**
Using wrong import statement format or separators.

**Incorrect Code:**
```sysml
import ScalarValues.Real;           // Wrong separator
import from ScalarValues::Real;     // Wrong keyword
include ScalarValues::*;           // Wrong keyword
```

**Correct Solution:**
```sysml
import ScalarValues::Real;
import ScalarValues::*;
import ISQ::MassValue;
```

**Key Points:**
- Use `::` separator for namespace qualification
- Use `import` keyword, not `include` or `using`
- Import specific types or use wildcard `*`

---

## Search and Retrieval Tags

**Common Error Terms:**
- part definition, block syntax, SysML v1, SysML v2
- action definition, flow connection, succession
- state definition, state machine, transition
- constraint definition, expression syntax, KerML
- requirement definition, traceability, verification
- interface definition, connection, port
- allocation, namespace, import
- attribute definition, type system, specialization
- parser error, syntax error, missing brace
- multiplicity, feature, temporal, flow

**Severity Levels:**
- High: Critical errors preventing compilation
- Medium: Semantic errors affecting model validity  
- Low: Style and best practice recommendations

**Error Categories:**
- Syntax: Parser and language syntax issues
- Semantic: Type system and model consistency issues  
- Structural: Architectural and design pattern issues
- Behavioral: Action, state, and flow modeling issues