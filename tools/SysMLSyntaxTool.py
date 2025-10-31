from langchain.tools import BaseTool


class SysMLSyntaxTool(BaseTool):
    """Quick syntax reference lookup"""

    name: str = "sysml_syntax_reference"
    description: str = """
    Get syntax reference for specific SysML v2 constructs.
    Input: Type of construct (e.g., "part definition", "connection", "state")
    Output: Syntax rules and examples for that construct
    Use this when you're unsure about correct syntax.
    """

    def _run(self, construct_type: str) -> str:
        """Get syntax reference"""
        # This would query a syntax database or structured docs
        syntax_db = {
            "package": """
Package Syntax:
package <name> [ { 
    import <package-name> [ as <alias> ];
    alias <oldName> = <newName>;
    <member-definitions…>
} ]

Example:
package Vehicles {
    import Controls as Ctl;
    alias Engine = Ctl.Engine;
    // part, type, requirement definitions …
}
    """,
            "alias": """
Alias Syntax:
alias <oldName> = <newName>;
// Alternative: alias <alias> for <name>;

Example:
alias Engine = Ctl.Engine;
alias MV for ISQ::MassValue;
    """,
            "import": """
Import Syntax:
<visibility> import <packageName> [ as <alias> ];
// visibility: public or private

Example:
import Controls as Ctl;
public import VehicleParts;
private import ISQ::MassValue;
private import SI::*;
    """,
            "type definition": """
Type Definition Syntax:
typedef <name> [: <supertype>] {
    attribute <name> : <type> [= <value>];
    // optional operations, constraints, nested definitions
}

Example:
typedef Weight : Real {
    attribute unit : String = "kg";
}
    """,
            "enumeration definition": """
Enumeration Definition Syntax:
enum <name> {
    <literal1>,
    <literal2>,
    …
}

Example:
enum Color {
    Red,
    Green,
    Blue
}
    """,
            "part definition": """
Part Definition Syntax:
part def <name> [: <specialization>] {
    attribute <name> : <type> [= <value>];
    part <name> : <type>;
    port <name> : <type>;
    // nested parts, flows, constraints, etc.
}

Example:
part def Vehicle {
    attribute mass : Real = 1500.0;
    part engine : Engine;
}
    """,
            "part usage": """
Part Usage Syntax:
part <name> : <type> {
    :>> <attribute> = <value>;  // redefinition
    part <subpart> : <type>;
}

Example:
part vehicle : Vehicle {
    :>> mass = 1500 [kg];
    part engine : Engine;
}
    """,
            "item definition": """
Item Definition Syntax:
item def <name> [: <specialization>] {
    attribute <name> :> <type>;
    // attributes, constraints
}

Example:
item def Fuel {
    attribute fuelMass :> ISQ::mass;
}
    """,
            "port definition": """
Port Definition Syntax:
port def <name> [: <specialization>] {
    attribute <name> : <type> [= <value>];
    in <signalName> : <type>;
    out <signalName> : <type>;
}

Example:
port def FuelingPort {
    attribute maxFlow : Real = 500.0;
    out fuelOut : Fuel;
    in fuelIn : Fuel;
}
    """,
            "port usage": """
Port Usage Syntax:
port <name> : <type>;

Example:
port fuelTankPort : FuelingPort;
    """,
            "connection definition": """
Connection Definition Syntax:
connection def <name> {
    end part <name> : <type>;
    end part <name> : <type>;
    attribute <name> : <type>;
}

Example:
connection def DeviceConn {
    end part hub : Hub;
    end part device : Device;
    attribute bandwidth : Real;
}
    """,
            "connection usage": """
Connection Usage Syntax:
connection <name> : <type> {
    end part <name> ::> <target>;
    end part <name> ::> <target>;
}

Example:
connection conn : DeviceConn {
    end part hub ::> mainSwitch;
    end part device ::> sensorFeed;
}
    """,
            "action definition": """
Action Definition Syntax:
action [def] <name> [: <specialization>] {
    in <param> : <type>;
    out <param> : <type>;
    // body: local parts, statements, actions, use of flows
}

Example:
action def CalculateFuelConsumption {
    in mass : Real;
    in distance : Real;
    out fuelUsed : Real;
    // … statements …
}
    """,
            "composite action": """
Composite Action Syntax:
action def <name> {
    action <subAction1> : <type>;
    action <subAction2> : <type>;
    first <action1> then <action2>;
}

Example:
action def Drive {
    action accelerate : Accelerate;
    first start then accelerate;
}
    """,
            "perform action": """
Perform Action Syntax:
perform action <name> : <type>;

Example:
part vehicle : Vehicle {
    perform action driveVehicle : Drive;
}
    """,
            "state definition": """
State Definition Syntax:
state def <name> [: <specialization>] {
    state <subState1>;
    state <subState2>;
    entry / <actionName>;
    exit / <actionName>;
    transition <from> -> <to> [ when <condition>; ] [ do <actionName>; ];
}

Example:
state def EngineStates {
    state Off;
    state Running;
    entry / InitializeEngine;
    exit / ShutDownEngine;
    transition Off -> Running when startSignal = true do StartEngine;
}
    """,
            "constraint definition": """
Constraint Definition Syntax:
constraint def <name> {
    in <param> : <type>;
    <boolean-expression>
}

Example:
constraint def IsFull {
    in tank : FuelTank;
    tank.fuelLevel == tank.maxFuelLevel
}
    """,
            "constraint usage": """
Constraint Usage Syntax:
constraint <name> : <type> {
    in <param> = <value>;
}

Example:
part def Vehicle {
    part fuelTank : FuelTank;
    constraint tankIsFull : IsFull {
        in tank = fuelTank;
    }
}
    """,
            "requirement definition": """
Requirement Definition Syntax:
requirement def <name> {
    subject <element> : <type>;
    attribute <name> : <type> [= <value>];
    require constraint { <boolean-expression> }
}

Example:
requirement def MassRequirement {
    subject vehicle : Vehicle;
    attribute massActual : ISQ::MassValue;
    attribute massLimit : ISQ::MassValue = 2000.0;
    require constraint { massActual <= massLimit }
}
    """,
            "requirement usage": """
Requirement Usage Syntax:
requirement <<id>> <name> : <type> {
    attribute :>> <attr> = <value>;
    subject <element> = <target>;
}

Example:
requirement <R1> vehicleMass : MassRequirement {
    attribute :>> massActual = vehicle.mass;
    attribute :>> massLimit = 1800 [kg];
}
    """,
            "satisfy requirement": """
Satisfy Requirement Syntax:
satisfy <requirement-id> by <element>;

Example:
satisfy R1 by vehicle;
    """,
            "flow definition": """
Flow Definition Syntax:
flow def <name> [: <specialization>] {
    source <port> : <type>;
    target <port> : <type>;
    attribute <name> : <type> [= <value>];
    // maybe constraints
}

Example:
flow def FuelFlow {
    source fuelingPort.fuelIn : Fuel;
    target engine.fuelIn : Fuel;
    attribute rate : Real = 500.0;
}
    """,
            "connector definition": """
Connector Syntax:
connector def <name> [: <specialization>] ( <end1> , <end2> ) {
    // optional attributes, constraints
    attribute <name> : <type> [= <value>];
}

Example:
connector def DriveShaft ( engine.output, transmission.input ) {
    attribute torqueCapacity : Real = 250.0;
}
    """,
            "unit / quantity": """
Unit / Quantity Syntax:
quantity def <name> [: <unitType>] {
    unit <unitName> = <value> <baseUnit>;
}

Example:
quantity def DistanceValue : Length {
    unit metre = 1.0 m;
    unit kilometre = 1000.0 m;
}
    """,
            "specialization": """
Specialization Syntax:
<element> def <name> :> <supertype> {
    :>> <redefined-property> = <value>;
}
// Short form: :>

Example:
part def SportsCar :> Vehicle {
    :>> wheels = 4;
}
    """,
            "subsetting": """
Subsetting Syntax:
<element> <name> [<multiplicity>] :> <set>;
// Short form: :>

Example:
part engine1 [1] :> engines;
    """,
            "redefinition": """
Redefinition Syntax:
:>> <property> = <value>;
// Short form: :>>

Example:
:>> total_wheels = 2;
:>> mass = 1500 [kg];
    """,
            "references": """
References Syntax:
<element> ::> <target>;
// Short form: ::>

Example:
end part hub ::> mainSwitch;
    """,
            "multiplicity": """
Multiplicity Syntax:
<name> [<count>] : <type>           // exact count
<name> [<lower>..<upper>] : <type>  // range

Example:
wheel [4] : Wheel;
Wheel [0..*];
part [1..5] : Component;
    """,
            "assignment": """
Assignment Syntax:
<target> = <value>;

Example:
mass = 1500 [kg];
tank.fuelLevel = tank.maxFuelLevel;
    """,
            "comparison operators": """
Comparison Operators:
< <= == != >= >     // value comparison
=== !==             // identity comparison

Example:
tank.fuelLevel == tank.maxFuelLevel
massActual <= massLimit
wheels === 4
    """,
            "metadata": """
Metadata Syntax:
@<metadata-name>

Example:
@ToolMetadata
@Deprecated
    """,
            "user-defined keyword": """
User-Defined Keyword Syntax:
#<keyword> '<text>'

Example:
#cause 'Short Circuit'
#rationale 'Safety requirement'
    """,
            "comment": """
Comment Syntax:
// <text>              // line comment
/* <text> */          // block comment
//* <text> *//         // also block comment

Example:
// This is a single line comment
/* This is a
   multi-line comment */
    """,
            "documentation": """
Documentation Syntax:
doc /* <text> */

Example:
doc /* This part represents a vehicle with engine and wheels */
    """,
            "qualified reference": """
Qualified Reference Syntax:
<namespace>::<member>

Example:
ISQ::MassValue
SI::kg
VehicleModel::Vehicle
    """,
        }

        return syntax_db.get(
            construct_type.lower(),
            f"Syntax reference for '{construct_type}' not found.",
        )

    async def _arun(self, construct_type: str) -> str:
        return self._run(construct_type)


syntax_tool = SysMLSyntaxTool()
