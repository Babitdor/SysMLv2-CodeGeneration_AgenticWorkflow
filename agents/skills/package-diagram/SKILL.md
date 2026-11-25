---
name: package-diagram
description: Generate SysML v2 Package Diagrams — top-level `package` declarations with `public`/`private import` statements, nested packages for organisation, aliasing, recursive imports, and `filter` for visibility restriction. Use when the user wants to organise a multi-file model, control name visibility between packages, or document the model's namespace hierarchy. Every well-formed `.sysml` file starts with a package declaration; this skill teaches the idiomatic structure.
license: proprietary
---

## Purpose

A Package Diagram declares the **namespace structure** of a model — which package owns which elements, who imports whom, and what's public versus private across the boundary. SysML v2 makes packages first-class with explicit visibility (`public import`, `private import`), recursive imports (`Pkg::*::**`), aliasing (`import X as Y`), and filtered re-exports (`filter X and not Y`).

Every model element lives transitively inside a package. Getting the package structure right up front prevents the most common validator failures: unresolved references caused by missing or wrong-visibility imports.

## Syntax Rules

- Every `.sysml` file SHOULD begin with exactly one top-level `package <Name> { … }` declaration — the recommended convention even when v2 technically allows top-level members
- Package names use `PascalCase`: `VehicleModel`, `Requirements`, `Verification`
- Imports MUST declare visibility — `public import` re-exports the imported members from this package; `private import` (or shorthand `import`) keeps them local
- Import forms:
  - `import OtherPkg::*;` — all direct members of OtherPkg
  - `import OtherPkg::SomeDef;` — one named member
  - `import OtherPkg::*::**;` — recursive (all members of all sub-packages)
  - `import OtherPkg::SomeDef as Alias;` — aliasing for collision avoidance or clarity
- Standard library packages use the `standard library package` modifier and are NEVER declared by user code
- Common standard imports the v2 agent needs almost everywhere:
  - `public import ScalarValues::*;` — `Real`, `Integer`, `Boolean`, `String`
  - `public import ISQ::*;` — quantity types (`MassValue`, `LengthValue`, `SpeedValue`, `PowerValue`, etc.)
  - `public import SI::*;` — unit symbols (`kg`, `m`, `s`, `W`, `km/h`)
- Sub-packages nest naturally: `package Outer { package Inner { part def X; } }`
- Use `filter <expr>;` inside a package body to restrict visible members (e.g. `filter Vehicle and not Engine;`) — typically combined with `public import OriginalPkg::*;`
- NEVER use `Namespace`, `module`, `library` (lowercase), or `using` — these are not v2 keywords
- NEVER mix top-level package members and a top-level `package` declaration in the same file — pick one

## Few-Shot Examples

### Example 1: Minimal Single-Package File (Most Common Shape)

```sysml
package WarehouseModel {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    part def Warehouse {
        attribute totalArea : AreaValue = 5000.0 [m * m];
        attribute capacity  : Integer   = 10000;
    }
}
```

### Example 2: Multi-Package File with Public Re-Exports

```sysml
package VehicleModel {
    package Structure {
        public import ScalarValues::*;
        public import ISQ::*;
        public import SI::*;

        part def Vehicle {
            attribute mass : MassValue = 1500.0 [kg];
        }
    }

    package Behavior {
        public import Structure::*;     // pulls Vehicle into Behavior's scope

        action def Drive {
            in vehicle : Vehicle;
        }
    }

    package Requirements {
        public import Structure::*;     // Vehicle visible here too

        requirement def Safe {
            subject veh : Vehicle;
            require constraint { veh.mass <= 2500.0 [kg] }
        }
    }
}
```

### Example 3: Cross-File Aliasing + Filtered Re-Export

```sysml
// File: ExternalVehicleParts.sysml
package ExternalVehicleParts {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;

    part def Wheel  { attribute pressure : PressureValue; }
    part def Engine { attribute mass     : MassValue;     }
    part def Brake;
}

// File: VisibleSubset.sysml — only re-expose Wheel + Brake, hide Engine
package VisibleSubset {
    filter Wheel or Brake;
    public import ExternalVehicleParts::*;
}

// File: MyApp.sysml — alias to avoid name collision with another package's Wheel
package MyApp {
    public import VisibleSubset::Wheel as ExtWheel;
    public import VisibleSubset::Brake;
}
```

## Common Pitfalls

- **Missing visibility on import** → defaults to `private`; downstream files that try to `import MyPkg::*` won't see members that arrived via private import. Always use `public import` for members you want to re-export
- **`OtherPkg::*::**` (recursive) on big standard libraries** → bloats the namespace and slows the validator. Prefer the narrowest import that satisfies the file's needs
- **Two top-level packages in one file** → not legal. Split into separate files; cross-reference via import
- **Standard library import without visibility** → `import ScalarValues::*;` (no `public`) means downstream packages can't see `Real`, `Integer`, etc. Use `public import ScalarValues::*;` in any package whose elements expose scalar-typed features
- **Aliasing applied to a wildcard** — `import OtherPkg::* as X;` is NOT legal; aliasing only applies to a named element
- **Name collisions across imports** → if `PkgA::*` and `PkgB::*` both export `Vehicle`, references become ambiguous. Resolve with explicit named imports + aliases, or fully qualify at the use site (`PkgA::Vehicle`)

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "Start a new model file." | Minimal single-package skeleton with standard library imports |
| "Split a growing model into Structure / Behavior / Requirements." | Multi-package file with `public import` to re-export shared types into sibling packages |
| "Hide a few internal types from downstream consumers." | `filter` + `public import` to construct a visibility-restricted re-export package |
| "Two upstream packages both export `Wheel`." | Aliased imports (`import PkgA::Wheel as PkgAWheel;`) |
| "Pull in everything from a sub-tree of packages." | Recursive import `OtherPkg::*::**` — but only when you really need every member |
| "Conform to the OMG file-organisation convention." | One top-level package per file, file name = top-level package name (`Structure.sysml` → `package Structure { … }`) |
