---
name: verification-case-diagram
description: Generate SysML v2 Verification Case Diagrams — `verification case def` with `subject`, `objective verify <Requirement>`, ordered `action` steps, and per-case usages that bind the abstract verification to concrete subjects. Use when the user wants to express "prove that this part meets that requirement" — the formal counterpart of UML's «testCase». A verification engine can walk every verification case usage and evaluate pass/fail per case from the underlying require-constraint.
license: proprietary
---

## Purpose

A Verification Case Diagram declares **how a requirement is checked**. It is a kind of `action def` specialised for verification — it has a `subject` (the thing being verified), one or more `objective verify <Requirement>` declarations (the requirements being checked), and ordered sub-actions describing the test procedure. A verification case **passes** when the linked requirement evaluates to `true` after the case runs.

Where `requirement def` says "this is the rule", `verification case def` says "this is the test that demonstrates the rule holds for a concrete subject". A CI pipeline can run every `verification case` usage and aggregate the results.

## Syntax Rules

- Use `verification case def <Name> { … }` — the keyword is two words (`verification case`), not `verificationcase`
- Required elements inside the body:
  - `subject <name> : <SubjectPart>;` — exactly ONE subject
  - `objective <name> { verify <RequirementUsage>; }` — at least one objective with a `verify` link to a `requirement`
- Recommended elements:
  - `doc /* free-form description of what is verified and how */`
  - Action declarations (`action setup; action runTest; action measureResult;`) with `first … then …` ordering for the procedure
- For a concrete usage that binds the abstract case to a specific subject:
  `verification case <name> : <VerificationCaseDef> { subject = <concretePart>; }`
- `verify <Target>` accepts a `requirement` usage or a sub-requirement reference (e.g. `verify VehicleSafety::crashWorthiness`)
- Imports MUST have visibility: `public import ScalarValues::*;` and the requirement package being verified
- Reuse via `:>` — `verification case def CrashTestVariantB :> CrashTest { … }` overrides or extends behaviour while keeping the same verification objective
- Optional `actor` and `stakeholder` declarations attach external participants and interested parties (e.g. `actor tester : Person;` / `stakeholder safetyEngineer;`)
- NEVER use v1 keywords: `«testCase»`, `«verifies»`, `testedBy` stereotypes are not v2 keywords
- NEVER omit the `verify` link inside `objective` — a verification case without a verify target is structurally indistinguishable from a generic action

## Few-Shot Examples

### Example 1: Mass Verification (Single Requirement)

```sysml
package MassVerification {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;
    public import VehicleRequirements::*;

    verification case def MassCheck {
        doc /* Weighs the vehicle and verifies the mass requirement. */

        subject veh : Vehicle;
        objective verifyMass {
            verify MaxMassLimit;
        }

        action putOnScale;
        action readScale;
        action recordResult;

        first putOnScale then readScale then recordResult;
    }

    // Concrete usage — binds to a specific vehicle instance.
    verification case checkMyCar : MassCheck {
        subject = myCar;
    }
}
```

### Example 2: Crash Test with Multiple Sub-Steps and an Actor

```sysml
package CrashVerification {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;
    public import VehicleRequirements::*;

    action def MountCrashDummy;
    action def AccelerateToImpact;
    action def MeasureForces;
    action def ReportRating;

    verification case def CrashTest {
        doc /* Frontal-impact verification per FMVSS 208. */

        subject veh : Vehicle;
        actor   tester : CertifiedTester;
        stakeholder safetyEngineer;

        objective verifyCrash {
            verify VehicleSafety::crashWorthiness;
        }

        action setup     : MountCrashDummy;
        action runImpact : AccelerateToImpact;
        action measure   : MeasureForces;
        action report    : ReportRating;

        first setup
        then  runImpact
        then  measure
        then  report;
    }
}
```

### Example 3: Multiple Objectives in One Case

```sysml
package PowertrainVerification {
    public import ScalarValues::*;
    public import ISQ::*;
    public import SI::*;
    public import VehicleRequirements::*;

    verification case def PowertrainBench {
        doc /* Single bench test that exercises multiple requirements. */

        subject veh : Vehicle;

        objective verifyTopSpeed {
            verify TopSpeedRequirement;
        }
        objective verifyAcceleration {
            verify Accel0to60Requirement;
        }
        objective verifyEfficiency {
            verify FuelEconomyRequirement;
        }

        action mountDyno;
        action runWideOpen;
        action runUrbanCycle;
        action recordAll;

        first mountDyno then runWideOpen then runUrbanCycle then recordAll;
    }
}
```

## Common Pitfalls

- **Writing `verificationCase def`** (single token) → validator error `Unexpected token` — the keyword is `verification case` (two tokens, space-separated)
- **Missing `subject`** → semantic error; every verification case needs exactly one subject
- **`verify` outside an `objective` block** → the `verify` clause must live inside an `objective { … }` block. Bare `verify X;` at the verification case body level is invalid
- **Multiple subjects** → not legal; if you need to verify a relationship between two parts, model the relationship as a single owning part and verify that part
- **Forgetting to import the requirement package** → the `verify <RequirementName>` reference fails to resolve. Add `public import <RequirementsPkg>::*;`
- **Mixing UML stereotypes** (`«testCase»`, `«verifies»`) → unknown identifiers; rewrite as `verification case def` + `objective verify`

## When to Reach for This Skill

| User intent | Use |
|---|---|
| "Test that the vehicle meets the mass limit." | `verification case def MassCheck` + `objective verify MaxMassLimit` + usage binding `subject = myCar` |
| "Run a single bench test that checks several requirements at once." | One verification case def with multiple `objective` blocks, each verifying a different requirement |
| "The crash test is performed by a certified tester for the safety engineer." | Add `actor tester : CertifiedTester;` and `stakeholder safetyEngineer;` |
| "Re-use the crash test for variant B but change the impact angle." | Specialisation: `verification case def CrashTestSideImpact :> CrashTest { … }` |
| "Walk all verification cases and report pass/fail." | Tooling-side concern — the case definitions provide the data; a verification engine evaluates each `verify` link |
