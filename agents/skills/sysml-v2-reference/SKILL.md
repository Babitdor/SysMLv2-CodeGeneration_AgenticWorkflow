---
name: sysml-v2-reference
description: Authoritative SysML v2 spec material — language reference, AI-agent guidance, and runnable example models from voidaliot/sysml-v2-docs. Reference material (not generation rules) — reach for it when a per-diagram skill needs a citation, when validator errors mention an unrecognised construct, or when an example for a specific language feature is needed.
license: proprietary
---

# SysML v2 Reference

A curated copy of the [voidaliot/sysml-v2-docs](https://github.com/voidaliot/sysml-v2-docs) documentation set — language reference, AI-agent guidance, and runnable example models — mounted alongside the per-diagram skills so the SysML agent can read authoritative spec material without an internet round-trip.

This is **reference material, not generation rules**. The diagram-specific skills under `/skills/<diagram>/SKILL.md` remain the primary contract for what to emit. Reach for the files here when:

- You hit a construct the inline skill examples don't cover.
- The validator reports an *unrecognised construct* and you'd otherwise call `sysml_syntax_reference_tool`.
- You need to remind yourself of a hard rule (keyword reservations, definition-vs-usage, import visibility, requirement satisfy-vs-assume distinction).
- You want to see a complete worked model end-to-end (the `examples/` files are minimal but valid).

## File index

### `language/` — the SysML v2 language itself

| File | Purpose |
|---|---|
| [`01-overview.md`](language/01-overview.md) | Architecture: KerML → SysML → API stack. Read first if you've never seen the layered language. |
| [`02-textual-notation-syntax.md`](language/02-textual-notation-syntax.md) | Complete grammar reference for the textual notation. |
| [`03-keywords-and-operators.md`](language/03-keywords-and-operators.md) | All reserved words and symbols — useful for "why won't this name parse?". |
| [`04-definition-vs-usage.md`](language/04-definition-vs-usage.md) | The core paradigm of `def` vs usage. **Read before authoring new types.** |
| [`05-structural-modeling.md`](language/05-structural-modeling.md) | `part` / `port` / `item` / `interface` / `connection` — covers BDD and IBD. |
| [`06-behavioral-modeling.md`](language/06-behavioral-modeling.md) | `action` / `state` / `transition` / `flow` / `calculation` — covers activity diagrams. |
| [`07-requirements-and-constraints.md`](language/07-requirements-and-constraints.md) | `requirement` / `constraint` / `satisfy` / `assume` / `require`. |
| [`08-analysis-and-verification.md`](language/08-analysis-and-verification.md) | `analysis case` / `verification case` / `use case`. |
| [`09-packages-imports-and-views.md`](language/09-packages-imports-and-views.md) | Namespaces, imports, `view` / `viewpoint` / `rendering`. |
| [`10-standard-libraries.md`](language/10-standard-libraries.md) | Kernel / Systems / Domain libraries (units, geometry…). |

### `ai-agents/` — guidance written for LLM agents

| File | Purpose |
|---|---|
| [`01-modeling-guide.md`](ai-agents/01-modeling-guide.md) | How to translate user intent into SysML v2. |
| [`02-prompting-rules.md`](ai-agents/02-prompting-rules.md) | Hard rules and invariants the agent must respect. |
| [`03-common-patterns.md`](ai-agents/03-common-patterns.md) | Reusable templates (system block, requirement block, state machine…). |
| [`04-validation-checklist.md`](ai-agents/04-validation-checklist.md) | Pre-output checklist — walk through this **before** returning generated code. |

### `examples/` — runnable starter models

| File | Purpose |
|---|---|
| [`examples/flashlight.sysml`](examples/flashlight.sysml) | Minimal end-to-end model. |
| [`examples/vehicle-skeleton.sysml`](examples/vehicle-skeleton.sysml) | Larger model showcasing all major constructs. |

## How to use this without burning context

The temptation is to load every file into the conversation at once. Don't — the language reference alone is ~75 KB. Instead:

1. **Read the per-diagram skill first** (`/skills/<diagram>/SKILL.md`). It has inline few-shot examples that cover the common 90% of cases.
2. **Only read a `language/` file when you hit a gap** — and read just that one file, not the whole tree.
3. **`ai-agents/04-validation-checklist.md`** is the highest-leverage single read. Consult it once per session before emitting the first `.sysml` file.
4. **`examples/`** is for shape calibration — skim a relevant one when you're unsure how a complete file should be structured, not as a copy source.

## Provenance

Files are mirrored verbatim from `voidaliot/sysml-v2-docs@main` and may lag the upstream repo. Refresh by re-running the fetch step in commit `<refresh-commit-sha>` against the current upstream `main`.
