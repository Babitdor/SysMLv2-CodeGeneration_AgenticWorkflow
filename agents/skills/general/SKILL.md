---
name: general
description: Cross-cutting SysML v2 fix patterns that don't map to a single diagram type — package imports, ScalarValues usage, visibility modifiers, top-level package syntax, scalar-attribute default syntax, and constructs from diagram types without a dedicated skill. The SysML agent reads this in Phase A and appends fixes in Phase C alongside per-diagram LEARNED.md files.
license: proprietary
---

# General SysML v2 Fixes (Cross-Cutting)

This is the **fallback bucket** for SysML v2 fix patterns that don't belong to any single per-diagram skill. The SysML agent reads `LEARNED.md` here in Phase A and appends to it in Phase C whenever a fix is cross-cutting.

## When to use the general bucket

Append to `/skills/general/LEARNED.md` instead of a per-diagram `LEARNED.md` when the fix applies to **any** of:

- **Cross-diagram patterns** — the same validator error and fix recur across two or more of `requirement_diagram`, `usecase_diagram`, `activity_diagram`, `bdd`, `ibd`. Recording it once here is better than scattering five near-duplicates across diagram-specific files.
- **Top-level package syntax** — `package Foo { ... }` rules, naming, nesting, comments inside packages.
- **Imports & visibility** — `public import ScalarValues::*;`, `private import …`, qualified-name resolution.
- **Standard library usage** — anything from `ScalarValues`, `Quantities`, `MeasurementReferences`, `Time` that bites you generically (not specific to a single diagram).
- **Reserved keywords / lexer issues** — names colliding with `variant`, `action`, `flow`, etc. that the parser rejects regardless of which diagram you're authoring.
- **Constructs from diagram types without a dedicated skill** — e.g. a state-machine or constraint-diagram pattern you tried for a downstream user request that has no `/skills/<diagram>/SKILL.md` yet.

If a fix is clearly specific to a single diagram (say, "always declare action usages separately in activity decisions"), it still goes in that diagram's `LEARNED.md` — not here.

## Entry format

Identical to per-diagram `LEARNED.md` files so a future indexer can fold all of them together:

```markdown
## Session YYYY-MM-DD
- Error: `<verbatim validator error message>`
  Fix: `<what was changed and why>`
- Error: `<verbatim validator error>`
  STATUS: UNRESOLVED — `<why it couldn't be fixed>`
```

Keep entries to one logical fix per bullet. Consolidate duplicates — if the pattern already exists in this file from a prior session, do NOT add it again.

## What this skill is NOT

- It is **not** a syntax reference. For authoritative SysML v2 syntax, use `sysml_syntax_reference_tool` or `read_file` on `/skills/sysml-v2-reference/language/*.md`.
- It is **not** a tutorial. The per-diagram `SKILL.md` files own the few-shot examples and authoring rules.
- It is **not** read-only. The point is that the SysML agent writes back to `LEARNED.md` here over time, so this bucket grows as a corpus of observed cross-cutting failure modes.
