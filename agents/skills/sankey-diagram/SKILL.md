---
name: sankey-diagram
description: Generate Sankey diagrams from sandbox artifacts showing quantitative flows between MBSE entities (Activities, Logical Blocks, Requirements, Use Cases). Use when the user wants to visualize traceability chains, activity-to-block flows, satisfaction coverage, or morphological box structure.
license: proprietary
---

## Purpose

A Sankey diagram visualizes the *quantitative magnitude* of flows between MBSE entities — traceability chains, activity-to-block realizations, satisfaction coverage, or morphological decomposition. The agent reads structured markdown artifacts and SysML files from `/workspace/current/`, extracts entities and their relationships, then builds node/link arrays rendered as interactive Plotly Sankey diagrams.

Use this skill when the user wants to:
- Visualize the full traceability chain: Requirements → Use Cases → Activities → Logical Blocks
- Show which Activities are realized by which Logical Blocks (activity_flow)
- Audit satisfaction coverage: Requirements → Logical Blocks
- Explore morphological box structure: Functions → Options

## Data Sources (Sandbox Files)

The Sankey agent reads artifacts and SysML files from the project sandbox at
`/workspace/current/`. All data is extracted from structured markdown and
SysML v2 files produced by upstream agents.

### Primary Files (always read)

| File | Produced By | Contains |
|---|---|---|
| `logical/BDD.md` | @Logical | Block definitions, subsystem decomposition, properties |
| `logical/IBD.md` | @Logical | Internal block connections, ports, flows between parts |
| `physical/sysml/bdd.sysml` | @SysML | Validated SysML v2 block definitions |
| `physical/sysml/ibd.sysml` | @SysML | Validated SysML v2 internal block connections |

### Supplementary Files (read when available for richer diagrams)

| File | Produced By | Contains |
|---|---|---|
| `requirements/REQUIREMENTS.md` | @Ingestion | Functional (REQ-NNN) and non-functional (REQ-NF-NNN) requirements |
| `requirements/USECASES.md` | @Requirements | Actors, use cases (UC-NNN) with triggers and scenarios |
| `functional/ACTIVITIES.md` | @Functional | Activities with numbered steps and decision points |
| `requirements/BOX.md` | @Ingestion | Morphological box: functions → solution options |
| `requirements/RULES.md` | @Ingestion | Design constraints (C1, C2, ...) |

### Extracting Entities from Files

#### Logical Blocks (from BDD.md)
Parse BDD.md for block hierarchy:
- Top-level block name from `## <Top-Level Block>` heading
- `Subsystems`: comma-separated list of child block names
- `Properties`: key-value pairs (`<prop>: <Type>`)
- Sub-blocks from `## <Sub-Block>` headings with their own `Parts` and `Properties`

#### Connections & Ports (from IBD.md)
Parse IBD.md for flow data:
- Connections: `- <PartA> ─[<flowName>]→ <PartB>` — each line is a directed flow
- Ports: `- <PartA>.<portName>: <PortType>` — port-level interface data
- The top-level block name from `# Internal Block: <Top-Level Block>`

#### Activities (from ACTIVITIES.md)
- Activity name from `# Activity: <Name>` headings
- Numbered steps under each activity
- Decision points: `- If <condition> → <action>`

#### Requirements (from REQUIREMENTS.md)
- Functional: `REQ-NNN` IDs under `# Functional Requirements`
- Non-functional: `REQ-NF-NNN` IDs under `# Non-Functional Requirements`

#### Use Cases (from USECASES.md)
- Actors from `# Actors` section (bullet list)
- Use cases with `UC-NNN: <Name>` IDs, `Actor`, `Trigger`, `Precondition`, `Main Success Scenario` steps

#### Morphological Box (from BOX.md)
- Table rows under `# Morphological Box`: `| <Function> | <Option A> | <Option B> | ... |`
- Column headers define available options per function dimension

### Flow Derivation Rules

- **Block-to-Block flows**: Directly from IBD.md connection lines (`PartA ─[flow]→ PartB`)
- **Satisfaction flows**: If a block from BDD.md and a requirement from REQUIREMENTS.md share traceability → draw Block → Requirement or Requirement → Block link
- **Decomposition flows**: Blocks listed in `Subsystems` or `Parts` → parent-to-child flows
- **Activity-to-Block flows**: If ACTIVITIES.md steps reference block names from BDD.md → Activity → Block link

## Building the Activity Flow Sankey

The `activity_flow` diagram type shows the **functional-to-physical mapping**:

1. **Left column**: Activities (from ACTIVITIES.md)
2. **Right column**: Logical Blocks (from BDD.md)
3. **Flows between them** are derived from shared traceability:
   - If an Activity and a LogicalBlock reference the same Requirement → draw a flow
   - If an Activity name matches or references a block from BDD.md → draw a flow
   - IBD.md connections create LogicalBlock → LogicalBlock links (same column)
   - Activity step references to other activities create Activity → Activity links (same column)
4. **Flow values**: see Flow Value Derivation below

### Activity Flow Construction Rules

- NEVER include abstract entities (requirements, use cases) as nodes in activity_flow
- Each Activity node gets `entity_type: "activity"`
- Each LogicalBlock node gets `entity_type: "logical_block"`
- If an Activity references REQ-001 and LogicalBlock B1 also references REQ-001, draw a link: Activity → B1
- Parallel flows (same source → target) are summed by the tool; you do not need to pre-aggregate
- IBD.md connections create LogicalBlock → LogicalBlock links

## Node/Link Construction for sankey_generator_tool

### Node Format
```json
{"id": "B1", "label": "Brake Control Unit", "entity_type": "logical_block", "customdata": "B1: Main braking control logic"}
```
Give every node the `id` the artifacts already use for it (`REQ-001`, `UC-003`, `B1`, or the block name).

### Link Format
```json
{"source": "REQ-001", "target": "B1", "value": 1, "label": "SATISFIES"}
```
`source` / `target` name a node by `id` (or label). Indices also work but shift whenever a node is added — avoid them.

### Entity Type → Color Mapping
The `sankey_generator_tool` auto-assigns colors based on `entity_type`:
- `requirement` → blue
- `use_case` → purple
- `activity` → teal
- `logical_block` → green
- `candidate` → red
- `function_dimension` → orange
- `option` → brown

## Workflow for Each Diagram Type

### Traceability Sankey (Requirements → Use Cases → Activities → Blocks)
1. Read `requirements/REQUIREMENTS.md` — extract all REQ-NNN IDs and text
2. Read `requirements/USECASES.md` — extract all UC-NNN IDs, actors, and scenarios
3. Read `functional/ACTIVITIES.md` — extract activity names and steps
4. Read `logical/BDD.md` — extract block names and subsystems
5. Map requirement references across files (REQ-NNN in ACTIVITIES.md, REQUIREMENTS.md)
6. Build 4-column nodes: Requirements → Use Cases → Activities → Logical Blocks
7. Derive links from ID references and cross-file traceability
8. Call `sankey_generator_tool` ONCE with final node/link arrays

### Activity Flow Sankey (Activities + Logical Blocks)
1. Read `logical/BDD.md` — extract all block names and hierarchy
2. Read `logical/IBD.md` — extract all connections (`PartA ─[flow]→ PartB`)
3. Read `physical/sysml/bdd.sysml` and `physical/sysml/ibd.sysml` for validated block/connection data
4. Read `functional/ACTIVITIES.md` if available — extract activity names
5. Left column: Activities (entity_type="activity")
6. Right column: Logical Blocks (entity_type="logical_block")
7. Links from IBD.md connections → LogicalBlock → LogicalBlock
8. Links from activity-block name references → Activity → LogicalBlock
9. Call `sankey_generator_tool` ONCE with final node/link arrays

### Satisfaction Sankey (Requirements → Logical Blocks)
1. Read `requirements/REQUIREMENTS.md` — extract all REQ-NNN IDs
2. Read `logical/BDD.md` — extract all block names and their properties
3. Cross-reference: if a block property mentions a REQ-NNN ID → satisfaction link
4. Left column: Requirements (entity_type="requirement")
5. Right column: Logical Blocks (entity_type="logical_block")
6. Links: Requirement → LogicalBlock for each matched ID
7. Call `sankey_generator_tool` ONCE

### Morphological Box Sankey (Candidates → Functions → Options)
1. Read `requirements/BOX.md` — parse the morphological box table
2. Read `logical/PRINCIPLE_SOLUTIONS.md` — identify selected option per function
3. Left column: Solutions/candidates
4. Middle column: Function dimensions (from BOX.md row labels)
5. Right column: Options (from BOX.md column values)
6. Links from option selections in PRINCIPLE_SOLUTIONS.md
7. Call `sankey_generator_tool` ONCE

## Flow Value Derivation (CRITICAL)

A Sankey diagram represents FLOW MAGNITUDES — not relationship counts.

Priority order for assigning numeric values:
1. **Quantitative attributes from markdown artifacts** — if IBD.md or BDD.md contain throughput, capacity, or bandwidth values → use directly
2. **Structural heuristics** — equal distribution across outgoing edges, or normalized weights
3. **NEVER use raw edge count as value** — this misrepresents the diagram

If no quantitative data exists:
→ Treat values as **relative flow weights (dimensionless)**, assign 1.0 per edge

## Error Patterns to Avoid

- **Using raw edge count as flow value** — a Sankey shows magnitude, not topology; 3 edges ≠ value 3
- **Mixing entity types in the same column** — activities and logical blocks must be in separate columns
- **Including abstract entities as nodes** — requirements and use cases are NOT nodes in activity_flow
- **Zero-value links** — value must be > 0 or the link is invisible
- **Self-loops** — source and target must be different nodes unless explicitly meaningful
- **Index references** — refer to nodes by `id`, not by position
- **Inventing entities** — only use entities extracted from sandbox files; never fabricate nodes
- **Missing entity_type** — every node MUST have entity_type for correct color assignment
- **Building piecemeal** — send the complete node/link arrays in one call; call again only to fix reported problems
- **Skipping IBD.md** — IBD.md is the primary source of block-to-block flows; BDD.md alone is not enough
- **Using BDD.md Properties as flows** — Properties are typed attributes, not flow data; flows come from IBD.md
- **Ignoring SysML files** — `physical/sysml/*.sysml` files are the validated ground truth; prefer them over .md when both exist

### File Bindings

**Consumes (reads from /workspace/current/):**
- `logical/BDD.md` — block names, subsystems, properties
- `logical/IBD.md` — connections (`PartA ─[flow]→ PartB`), ports, top-level block
- `requirements/REQUIREMENTS.md` — REQ-NNN / REQ-NF-NNN IDs (optional, for traceability/satisfaction)
- `requirements/USECASES.md` — UC-NNN IDs and actors (optional, for traceability)
- `functional/ACTIVITIES.md` — activity names and steps (optional, for activity flow)
- `requirements/BOX.md` — morphological box table (optional, for morphological box sankey)
- `logical/PRINCIPLE_SOLUTIONS.md` — selected option per function (optional)
- `physical/sysml/bdd.sysml` — validated SysML v2 block definitions
- `physical/sysml/ibd.sysml` — validated SysML v2 connections

**Produces (writes to /workspace/current/):**
- `physical/sankey.json` — SOURCE of truth: your `{title, nodes, links}` payload, saved by the tool
- `physical/sankey.html` — BUILD OUTPUT: written by the tool when the build passes; never read or edited by hand

## Fixing Problems

The tool reports every payload problem in one list (unknown node reference, zero value, self-loop, duplicate id) and writes nothing until they are all fixed. Correct them and call it again.

After a successful build, change the diagram by editing `physical/sankey.json` (one node/link per line) and rebuilding with `{"source_path": "/workspace/current/physical/sankey.json"}`. Never open `sankey.html`.

Warnings to act on:
- **nodes not found in the input artifacts** — a possible invented node. Confirm where each comes from or remove it.
- **nodes with no links** — they are invisible. Link or remove them.
- **entity_type not recognised** — the node is drawn grey.

## Learned Error Patterns

<!-- Error patterns recorded by skill_improver_tool during validation fixes.
     Each entry has a Pattern (what went wrong) and a Fix (what resolved it).
     These help future runs avoid repeating the same mistakes. -->
