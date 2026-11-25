---
name: layout-generation
description: Rules and patterns for generating Microsoft Visio .vsdx layout diagrams from MBSE entity data.
license: proprietary
---

## Purpose

Generate a Microsoft Visio `.vsdx` layout diagram from committed USS entities.
The agent queries entities and relationships from the USS Knowledge Agent,
computes layout positions, and calls the VSDX generator tool to produce a
downloadable Visio file.

## Output

The agent must call `vsdx_generator_tool` with:
- `shapes_json`: JSON array of shape objects
- `connectors_json`: JSON array of connector objects
- `output_filename`: filename for the .vsdx output

## Shape Object Schema

```json
{
  "id": "B1",
  "master": "rectangle",
  "pin_x": 2.0,
  "pin_y": 5.0,
  "width": 1.5,
  "height": 0.75,
  "text": "BrakeController",
  "fill_color": "#70AD47",
  "stereotype": "«block»",
  "font_size": 8.0,
  "bold": false
}
```

**Master shape types**: `rectangle`, `rounded_rect`, `diamond`, `circle`, `port`

## Connector Object Schema

```json
{
  "id": "C1",
  "begin_shape_id": "B1",
  "end_shape_id": "B2",
  "label": "connects",
  "connector_type": "right_angle"
}
```

**Connector types**: `right_angle` (default), `straight`, `curved`

## Color Scheme

| Entity Type | Color | Hex |
|-------------|-------|-----|
| Requirement | Blue | `#4472C4` |
| Use Case | Gold | `#FFC000` |
| Activity | Orange | `#ED7D31` |
| Logical Block | Green | `#70AD47` |
| Constraint | Purple | `#7030A0` |
| Port | Light Blue | `#5B9BD5` |
| Candidate | Gray | `#A5A5A5` |

## Rules

- Every shape MUST have a unique `id` matching the entity ID (e.g., "B1", "R1", "ACT1").
- Every shape MUST have `pin_x` and `pin_y` > 0 (positions in inches from bottom-left).
- Shapes must not overlap — leave at least 0.25" gap between shapes.
- Connectors must reference valid shape IDs in `begin_shape_id` and `end_shape_id`.
- Use `stereotype` field for SysML stereotypes: `«block»`, `«requirement»`, `«constraint»`, `«activity»`, `«use case»`.
- Use `right_angle` connectors for BDDs and IBDs; `straight` for activity flows.
- Layout should be hierarchical (top-down) for BDDs, requirements, and activities.
- Layout should be force-directed for IBDs and use case diagrams.

## Tools Available

- **uss_query_tool**: Query committed entities and relationships from the USS Knowledge Agent.
- **vsdx_generator_tool**: Generate the .vsdx file from shapes and connectors data.

## Workflow

1. Call `uss_query_tool` to fetch entities for the requested diagram type.
2. Parse the returned entities and relationships.
3. Compute layout positions for each entity (hierarchical, force-directed, or spatial).
4. Build `shapes_json` array with position, size, text, color, and stereotype for each entity.
5. Build `connectors_json` array from relationships (SATISFIES, REALIZES, FLOWS_TO, CONNECTS_TO, CONTAINS).
6. Call `vsdx_generator_tool` with the shapes and connectors data.
7. Return the output file path.

## Per-Diagram Layout Guidelines

### BDD (Block Definition Diagram)
- Top-down hierarchy: parent blocks above children
- Use `rectangle` master for blocks
- Stereotype: `«block»`
- Connectors: composition (parent→child), generalization (child→parent), association

### IBD (Internal Block Diagram)
- Force-directed layout with containment clustering
- Use `rectangle` for parts, `port` for ports on borders
- Connectors: connections between ports

### Activity Diagram
- Top-down flow: initial node at top, final node at bottom
- Use `rounded_rect` for actions, `diamond` for decisions, `circle` for start/end
- Connectors: control flow arrows

### Requirement Diagram
- Hierarchical tree: top-level requirements at top, derived below
- Use `rectangle` with `«requirement»` stereotype
- Connectors: satisfy (dashed), derive (solid)

### Use Case Diagram
- Force-directed with loose grouping
- Use `circle`/`ellipse` for use cases
- Connectors: association, include, extend

### Physical Layout (Groblayout)
- Spatial arrangement: zones as large rectangles, blocks inside zones
- Use `rectangle` for zones and blocks
- Consider spatial constraints (adjacency, separation, capacity)