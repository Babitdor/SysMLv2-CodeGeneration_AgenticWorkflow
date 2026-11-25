---
name: floorplan
description: Render the warehouse layout from @Optimization's layout_v1.json as a production-grade draw.io floor-plan diagram (.drawio + optional PNG export). Covers the building outline, multi-line zone labels with capacity and area, vertical text for tall narrow zones (Gefahrstofflager, WE & WA), dashed sub-allocations (Konsi Apotheke, Erweiterung), external truck-dock annexes, and double-headed dimension arrows. Use when the user asks for a warehouse, logistics, or industrial floor-plan rendered with draw.io tooling.
license: proprietary
---

# Floor-Plan Skill (draw.io)

Render the warehouse layout from `@Optimization`'s `layout_v1.json` as a **draw.io floor plan diagram** that respects the optimisation's spatial coordinates exactly. The output is a `.drawio` file (openable in draw.io, draw.io Desktop, or any compatible editor) with an optional PNG export.

This is the cross-platform counterpart to `@Layout` (which uses Microsoft Visio on Windows).

## Style Reference

See `reference/STYLE_GUIDE_DRAWIO.md` in the floorplan directory for the warehouse colour palette, multi-line label syntax, vertical-text rotation, building-outline + dimension-arrow patterns, and the layered render order. Read it once at the start of a session; the conventions below summarise the parts you'll use most.

## Available tools

The drawio tools build the diagram in-process memory; none of them write to disk in the sandbox. After you're done adding shapes, `drawio_save_diagram` returns the full XML and YOU persist it via `write_file` to `/workspace/project_<id>/physical/floorplan.drawio` — the artifact viewer renders any `.drawio` file there as an interactive pan/zoom canvas.

### Hard rule — these tools are the ONLY way to produce a `.drawio` file

Do NOT hand-craft draw.io XML in a Python script (`execute_command python -c "..."`), a heredoc, or any string-concatenation in `write_file`. The format has many small traps (raw HTML in `value=`, geometry as attributes not JSON, mxCell parent chain, root envelope) that the tools handle and that hand-rolled XML reliably gets wrong. If the artifact viewer ever shows only the building outline (a yellow border), the fix is to **re-call the drawio_* tools for the missing zones**, not to bypass them.

| Tool                            | Purpose                                                                       |
|---------------------------------|-------------------------------------------------------------------------------|
| `drawio_create_diagram`         | Create a new in-memory `.drawio` canvas. Call FIRST.                          |
| `drawio_add_building_outline`   | Thick perimeter wall (default orange, 14 px stroke). Call SECOND.             |
| `drawio_add_shape`              | Add a zone / room / subzone / truck-dock. Returns the shape ID.               |
| `drawio_add_edge`               | Connector between two shapes by their IDs. Rarely used in warehouse plans.    |
| `drawio_add_text`               | Floating annotation (titles, notes).                                          |
| `drawio_add_dimension`          | Double-headed arrow with a label — building extents, outside the wall.        |
| `drawio_save_diagram`           | Finalise: returns the full XML as the tool result. Then `write_file` it.      |
| `drawio_export_diagram`         | Optional PNG/SVG/PDF render via the draw.io Desktop CLI (if installed).       |

## JSON field → tool arg mapping

The optimisation JSON already carries everything the renderer needs. Map each block field 1:1:

| `layout_v1.json` field              | Goes to                                                                  |
|-------------------------------------|--------------------------------------------------------------------------|
| `name`                              | `drawio_add_shape(label=...)`                                            |
| `source_refs`                       | `drawio_add_shape(source_refs=...)` — copy artifact/element references unchanged into XML metadata |
| `capacity`                          | `drawio_add_shape(capacity=...)` — second label line (auto-sized)        |
| `area_qm`                           | `drawio_add_shape(area_qm="924 qm")` — third label line (auto-sized)     |
| `zone`                              | `drawio_add_shape(fill_color=zone)` — preset name resolves to hex        |
| `label_orientation: "vertical"`     | `drawio_add_shape(label_orientation="vertical")`                         |
| `external: true`                    | `drawio_add_shape(external=True)` — dashed dimmed annex                  |
| `subzones[]`                        | One `drawio_add_shape(shape_type="subzone", ...)` per entry              |
| `building.{x,y,width,height}`       | `drawio_add_building_outline(...)`                                       |
| `building.dim_label_width`          | `drawio_add_dimension(label="77.500 mm")` — horizontal, below the wall   |
| `building.dim_label_height`         | `drawio_add_dimension(label="74.300 mm")` — vertical, left of the wall   |

## Shape types

| `shape_type`       | Use for                                                  |
|--------------------|----------------------------------------------------------|
| `room`             | Indoor zones (storage, pallet, consol, picking, …)       |
| `subzone`          | Dashed inner allocation (Konsi Apotheke, Erweiterung)    |
| `truck_dock`       | Dimmed external annex (WE-Rampen, container parking)     |
| `rectangle`        | Generic / fallback                                       |
| `wall`             | Internal partitions inside the building outline          |
| `door`             | Doorways between zones                                   |
| `window`           | Windows on outer walls                                   |
| `staircase`        | Stairs (parallel line pattern)                           |

For external annexes the cleaner alternative is `shape_type="room"` + `external=true` — gives you the dashed + 60% opacity styling on any base shape.

## Warehouse fill colour presets

| Preset             | Hex       | Use for                                          |
|--------------------|-----------|--------------------------------------------------|
| `storage`          | `#DAE8FC` | Container / tote storage (Behälterlager)         |
| `pallet_storage`   | `#D5E8D4` | Pallet racking (Palettenlager)                   |
| `automated`        | `#D5E8D4` | Automated narrow-aisle (Schmalganglager)         |
| `hazmat`           | `#FFCD28` | Hazmat (Gefahrstofflager) — matches wall colour  |
| `cold`             | `#DAE8FC` | Cold / Tiefkühl storage                          |
| `consol`           | `#E1D5E7` | Consolidation (AP Konsolidierung)                |
| `picking`          | `#E1D5E7` | Picking                                          |
| `qc`               | `#E6D0DE` | Quality control (small pink callout)             |
| `buffer` / `vorzone` | `#D5E8D4` | Buffer zone                                     |
| `receiving` / `shipping` / `we_wa` | `#FFE6CC` | Goods-in / -out, combined dock     |
| `expansion`        | `#BAC8D3` | Planned "Erweiterung auf …" expansion area       |
| `external`         | `#F5F5F5` | Truck docks outside the hall                     |

Any hex colour (e.g. `#ff8844`) also works. Office-style presets (`office`, `meeting`, `bathroom`, `kitchen`, etc.) are still available for non-warehouse layouts.

## Scale factor

The accepted layout JSON owns geometry. Do not derive new geometry from .sysml.
Preserve each block/subzone's source_refs; pass connection source_refs through
drawio_add_edge too. These references are provenance, not validation evidence.
Check recorded traceability.source_checks hashes before accepting the rendering;
stale or conflicting sources return to Optimization. Legacy layouts without
references remain renderable with source review reported as unavailable.
When editing XML, preserve source_refs and the layout-to-pixel mapping. A visual
layout defect must be corrected in Optimization before rendering new coordinates.

The layout JSON uses real-world units (mm / m / ft). Scale them to a bounded, readable canvas with ONE factor so you never get giant shapes with unreadable text:

- `S = 1200 / max(building.width, building.height)` (in the JSON's own unit).
- Multiply EVERY coordinate and size — building, blocks, subzones, dimension-arrow endpoints — by `S`, then round to integers.
- Dimension **labels** keep their real-world strings (`"77.500 mm"`); only the arrow *coordinates* are scaled.

Fonts are auto-sized to each shape by the tools, so **do not pass `font_size`** — label text (and the capacity/area sub-lines) scales with the zones and stays legible at any diagram scale. Set the canvas in `drawio_create_diagram` to the scaled building extent + ~300 px padding (room for dimension arrows outside the wall and external annexes to the right).

## Workflow (in this exact order — see STYLE_GUIDE §5.7)

1. **Read** `/workspace/project_*/physical/layout/layout_v1.json`. If missing or its variants are empty, STOP and report that `@Optimization` must run first.
2. **Pick a variant** — by default `variants[0]` unless the user explicitly asks for another.
3. **Scale coordinates** from the JSON units to pixels.
4. **Create the canvas** (`drawio_create_diagram`) with width / height = scaled building extent + 300 px padding.
5. **Building outline FIRST** (`drawio_add_building_outline`) — at the scaled `building.{x,y,width,height}`. Use defaults (orange, 14 px).
6. **Main zones** — for each non-external block, one `drawio_add_shape(shape_type="room", ...)` call, passing through `label`/`capacity`/`area_qm`/`fill_color`/`label_orientation`. SAVE every returned shape ID.
7. **Subzones** — for each block that has `subzones`, emit one `drawio_add_shape(shape_type="subzone", ...)` per entry, AFTER its parent zone so it renders on top.
8. **External annexes** — for blocks flagged `external: true`, call `drawio_add_shape(external=True)` (use `shape_type="room"` or `"truck_dock"`).
9. **Dimensions LAST** — `drawio_add_dimension` for the two building extents. Place them ~40 px outside the wall (below for width, left for height). Use `building.dim_label_width` / `dim_label_height` verbatim so locale formatting (`77.500 mm`, not `77500 mm`) survives.
10. **Save (get the XML)** — `drawio_save_diagram(name="floorplan")` returns the assembled `.drawio` XML inside a fenced ```xml block in its tool result. Read the XML from the result.
11. **Persist to the sandbox (mandatory)** — call `write_file` with path `/workspace/project_<project_id>/physical/floorplan.drawio` and the XML extracted from step 10 as the content. The artifact viewer renders any `.drawio` file at that path interactively. **Without this step the floor plan is invisible** — the drawio tools cannot reach your sandbox themselves (same separation as Sankey).
12. **Export PNG (optional)** — `drawio_export_diagram("floorplan.png")` shells out to the draw.io Desktop CLI for a rendered PNG. Graceful no-op if the CLI is missing; the `.drawio` artifact from step 11 already renders interactively.

## Mapping conventions

- Every block in `variant.blocks` must become exactly one `drawio_add_shape` call.
- Every `subzones[]` entry must become exactly one additional `drawio_add_shape(shape_type="subzone")` call AFTER its parent.
- The `building` rectangle becomes exactly one `drawio_add_building_outline` call.
- `connections` (when present) become `drawio_add_edge` calls using the SAVED shape IDs from step 6.
- Coordinates share the same origin across the whole variant — subzone coordinates are NOT relative to their parent block.

## Output reporting

```
Variant rendered: <variant name>
Blocks:           <N>
Subzones:         <S>
Dimensions:       <D>
Saved to:         physical/floorplan.drawio
Export:           physical/floorplan.png (or "not available — draw.io CLI not installed")
```

## Error patterns

- *"No .drawio file found"* — you called `drawio_add_shape` (or any other add tool) before `drawio_create_diagram`. Always create the canvas first.
- *"shape ID not found"* — you're trying to connect shapes using IDs from a previous session. Re-create the diagram fresh.
- *"draw.io Desktop CLI not found"* — export unavailable; the `.drawio` file is still valid and the artifact viewer can render it.
- *"Invalid .drawio file"* — the file was corrupted. Delete it and re-create with `drawio_create_diagram`.
