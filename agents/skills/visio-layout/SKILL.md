---
name: visio-layout
description: Convert a warehouse layout_v1.json (from @Optimization) into a styled Microsoft Visio diagram (.vsdx + PNG). Covers block placement, material flow, and zone boundaries on a single landscape page. Use when the user wants a Visio rendering of the warehouse layout — Windows-only, requires Microsoft Visio + pywin32.
license: proprietary
---

# Visio Layout Skill

Convert a structured warehouse `layout_v1.json` (produced by `@Optimization`) into a styled Microsoft Visio diagram. The diagram visualises block placement, material flow, and zone boundaries on a single landscape page.

> **Platform constraint** — Visio COM automation only runs on Windows with Microsoft Visio installed. When the backend runs in a Linux container, the Visio tools return `ERROR: Visio tools require Windows with Microsoft Visio + pywin32 installed.` Run the backend on the host (`python main.py`) when producing diagrams.

> The canonical style reference lives alongside this file at `reference/STYLE_GUIDE.md` — `read_file('/skills/visio-layout/reference/STYLE_GUIDE.md', limit=1000)` to load it. Anything below restates the parts the agent needs at generation time.

## Available tools

| Tool                  | Purpose                                                                    |
|-----------------------|----------------------------------------------------------------------------|
| `visio_create_diagram` | Initialise a new `.vsdx` file (landscape 11 × 8.5 in). **Call first.**     |
| `visio_add_shape`     | Drop a styled shape; returns the shape's numeric ID — **save it**.         |
| `visio_add_connector` | Connect two shape IDs with an optional label and style.                    |
| `visio_set_text`      | Update text on an existing shape by ID.                                    |
| `visio_export_diagram` | Export the active page to PNG / SVG / JPG (extension picks the format).   |
| `visio_save_diagram`  | Save the active document under a chosen name.                              |

## Workflow

1. Read `/workspace/project_*/physical/layout/layout_v1.json`. If missing, STOP and report that `@Optimization` must run first.
2. Pick ONE variant from the JSON to render — by default the first (`variants[0]`).
3. Call `visio_create_diagram(name="layout_v1")`. This creates the document and the active page.
4. **Building outline** — if the variant has a `building` object, draw it FIRST as a thick-bordered rectangle (use `visio_add_shape(shape_type="rectangle", ...)`, then if Visio exposes it set a thicker line weight). This establishes the outer hall before any zones are placed inside.
5. **Zones** — for each block in the variant:
   - Call `visio_add_shape(shape_type="rectangle", x, y, width, height, text=<multi-line label>, fill_color=<zone colour>)`. Save the returned ID.
   - Build the label as `"<name>\n<capacity>\n<area_qm> qm"` — capacity and area lines are optional, include only when present in the JSON.
   - For blocks with `external: true`, use a muted fill colour and a smaller text size (Visio doesn't auto-dim like the Plotly tool does).
   - For blocks with `label_orientation: "vertical"`, rotate the text shape 90° (set `Cells("Angle")` if possible, otherwise live with horizontal text and accept the cramping).
6. **Subzones** — for every `subzones` entry inside a block, draw a SEPARATE dashed rectangle on top: `visio_add_shape(shape_type="rectangle", ...)` with no fill, then set `LinePattern = 2` (dashed). Subzone coordinates share the same origin as the parent block, NOT relative coords.
7. **Dimension annotations** — if `building.show_dimensions` is true, draw two arrow shapes outside the building rectangle (one along the bottom edge for width, one along the left edge for height) labelled with `building.dim_label_width` / `building.dim_label_height` (or the auto-formatted value if no override).
8. **Connections** (optional) — for each `connections` entry, call `visio_add_connector` with the source and target IDs from step 5.
9. Call `visio_export_diagram(file_name="layout_v1.png")` so the user has a static image they can preview.
10. The `.vsdx` is auto-saved by `visio_create_diagram`. Call `visio_save_diagram` again only if you change the diagram after the initial create.

> The cross-platform counterpart `@Floorplan` consumes the same `layout_v1.json` and produces an interactive HTML floor plan. Both agents read the same enriched schema (capacity, area_qm, subzones, external, label_orientation, building.dim_label_*) — keep the visual treatment in sync where Visio permits.

## Style essentials

- **Page**: landscape 11 × 8.5 in (handled by `visio_create_diagram`).
- **Shapes**: rounded corners 0.06 in and 15% transparent fills (applied automatically by `visio_add_shape`).
- **Connectors**: filled triangle arrowheads, 1 pt line weight, 0.15 in rounding (applied automatically).
- **Colours** — use named palette entries: `azure_blue`, `dark_blue`, `teal`, `orange`, `purple`, `green`, `red`.
- **Dashed lines** for failover or secondary paths: pass `dashed=True` to `visio_add_connector`.
- **Bidirectional arrows** for replication links: `bidirectional=True`.
- **Top-to-bottom flow** for receiving → storage → picking → shipping; **left/right symmetry** for redundant zones.

## Mapping `layout_v1.json` to shapes

The enriched schema produced by `@Optimization` (shared with `@Floorplan`):

```json
{
  "variants": [
    {
      "name": "Variant A",
      "unit": "m",
      "building": {
        "x": 0, "y": 0, "width": 77.5, "height": 74.3,
        "show_dimensions": true,
        "dim_label_width":  "77.500 mm",
        "dim_label_height": "74.300 mm"
      },
      "blocks": [
        {"name": "Behälterlager", "x": 12, "y": 30, "width": 30, "height": 22,
         "zone": "storage", "capacity": "OLS - 34.320", "area_qm": 924,
         "subzones": [
           {"name": "Erweiterung auf 48.048 Plätze",
            "x": 13.5, "y": 41.5, "width": 22, "height": 4.5, "area_qm": 300}
         ]},
        {"name": "WE-Rampe (Apotheke)", "x": 80, "y": 20, "width": 5, "height": 4,
         "zone": "external", "external": true}
      ],
      "connections": []
    }
  ]
}
```

Map every enriched field even when Visio can't render it natively — Visio's text + style mechanisms still cover most cases (multi-line text via `\n`; dashed lines via `LinePattern`; subzones as separate dashed rectangles).

- One `visio_add_shape` per block; pick `rectangle` for storage/process zones and `rounded_rectangle` for endpoints.
- One `visio_add_connector` per connection; the label becomes the connector text.
- Use `fill_color` to colour-code by zone — keep the mapping in your reasoning so the same zone always reads the same colour.

## Output reporting

When done, give the user a 2-line summary:
- `Variant rendered: <name>`
- `Saved to: <path returned by visio_create_diagram>` and `Exported: <PNG path>`

Do **not** dump tool JSON. Synthesise the diagram contents into the summary.

## Error patterns

- *"No active Visio page"* — you forgot `visio_create_diagram`; call it first.
- *"Shape ID N not found"* — you didn't capture the ID returned by `visio_add_shape`; rebuild your ID dictionary before calling `visio_add_connector`.
- *"Visio tools require Windows…"* — the backend is running in a Linux container; tell the user this agent needs the backend running on the Windows host.
