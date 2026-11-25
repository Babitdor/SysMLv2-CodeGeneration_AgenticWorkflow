# draw.io Style Reference for AI File Generation

This document is a companion to the draw.io tools and provides all information
needed to programmatically generate valid draw.io (.drawio) files with proper styling.
All data was extracted from the draw.io source code.

---

## 1. File Structure Overview

A minimal valid draw.io file:

```xml
<mxfile>
  <diagram id="page-1" name="Page-1">
    <mxGraphModel dx="0" dy="0" grid="1" gridSize="10" guides="1"
                  tooltips="1" connect="1" arrows="1" fold="1"
                  page="1" pageScale="1" pageWidth="850" pageHeight="1100"
                  math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <!-- Diagram elements: parent="1" -->
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

### Critical Rules
1. `id="0"` — root container, no parent. `id="1"` with `parent="0"` — default layer. Always required.
2. All diagram elements must have `parent="1"` (or parent of a group/layer).
3. IDs must be unique within the diagram.
4. Vertices need `vertex="1"`, edges need `edge="1"` — mutually exclusive.
5. Style strings end with semicolon: `"rounded=1;whiteSpace=wrap;html=1;"`
6. Coordinates: origin (0,0) is top-left, x increases rightward, y increases downward.

---

## 2. Color Themes (use as bare tokens in style strings)

| Class | fillColor | strokeColor | Effect |
|-------|-----------|-------------|--------|
| `blue` | #DAE8FC | #6C8EBF | Shadow + glass |
| `green` | #D5E8D4 | #82B366 | Shadow + glass |
| `yellow` | #FFF2CC | #D6B656 | Shadow + glass |
| `orange` | #FFCD28 | #D79B00 | Shadow + glass |
| `red` | #F8CECC | #B85450 | Shadow + glass |
| `purple` | #E1D5E7 | #9673A6 | Shadow + glass |
| `pink` | #E6D0DE | #996185 | Shadow + glass |
| `gray` | #F5F5F5 | #666666 | Shadow + glass |
| `turquoise` | #D5E8D4 | #67AB9F | Shadow + glass |

`plain-*` variants (e.g. `plain-blue`) set the same colors without shadow/glass.

## Warehouse Floor-Plan Colour Mapping

Anchored to a production reference layout (Behälterlager / Palettenlager /
Gefahrstofflager — German logistics floor-plan). Pass these names directly
as `fill_color` to `drawio_add_shape`; the tool resolves them to the hex
codes below.

| Zone preset | Hex | Typical use |
|-------------|-----|-------------|
| `storage` | `#DAE8FC` | Container / tote storage (Behälterlager) |
| `pallet_storage` | `#D5E8D4` | Pallet racking (Palettenlager) |
| `automated` | `#D5E8D4` | Automated narrow-aisle (autom. Schmalganglager) |
| `hazmat` | `#FFCD28` | Hazmat storage (Gefahrstofflager) — same orange as the building wall |
| `cold` | `#DAE8FC` | Cold / Tiefkühl storage |
| `consol` | `#E1D5E7` | Consolidation (AP Konsolidierung) |
| `picking` | `#E1D5E7` | Picking |
| `qc` | `#E6D0DE` | Quality control (small pink callout) |
| `buffer` / `vorzone` | `#D5E8D4` | Buffer / Vorzone |
| `receiving` / `shipping` / `we_wa` | `#FFE6CC` | Goods in / out, combined WE & WA dock |
| `loading` | `#F5F5F5` | Loading area |
| `expansion` | `#BAC8D3` | Planned expansion ("Erweiterung auf …") |
| `external` | `#F5F5F5` | Truck docks outside the hall — combine with `external=true` |

The building outline itself is **thick orange** (`#FFCD28`, 14 px stroke) by
default — `hazmat` zones use the same hue intentionally to signal continuity
between hazmat enclosures and the structural wall.

## Office / Admin Colour Mapping (kept for non-warehouse use)

| Zone | Recommended Class |
|------|------------------|
| Office / Admin | `plain-blue` |
| Meeting | `plain-green` |
| Hallway | `plain-gray` |
| Bathroom | `plain-purple` |
| Kitchen / Break Room | light orange |
| Server Room | `plain-red` |
| Reception / Lobby | `plain-pink` |

---

## 3. Shape Types

| shape= value | Description | Notes |
|---|---|---|
| `rectangle` | Rectangle (default) | |
| `ellipse` | Oval | Use `aspect=fixed` for circle |
| `rhombus` | Diamond | Use `perimeter=rhombusPerimeter` |
| `triangle` | Triangle | |
| `hexagon` | Hexagon | Use `perimeter=hexagonPerimeter2` |
| `cylinder3` | 3D Cylinder | For tanks, vessels |
| `document` | Document page | |
| `parallelogram` | Parallelogram | Use `perimeter=parallelogramPerimeter` |
| `cube` | 3D cube | |
| `note` | Sticky note | |
| `swimlane` | Container with header | Use `startSize` for header height |

### Floorplan-Specific Shapes

| shape= value | Use for |
|---|---|
| `rectangle` with `rounded=1` | Rooms, zones, offices |
| `rectangle` with `arcSize=50` | Doorways |
| `ellipse` with `aspect=fixed` | Columns, pillars |
| `swimlane;startSize=26` | Building wing / floor container |
| `text;html=1;` | Room labels, dimensions |

---

## 4. Style Properties Quick Reference

### Fill and Stroke
| Property | Values |
|----------|--------|
| `fillColor` | `#RRGGBB`, `none`, `default` |
| `strokeColor` | `#RRGGBB`, `none` |
| `strokeWidth` | number (px) |
| `dashed` | `0`, `1` |
| `opacity` | `0`–`100` |
| `rounded` | `0`, `1` |
| `arcSize` | `0`–`50` (corner radius %) |

### Text and Labels
| Property | Values |
|----------|--------|
| `fontSize` | number (px, default 12) |
| `fontFamily` | font name |
| `fontColor` | `#RRGGBB` |
| `fontStyle` | `0`=normal, `1`=bold, `2`=italic |
| `align` | `left`, `center`, `right` |
| `verticalAlign` | `top`, `middle`, `bottom` |
| `whiteSpace` | `wrap` |

### Edge / Connector
| Property | Values |
|----------|--------|
| `edgeStyle` | `orthogonalEdgeStyle` (most common) |
| `endArrow` | `classic`, `block`, `open`, `none` |
| `startArrow` | `classic`, `block`, `open`, `none` |
| `curved` | `0`, `1` |
| `dashed` | `0`, `1` |

---

## 5. Warehouse Floor-Plan Pattern

The production reference layout (warehouse with Behälterlager, Palettenlager,
Gefahrstofflager, WE & WA, etc.) is built from **six recurring elements**.
Reproduce these for any industrial / logistics floor-plan request.

### 5.1 Building outline (thick perimeter wall)

A filled-none rectangle with a heavy stroke. Drawn FIRST so rooms sit on
top of it. Use `drawio_add_building_outline` — no nesting / no swimlane;
the cell graph stays flat so dimension arrows and external annexes can
live alongside the building.

```xml
<mxCell id="bldg" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#FFCD28;strokeWidth=14;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="900" height="700" as="geometry"/>
</mxCell>
```

### 5.2 Multi-line zone label (name + capacity + area)

Production drawings stack three lines per zone: bold name (top), parenthesised
capacity (middle), area in m² (bottom). `drawio_add_shape` composes these
automatically from `label`, `capacity`, `area_qm`:

```xml
<mxCell id="r1" value="Behälterlager<br/><font style=&quot;font-size: 11px&quot;>(OLS - 34.320)</font><br/><font style=&quot;font-size: 11px&quot;>924 qm</font>"
        style="rounded=0;whiteSpace=wrap;html=1;fillColor=#DAE8FC;strokeColor=#23445d;fontSize=13;fontStyle=1;align=center;verticalAlign=middle;"
        vertex="1" parent="1">
  <mxGeometry x="200" y="200" width="280" height="200" as="geometry"/>
</mxCell>
```

### 5.3 Vertical text (tall narrow zones)

For zones where height > 2 × width (e.g. Gefahrstofflager, WE & WA), set
`horizontal=0;` to rotate the label 90° CCW. The geometry stays upright;
only the text rotates. Pass `label_orientation="vertical"` to the tool.

```xml
<mxCell id="haz" value="Gefahrstofflager<br/><font style=&quot;font-size: 11px&quot;>296 qm</font>"
        style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFCD28;strokeColor=#D79B00;fontSize=13;fontStyle=1;horizontal=0;align=center;verticalAlign=middle;"
        vertex="1" parent="1">
  <mxGeometry x="120" y="200" width="70" height="320" as="geometry"/>
</mxCell>
```

### 5.4 Subzone (dashed inner allocation)

Used for sub-allocations inside a parent zone — e.g. "Konsi Apotheke 300 qm"
sitting inside "AP Konsolidierung", or "Erweiterung auf 48.048 Plätze"
inside "Behälterlager". Renders as a dashed red border with no fill, so
the parent's colour shows through.

```xml
<mxCell id="sub" value="Konsi Apotheke<br/><font style=&quot;font-size: 11px&quot;>300 qm</font>"
        style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#B85450;strokeWidth=1.5;dashed=1;dashPattern=6 3;fontSize=11;align=center;verticalAlign=middle;"
        vertex="1" parent="1">
  <mxGeometry x="230" y="430" width="160" height="120" as="geometry"/>
</mxCell>
```

Pass `shape_type="subzone"` to `drawio_add_shape`.

### 5.5 External annex (truck docks)

For infrastructure that sits OUTSIDE the building outline — loading
ramps, container parking, external WE-Rampen. Dashed border + 60%
opacity dims them visually relative to the main hall. Pass
`external=true` to `drawio_add_shape`.

```xml
<mxCell id="dock1" value="WE-Rampen<br/><font style=&quot;font-size: 10px&quot;>(Apotheke / Pakete)</font>"
        style="rounded=0;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#666666;opacity=60;dashed=1;dashPattern=4 2;fontSize=10;align=center;"
        vertex="1" parent="1">
  <mxGeometry x="1030" y="180" width="120" height="80" as="geometry"/>
</mxCell>
```

### 5.6 Dimension arrow (double-headed) outside the building

Draw building extents OUTSIDE the perimeter wall — horizontal arrow
below the building, vertical arrow to the left. Use `drawio_add_dimension`
with the locale-formatted label (e.g. `"77.500 mm"` with a period as
the thousands separator). The tool detects horizontal vs vertical
orientation from the endpoints and rotates vertical labels automatically.

```xml
<!-- Horizontal extent (below the building) -->
<mxCell id="dimW" style="endArrow=classic;startArrow=classic;html=1;rounded=0;strokeColor=#000000;strokeWidth=1.5;" edge="1" parent="1">
  <mxGeometry relative="1" as="geometry">
    <mxPoint x="100" y="850" as="sourcePoint"/>
    <mxPoint x="1000" y="850" as="targetPoint"/>
  </mxGeometry>
</mxCell>
<mxCell id="dimW-label" value="77.500 mm" style="text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;" vertex="1" parent="1">
  <mxGeometry x="490" y="858" width="120" height="24" as="geometry"/>
</mxCell>
```

### 5.7 Layered render order

The cells in `<root>` render in document order — first cell sits at the
bottom. For warehouse layouts emit cells in this sequence:

1. **Building outline** (orange ring at z=0)
2. **Hazmat / structural fills** that line the wall on the inside
3. **Main zones** (storage, pallet, automated, consol, WE & WA, …)
4. **Subzones** (dashed allocations on top of their parent)
5. **External annexes** (truck docks outside the wall)
6. **Dimension arrows + labels** (always last, drawn on top)

`drawio_add_*` tools append to `<root>` in call order — call them in the
sequence above.

---

## 6. Validation Checklist

1. Valid XML with proper escaping
2. `<mxfile>` root containing at least one `<diagram>`
3. `id="0"` and `id="1" parent="0"` structural cells present
4. All cell IDs unique within diagram
5. Every cell has valid `parent` reference
6. Vertices have `mxGeometry` with x, y, width, height
7. Edges have `relative="1"` as geometry
8. Style strings use `key=value;` format
9. HTML in values is XML-escaped (`&lt;`, `&gt;`, `&amp;`, `&quot;`)
