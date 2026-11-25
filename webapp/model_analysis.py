"""Traceability, impact and review over a model index.

Pure functions over a ModelIndex: no ANTLR, no clock, no I/O, so a hand-built
index tests the analysis without a parser. Findings that come from the *absence*
of a link are worded as things to check, never as validated facts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from webapp.model_index import (
    REQUIREMENT_KINDS,
    ModelEdge,
    ModelElement,
    ModelIndex,
    ModelIssue,
    build_index,
)

# A change fans out quickly; both bounds keep one request's answer small.
IMPACT_DEPTH = 6
IMPACT_NODES = 200
MAX_FINDINGS = 400

# Edges that wire an element into the structure, as opposed to typing it.
_LINK_KINDS = frozenset({"connector_end", "flow_end", "perform_action"})


def _summary(element: ModelElement, edge: Optional[ModelEdge] = None) -> Dict[str, object]:
    row: Dict[str, object] = {
        "id": element.id,
        "name": element.name,
        "kind": element.kind,
        "line": element.line,
    }
    if edge is not None:
        row["via"] = edge.kind
    return row


def traceability(index: ModelIndex) -> Dict[str, object]:
    """Requirements, and whether anything in the model claims to satisfy them.

    Only satisfy links count. The grammar exposes "verify" as a token with no
    listener rule, so verification links cannot be resolved yet and are not
    reported as missing.
    """
    satisfied = {edge.target for edge in index.edges if edge.kind == "satisfy"}
    requirements: List[Dict[str, object]] = []
    unlinked: List[Dict[str, object]] = []
    for element in index.elements:
        if element.kind not in REQUIREMENT_KINDS:
            continue
        linked = element.id in satisfied
        requirements.append({**_summary(element), "satisfied": linked})
        if not linked:
            unlinked.append({**_summary(element), "satisfied": False})
    return {
        "requirements": requirements,
        "unlinked": unlinked,
        "counts": {
            "requirements": len(requirements),
            "satisfied": len(requirements) - len(unlinked),
            "unlinked": len(unlinked),
        },
    }


def impact(index: ModelIndex, element_id: str) -> Dict[str, object]:
    """What would notice a change to this element: reverse reachability.

    Walks the edges that point *at* the element (typing, specialization,
    subsetting, connector and flow ends, perform, satisfy) so "who depends on
    this" is one call. Raises KeyError for an id the index does not have.
    """
    by_id = {element.id: element for element in index.elements}
    if element_id not in by_id:
        raise KeyError(element_id)

    incoming: Dict[str, List[ModelEdge]] = {}
    for edge in index.edges:
        if edge.resolved:
            incoming.setdefault(edge.target, []).append(edge)

    direct: List[Dict[str, object]] = []
    transitive: List[Dict[str, object]] = []
    seen = {element_id}
    frontier = [element_id]
    truncated = False
    full = False

    for depth in range(1, IMPACT_DEPTH + 1):
        reached: List[str] = []
        for target in frontier:
            for edge in incoming.get(target, ()):
                if edge.source in seen:  # also what makes a cycle terminate
                    continue
                if len(direct) + len(transitive) >= IMPACT_NODES:
                    truncated = full = True
                    break
                source = by_id.get(edge.source)
                if source is None:  # an edge to an id the index does not list
                    continue
                seen.add(edge.source)
                (direct if depth == 1 else transitive).append(_summary(source, edge))
                reached.append(edge.source)
            if full:
                break
        if full or not reached:
            break
        frontier = reached

    affected = direct + transitive
    return {
        "direct": direct,
        "transitive": transitive,
        "requirements": [row for row in affected if row["kind"] in REQUIREMENT_KINDS],
        "truncated": truncated,
    }


def review(index: ModelIndex) -> List[ModelIssue]:
    """Absence-of-evidence findings, plus the validator's own issues.

    Every finding here is a "check this" hint: no satisfy link and possibly
    disconnected mean the model carries no evidence, not that it is wrong.
    """
    issues: List[ModelIssue] = list(index.issues)
    satisfied = {edge.target for edge in index.edges if edge.kind == "satisfy"}
    wired = {
        edge.target for edge in index.edges if edge.kind in _LINK_KINDS and edge.resolved
    }

    for edge in index.edges:
        if edge.resolved:
            continue
        issues.append(
            ModelIssue(
                severity="error",
                message=f"{edge.label} does not resolve to anything in the model.",
                line=edge.line,
                rule="reference",
                element=edge.source,
            )
        )

    for element in index.elements:
        if element.kind in REQUIREMENT_KINDS and element.id not in satisfied:
            issues.append(
                ModelIssue(
                    severity="warning",
                    message=(
                        f"{element.name} has no satisfy link: "
                        "nothing in the model claims it."
                    ),
                    line=element.line,
                    rule="traceability",
                    element=element.id,
                )
            )

    for element in index.elements:
        if element.kind != "part_usage" or element.id in wired:
            continue
        issues.append(
            ModelIssue(
                severity="warning",
                message=(
                    f"{element.name} is possibly disconnected: "
                    "no connector, flow or perform reaches it."
                ),
                line=element.line,
                rule="consistency",
                element=element.id,
            )
        )
    return issues[:MAX_FINDINGS]


def report(code: str) -> Dict[str, object]:
    """One call for the endpoint: the index, its review and its traceability."""
    index = build_index(code)
    return {
        **index.model_dump(),
        "review": [issue.model_dump() for issue in review(index)],
        "traceability": traceability(index),
    }
