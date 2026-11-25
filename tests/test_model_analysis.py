"""Traceability, impact and review, on hand-built indices.

These are the answers the feature gives, so they are tested without a parser: a
hand-built index is the only way to pin down a three-hop closure, a cycle and an
unknown id without depending on grammar details.
"""

from __future__ import annotations

import pytest

from webapp.model_analysis import IMPACT_NODES, impact, review, traceability
from webapp.model_index import ModelEdge, ModelElement, ModelIndex, ModelIssue


def element(element_id: str, kind: str, line: int = 1) -> ModelElement:
    return ModelElement(
        id=element_id,
        name=element_id.split("::")[-1],
        kind=kind,
        qualified_name=element_id,
        parent=element_id.rsplit("::", 1)[0] if "::" in element_id else None,
        line=line,
        end_line=line,
    )


def edge(source: str, target: str, kind: str, resolved: bool = True) -> ModelEdge:
    return ModelEdge(source=source, target=target, kind=kind, label=target, resolved=resolved)


def index(elements, edges=(), issues=()) -> ModelIndex:
    return ModelIndex(
        revision="test",
        elements=list(elements),
        edges=list(edges),
        issues=list(issues),
    )


# --- traceability ----------------------------------------------------------


def test_a_requirement_no_satisfy_edge_targets_is_unlinked():
    model = index(
        [element("BrakeReq", "requirement_def"), element("brake", "requirement_usage")],
        [edge("vehicle", "BrakeReq", "satisfy")],
    )
    trace = traceability(model)
    assert [row["id"] for row in trace["requirements"]] == ["BrakeReq", "brake"]
    assert [row["id"] for row in trace["unlinked"]] == ["brake"]
    assert trace["counts"] == {"requirements": 2, "satisfied": 1, "unlinked": 1}


def test_an_unlinked_requirement_is_a_traceability_finding():
    model = index([element("brake", "requirement_usage", line=7)])
    findings = [issue for issue in review(model) if issue.rule == "traceability"]
    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert findings[0].element == "brake"
    assert "no satisfy link" in findings[0].message


def test_a_satisfied_requirement_is_not_a_finding():
    model = index(
        [element("brake", "requirement_usage")],
        [edge("vehicle", "brake", "satisfy")],
    )
    assert not [issue for issue in review(model) if issue.rule == "traceability"]


# --- impact ----------------------------------------------------------------


def test_impact_walks_three_hops_backwards():
    model = index(
        [
            element("Wheel", "part_def"),
            element("w1", "part_usage"),
            element("car", "part_usage"),
            element("fleet", "part_usage"),
        ],
        [
            edge("w1", "Wheel", "type_ref"),
            edge("car", "w1", "type_ref"),
            edge("fleet", "car", "type_ref"),
        ],
    )
    reply = impact(model, "Wheel")
    assert [row["id"] for row in reply["direct"]] == ["w1"]
    assert {row["id"] for row in reply["transitive"]} == {"car", "fleet"}
    assert reply["truncated"] is False


def test_impact_reports_the_edge_it_travelled():
    model = index(
        [element("Wheel", "part_def"), element("w1", "part_usage")],
        [edge("w1", "Wheel", "type_ref")],
    )
    assert impact(model, "Wheel")["direct"][0]["via"] == "type_ref"


def test_impact_surfaces_the_requirements_in_the_closure():
    # Changing the requirement definition reaches the usage that types it.
    model = index(
        [element("BrakeReq", "requirement_def"), element("brake", "requirement_usage")],
        [edge("brake", "BrakeReq", "type_ref")],
    )
    reply = impact(model, "BrakeReq")
    assert [row["id"] for row in reply["direct"]] == ["brake"]
    assert [row["id"] for row in reply["requirements"]] == ["brake"]


def test_impact_terminates_on_a_cycle():
    # part A : B; part B : A; -> two edges pointing at each other.
    model = index(
        [element("A", "part_usage"), element("B", "part_usage")],
        [edge("A", "B", "type_ref"), edge("B", "A", "type_ref")],
    )
    reply = impact(model, "A")
    assert [row["id"] for row in reply["direct"]] == ["B"]
    assert reply["transitive"] == []


def test_impact_raises_for_an_unknown_element():
    with pytest.raises(KeyError):
        impact(index([element("A", "part_def")]), "B")


def test_impact_ignores_edges_that_point_nowhere():
    model = index(
        [element("A", "part_def")],
        [edge("A", "Missing", "type_ref", resolved=False)],
    )
    assert impact(model, "A")["direct"] == []


def test_impact_is_bounded():
    # A wide fan-in, which depth alone cannot bound.
    elements = [element("Wheel", "part_def")]
    elements += [element(f"w{n}", "part_usage") for n in range(IMPACT_NODES + 20)]
    edges = [edge(f"w{n}", "Wheel", "type_ref") for n in range(IMPACT_NODES + 20)]
    reply = impact(index(elements, edges), "Wheel")
    assert reply["truncated"] is True
    assert len(reply["direct"]) + len(reply["transitive"]) == IMPACT_NODES


# --- consistency -----------------------------------------------------------


def test_a_part_no_link_reaches_is_possibly_disconnected():
    model = index(
        [element("w1", "part_usage", line=4), element("w2", "part_usage", line=5)],
        [edge("connector", "w1", "connector_end")],
    )
    findings = [issue for issue in review(model) if issue.rule == "consistency"]
    assert [issue.element for issue in findings] == ["w2"]
    assert "possibly disconnected" in findings[0].message


def test_a_wired_part_is_not_a_consistency_finding():
    model = index(
        [element("w1", "part_usage")],
        [edge("connector", "w1", "flow_end")],
    )
    assert not [issue for issue in review(model) if issue.rule == "consistency"]


# --- carried-over findings -------------------------------------------------


def test_unresolved_edges_and_validator_issues_both_survive_review():
    carried = ModelIssue(severity="warning", message="Duplicate definition", rule="structural")
    model = index(
        [element("A", "part_usage")],
        [edge("A", "Nope", "type_ref", resolved=False)],
        [carried],
    )
    findings = review(model)
    unresolved = [issue for issue in findings if issue.rule == "reference"]
    assert len(unresolved) == 1
    assert unresolved[0].severity == "error"
    assert unresolved[0].message == "Nope does not resolve to anything in the model."
    assert carried in findings
