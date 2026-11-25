"""The model index must come from the real parse tree, not from keyword scans.

The fixture is deliberately one of each shape the feature claims to cover:
package, part def and usage, a nested usage body, port def and usage, item,
action with a perform, a connect, a typed attribute, a requirement def and a
satisfied requirement usage, and one misspelt type.
"""

from __future__ import annotations

import pytest

from webapp.model_index import build_index

CODE = """package Vehicle {
    part def Wheel {
        attribute pressure : Real;
    }
    port def HydraulicPort;
    item def Fluid;
    action def Pressurise;
    part def Car {
        part w1 : Wheel;
        part w2 : Wheel;
        port inlet : HydraulicPort;
        item oil : Fluid;
        connect w1.pressure to w2.pressure;
        action run {
            perform Pressurise;
        }
    }
    requirement def BrakeReq {
        subject vehicle : Car;
    }
    part vehicle : Car {
        part w1 : Wheel;
        requirement brake : BrakeReq;
        satisfy brake;
    }
    part spare : Wheell;
}
"""

BROKEN = """package Broken {
    part def Wheel {
"""


def line_of(needle: str, text: str = CODE) -> int:
    """1-based line number of the first line holding the needle."""
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(f"{needle!r} is not in the fixture")


def element(index, element_id):
    for candidate in index.elements:
        if candidate.id == element_id:
            return candidate
    raise AssertionError(f"{element_id} is not in the index")


@pytest.fixture(scope="module")
def index():
    return build_index(CODE)


# --- elements --------------------------------------------------------------


def test_elements_nest_under_the_tree(index):
    got = {(e.id, e.kind, e.parent) for e in index.elements}
    assert {
        ("Vehicle", "package", None),
        ("Vehicle::Wheel", "part_def", "Vehicle"),
        ("Vehicle::Wheel::pressure", "attribute_usage", "Vehicle::Wheel"),
        ("Vehicle::HydraulicPort", "port_def", "Vehicle"),
        ("Vehicle::Fluid", "item_def", "Vehicle"),
        ("Vehicle::Pressurise", "action_def", "Vehicle"),
        ("Vehicle::Car", "part_def", "Vehicle"),
        ("Vehicle::Car::w1", "part_usage", "Vehicle::Car"),
        ("Vehicle::Car::inlet", "port_usage", "Vehicle::Car"),
        ("Vehicle::Car::oil", "item_usage", "Vehicle::Car"),
        ("Vehicle::Car::run", "action_usage", "Vehicle::Car"),
        ("Vehicle::BrakeReq", "requirement_def", "Vehicle"),
        ("Vehicle::vehicle", "part_usage", "Vehicle"),
        ("Vehicle::vehicle::w1", "part_usage", "Vehicle::vehicle"),
        ("Vehicle::vehicle::brake", "requirement_usage", "Vehicle::vehicle"),
    } <= got


def test_a_usage_body_is_not_flattened_into_the_package(index):
    # The inherited symbol table has no scope for a usage body; the index does.
    ids = {e.id for e in index.elements}
    assert "Vehicle::vehicle::w1" in ids
    assert "Vehicle::w1" not in ids


def test_line_ranges_come_from_the_tree(index):
    assert element(index, "Vehicle").end_line == len(CODE.strip().splitlines())
    assert element(index, "Vehicle::Car::w1").end_line == line_of("part w1 : Wheel;")
    assert element(index, "Vehicle::Car::run").end_line == line_of("perform Pressurise;") + 1
    assert element(index, "Vehicle::Car").end_line == line_of("perform Pressurise;") + 2
    assert element(index, "Vehicle::vehicle").end_line == line_of("satisfy brake;") + 1


# --- edges -----------------------------------------------------------------


def test_satisfy_is_an_edge_from_the_satisfier_to_the_requirement(index):
    edges = [e for e in index.edges if e.kind == "satisfy"]
    assert [(e.source, e.target) for e in edges] == [
        ("Vehicle::vehicle", "Vehicle::vehicle::brake")
    ]


def test_a_perform_edge_belongs_to_the_action_that_performs(index):
    edges = [e for e in index.edges if e.kind == "perform_action"]
    assert [(e.source, e.target) for e in edges] == [
        ("Vehicle::Car::run", "Vehicle::Pressurise")
    ]


def test_a_builtin_type_resolves_without_becoming_an_element(index):
    typed = [e for e in index.edges if e.source == "Vehicle::Wheel::pressure"]
    assert [(e.target, e.resolved) for e in typed] == [("Real", True)]
    assert "Real" not in {e.id for e in index.elements}


def test_a_misspelt_type_is_an_unresolved_edge(index):
    typed = [e for e in index.edges if e.source == "Vehicle::spare"]
    assert [(e.target, e.resolved) for e in typed] == [("Wheell", False)]
    assert index.stats.unresolved == 1


def test_stats_count_what_the_index_holds(index):
    assert index.validator == "antlr"
    assert index.truncated is False
    assert index.stats.elements == len(index.elements)
    assert index.stats.edges == len(index.edges)
    assert index.stats.requirements == 2


# --- degradation -----------------------------------------------------------


def test_a_syntax_error_is_named_instead_of_passing_as_a_whole_model():
    # The grammar recovers and still yields a partial tree; the index says so
    # rather than presenting the recovered fragment as the model.
    broken = build_index(BROKEN)
    assert broken.validator == "syntax-error"
    assert broken.issues and broken.issues[0].severity == "error"
    assert broken.issues[0].rule == "syntax"
    assert broken.issues[0].line == line_of("part def Wheel {", BROKEN)


def test_an_empty_model_is_named_as_such():
    empty = build_index("   ")
    assert empty.validator == "empty"
    assert empty.issues[0].message == "The model is empty."


# --- the rule table --------------------------------------------------------


def test_every_generated_rule_is_classified():
    """Regenerating NewGrammar must fail here, not silently drop elements."""
    from tools.sysml_parser.NewGrammar.SysMLv2ParserListener import (
        SysMLv2ParserListener,
    )

    from webapp.model_index import _ELEMENT_RULES, _IGNORED_RULES

    rules = {
        name[len("enter") :] + "Context"
        for name in dir(SysMLv2ParserListener)
        if name.startswith("enter") and name != "enterEveryRule"
    }
    assert rules == set(_ELEMENT_RULES) | set(_IGNORED_RULES)


# --- endpoints -------------------------------------------------------------


@pytest.fixture
def client():
    """A signed-in client: the analysis endpoints are account-scoped like the rest.

    Not a context manager on purpose — that would run the app's lifespan and tear
    down the session-wide connection pool the other tests share.
    """
    pytest.importorskip("httpx")
    import secrets

    from fastapi.testclient import TestClient

    from webapp import server

    http = TestClient(server.app)
    registered = http.post(
        "/api/auth/register",
        json={
            "email": f"index-{secrets.token_hex(4)}@example.test",
            "password": "correct-horse-battery",
        },
    )
    assert registered.status_code == 200, registered.text
    return http


def test_index_endpoint_returns_the_report(client):
    body = client.post("/api/model/index", json={"code": CODE}).json()
    assert body["validator"] == "antlr"
    assert body["truncated"] is False
    assert body["stats"]["elements"] == len(body["elements"])
    assert body["stats"]["requirements"] == 2
    assert body["traceability"]["counts"] == {
        "requirements": 2,
        "satisfied": 1,
        "unlinked": 1,
    }
    assert {issue["rule"] for issue in body["review"]} >= {"reference", "traceability"}


def test_index_endpoint_repeats_the_revision_for_the_same_model(client):
    first = client.post("/api/model/index", json={"code": CODE})
    second = client.post("/api/model/index", json={"code": CODE})
    assert first.status_code == 200
    assert second.json()["revision"] == first.json()["revision"]


def test_impact_endpoint_walks_backwards(client):
    reply = client.post(
        "/api/model/impact", json={"code": CODE, "element_id": "Vehicle::Wheel"}
    )
    assert reply.status_code == 200
    direct = {row["id"] for row in reply.json()["direct"]}
    assert direct == {"Vehicle::Car::w1", "Vehicle::Car::w2", "Vehicle::vehicle::w1"}


def test_both_endpoints_reject_empty_and_unknown_input(client):
    assert client.post("/api/model/index", json={"code": "  "}).status_code == 400
    blank = client.post("/api/model/impact", json={"code": "  ", "element_id": "x"})
    assert blank.status_code == 400
    unknown = client.post(
        "/api/model/impact", json={"code": CODE, "element_id": "nope"}
    )
    assert unknown.status_code == 400
    assert unknown.json()["detail"] == "unknown element"
