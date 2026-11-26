import json

import pytest

from vrp_service.eval import load_instance_set, read_request, read_solution
from vrp_service.llm import ROUTES_SCHEMA, ParseError, parse_routes
from vrp_service.serving.settings import project_root
from vrp_service.serving.solve import routes_from_tour

ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)
CVRP = read_request(INSTANCE_SET, "cvrp", 0, backend="llm")
OPEN = read_request(INSTANCE_SET, "ovrpbltw", 0, backend="llm")


def _answer(request, solver: str = "ortools") -> list[list[int]]:
    """The stored solver routes, written the way the model is asked to write them."""
    actions = read_solution(INSTANCE_SET, request.variant, solver, 0).actions
    return [[n for n in r.node_ids if n != 0] for r in routes_from_tour(actions)]


def _kinds(parsed) -> set[str]:
    return {violation.kind for violation in parsed.violations}


def test_solver_routes_written_as_an_answer_are_feasible() -> None:
    answer = _answer(CVRP)
    parsed = parse_routes(json.dumps({"routes": answer}), CVRP)
    assert parsed.violations == []
    assert parsed.routes == [[0, *route, 0] for route in answer]


def test_open_routes_get_no_closing_depot() -> None:
    answer = _answer(OPEN)
    parsed = parse_routes(json.dumps({"routes": answer}), OPEN)
    assert parsed.routes == [[0, *route] for route in answer]
    assert "depot_end" not in _kinds(parsed)


@pytest.mark.parametrize(
    "fence", ["```json\n{}\n```", "```JSON\n{}```", "```{}```", "```\n{}\n```"]
)
def test_a_code_fence_is_stripped(fence: str) -> None:
    text = fence.replace("{}", json.dumps({"routes": _answer(CVRP)}))
    assert parse_routes(text, CVRP).violations == []


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ('{"routes": [[1, 2], [3', "truncated"),
        ('{"routes": [[1, 2]]', "truncated"),
        ('```json\n{"routes": [[1, 2]', "truncated"),
        ("[" * 100_000, "invalid_json"),
        ('{"routes": [[1' + "0" * 5000 + "]]}", "invalid_json"),
        ("", "truncated"),
        ("Here are the routes: [[1, 2]]", "invalid_json"),
        ('{"routes": [[1, 2]]} extra', "invalid_json"),
        ('{"routes": [["1", 2]]}', "schema"),
        ('{"routes": [[1.0, 2]]}', "schema"),
        ('{"routes": [[true, 2]]}', "schema"),
        ('{"routes": [1, 2]}', "schema"),
        ('{"route": [[1, 2]]}', "schema"),
        ('{"routes": [[1, 2]], "cost": 3}', "schema"),
        ("[[1, 2]]", "schema"),
    ],
)
def test_unusable_answers_raise_with_their_kind(text: str, kind: str) -> None:
    with pytest.raises(ParseError) as raised:
        parse_routes(text, CVRP)
    assert raised.value.kind == kind


def test_a_dropped_customer_is_not_visited() -> None:
    answer = _answer(CVRP)
    missing = answer[0].pop()
    parsed = parse_routes(json.dumps({"routes": answer}), CVRP)
    assert [(v.kind, v.node) for v in parsed.violations] == [("not_visited", missing)]


def test_a_repeated_customer_is_visited_twice() -> None:
    answer = _answer(CVRP)
    answer.append([answer[0][0]])
    assert "visited_twice" in _kinds(parse_routes(json.dumps({"routes": answer}), CVRP))


def test_a_depot_written_inside_a_route_is_rejected() -> None:
    answer = _answer(CVRP)
    answer[0].insert(1, 0)
    assert "depot_inside" in _kinds(parse_routes(json.dumps({"routes": answer}), CVRP))


def test_unknown_ids_and_empty_routes_are_rejected() -> None:
    answer = _answer(CVRP) + [[], [999]]
    kinds = _kinds(parse_routes(json.dumps({"routes": answer}), CVRP))
    assert {"empty_route", "unknown_node"} <= kinds


def test_the_schema_allows_only_lists_of_integers() -> None:
    assert ROUTES_SCHEMA["additionalProperties"] is False
    assert ROUTES_SCHEMA["properties"]["routes"]["items"]["items"] == {"type": "integer"}
    # Without maxItems lm-format-enforcer stops every list at 20 items.
    routes = ROUTES_SCHEMA["properties"]["routes"]
    assert routes["maxItems"] >= 1000 and routes["items"]["maxItems"] >= 1000
