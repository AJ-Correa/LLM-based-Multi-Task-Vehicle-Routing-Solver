"""The LLM solve path with generation stubbed, so CI never loads the 7B weights."""

import json

from fastapi.testclient import TestClient

from vrp_service.eval import load_instance_set, read_request, read_solution
from vrp_service.eval.harness import evaluate
from vrp_service.llm.generate import Generation
from vrp_service.serving.app import create_app
from vrp_service.serving.settings import ServingSettings, project_root
from vrp_service.serving.solve import routes_from_tour, solve_llm

ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)


def _request(variant: str = "cvrp"):
    return read_request(INSTANCE_SET, variant, 0, backend="llm")


def _solver_answer(variant: str = "cvrp") -> str:
    actions = read_solution(INSTANCE_SET, variant, "ortools", 0).actions
    routes = [[n for n in r.node_ids if n != 0] for r in routes_from_tour(actions)]
    return json.dumps({"routes": routes})


def _stub(text: str, hit_limit: bool = False):
    return lambda _messages: Generation(text, 1500, 300, hit_limit)


def test_a_feasible_answer_returns_routes_cost_and_tokens() -> None:
    response = solve_llm(_request(), _stub(_solver_answer()))
    assert response.status == "ok" and response.feasible
    assert response.model_version == "llm/v0.1.0" and response.prompt_version == "v1"
    assert all(route.node_ids[0] == 0 and route.node_ids[-1] == 0 for route in response.routes)
    assert response.objective_value > 0
    assert response.debug == {
        "prompt_tokens": 1500,
        "completion_tokens": 300,
        "hit_token_limit": False,
    }


def test_open_variant_routes_do_not_return() -> None:
    response = solve_llm(_request("ovrpbltw"), _stub(_solver_answer("ovrpbltw")))
    assert response.feasible
    assert all(route.node_ids[-1] != 0 for route in response.routes)


def test_a_violating_answer_is_ok_but_infeasible() -> None:
    response = solve_llm(_request(), _stub('{"routes": [[1, 2, 3]]}'))
    assert response.status == "ok" and not response.feasible
    assert response.routes[0].node_ids == [0, 1, 2, 3, 0]


def test_an_unparseable_answer_is_an_error_with_its_kind_and_tokens() -> None:
    response = solve_llm(_request(), _stub('{"routes": [["1"]]}'))
    assert (response.status, response.error_type) == ("error", "schema")
    assert response.routes == [] and response.debug["completion_tokens"] == 300
    assert response.prompt_version == "v1"


def test_hitting_the_token_limit_counts_as_truncated() -> None:
    response = solve_llm(_request(), _stub('{"routes": [[1, 2', hit_limit=True))
    assert response.error_type == "truncated"


def test_raw_output_only_with_return_debug() -> None:
    request = _request()
    request.options.return_debug = True
    response = solve_llm(request, _stub("{}"))
    assert response.debug["raw_output"] == "{}"


def test_an_unknown_node_is_infeasible_with_no_cost() -> None:
    response = solve_llm(_request(), _stub('{"routes": [[999], [-1]]}'))
    assert response.status == "ok" and not response.feasible
    assert response.objective_value is None


def test_a_failed_llm_load_keeps_the_app_up(monkeypatch) -> None:
    def broken_load(*_args, **_kwargs):
        raise OSError("no such model")

    monkeypatch.setattr("vrp_service.llm.generate.load_llm", broken_load)
    with TestClient(create_app(ServingSettings(llm_model_path="/missing"))) as client:
        body = client.post("/solve", json=_request().model_dump(mode="json")).json()
        assert body["error_type"] == "model_unavailable"


def test_missing_model_wrong_backend_and_generation_failure() -> None:
    assert solve_llm(_request(), None).error_type == "model_unavailable"
    mtl = read_request(INSTANCE_SET, "cvrp", 0, backend="mtl")
    assert solve_llm(mtl, _stub("{}")).error_type == "unsupported_backend"

    for error in (RuntimeError("CUDA out of memory"), KeyError("token")):

        def broken(_messages, error=error):
            raise error

        response = solve_llm(_request(), broken)
        assert response.error_type == "generation_failed" and response.prompt_version == "v1"


def test_post_solve_routes_llm_requests_to_the_llm(monkeypatch) -> None:
    application = create_app(ServingSettings())
    application.state.llm = object()
    answer = _solver_answer()
    monkeypatch.setattr(
        "vrp_service.serving.app.generate_answer",
        lambda _llm, _messages: Generation(answer, 1500, 300, False),
    )
    body = TestClient(application).post("/solve", json=_request().model_dump(mode="json")).json()
    assert body["status"] == "ok" and body["feasible"] and body["model_version"] == "llm/v0.1.0"


def test_post_solve_without_an_llm_is_model_unavailable() -> None:
    client = TestClient(create_app(ServingSettings()))
    body = client.post("/solve", json=_request().model_dump(mode="json")).json()
    assert body["error_type"] == "model_unavailable"


def test_harness_rows_carry_token_counts() -> None:
    def solve(request):
        return solve_llm(request, _stub(_solver_answer()))

    [row] = evaluate(INSTANCE_SET, [("cvrp", 0)], solve, backend="llm")
    assert row["backend"] == "llm" and row["feasible"]
    assert (row["prompt_tokens"], row["completion_tokens"]) == (1500, 300)
    assert row["gap_ortools"] == 0.0
