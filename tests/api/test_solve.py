from fastapi.testclient import TestClient

from vrp_service.contracts import SolveRequest
from vrp_service.serving.app import create_app
from vrp_service.serving.settings import ServingSettings
from vrp_service.serving.solve import CVRP_EXAMPLE, routes_from_tour, solve_mtl


def _cvrp(backend: str = "mtl") -> SolveRequest:
    payload = {**CVRP_EXAMPLE, "backend": backend}
    return SolveRequest.model_validate(payload)


def test_routes_split_when_the_tour_returns_to_the_depot() -> None:
    routes = routes_from_tour((0, 1, 2, 0, 3, 0))
    assert [route.node_ids for route in routes] == [[0, 1, 2, 0], [0, 3, 0]]
    assert [route.vehicle_id for route in routes] == [0, 1]


def test_routes_start_at_the_depot_when_the_tour_does_not() -> None:
    routes = routes_from_tour((1, 0))
    assert [route.node_ids for route in routes] == [[0, 1, 0]]


def test_a_closed_route_returns_to_the_depot_after_its_last_customer() -> None:
    routes = routes_from_tour((3, 4, 0, 2))
    assert [route.node_ids for route in routes] == [[0, 3, 4, 0], [0, 2, 0]]


def test_open_routes_end_at_their_last_customer_and_drop_padding() -> None:
    routes = routes_from_tour((5, 2, 0, 3, 0, 1, 0, 0), open_route=True)
    assert [route.node_ids for route in routes] == [[0, 5, 2], [0, 3], [0, 1]]


def test_solve_returns_routes_cost_and_model_version() -> None:
    def policy(instance):
        assert instance.node_ids == (0, 1)
        return {"reward": [-2.5], "tours": [[0, 1, 0]]}

    response = solve_mtl(_cvrp(), policy)
    assert response.status == "ok"
    assert response.model_version == "mtl/v0.1.0"
    assert response.feasible is True
    assert response.objective_value == 2.5
    assert response.routes[0].node_ids == [0, 1, 0]
    assert response.gap_to_baseline is None


def test_post_solve_uses_an_injected_policy() -> None:
    application = create_app(ServingSettings())
    application.state.policy = lambda _instance: {"reward": [-2.5], "tours": [[0, 1, 0]]}
    response = TestClient(application).post("/solve", json=CVRP_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "mtl/v0.1.0"
    assert body["objective_value"] == 2.5
    assert body["routes"] == [{"vehicle_id": 0, "node_ids": [0, 1, 0]}]


def test_solve_rejects_a_missing_checkpoint() -> None:
    response = TestClient(create_app(ServingSettings())).post("/solve", json=CVRP_EXAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["error_type"] == "model_unavailable"
    assert body["feasible"] is False
    assert body["routes"] == []


def test_solve_rejects_a_non_mtl_backend() -> None:
    def policy(_instance):
        return {"reward": [-1.0], "tours": [[0, 1]]}

    response = solve_mtl(_cvrp("llm"), policy)
    assert response.status == "error"
    assert response.error_type == "unsupported_backend"


def test_ready_is_true_when_the_checkpoint_file_exists(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"")
    application = create_app(ServingSettings(checkpoint_path=str(checkpoint)))
    assert TestClient(application).get("/ready").json() == {"ready": True}


def test_startup_loads_the_policy_once(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"")
    loads = []

    def fake_load(checkpoint_path, source_root, device, cpu_threads):
        loads.append(checkpoint_path)
        return lambda _instance: {"reward": [-2.5], "tours": [[0, 1, 0]]}

    monkeypatch.setattr("vrp_service.serving.app.load_policy", fake_load)
    settings = ServingSettings(checkpoint_path=str(checkpoint), source_root=str(tmp_path))
    with TestClient(create_app(settings)) as client:
        assert len(loads) == 1
        for _ in range(2):
            assert client.post("/solve", json=CVRP_EXAMPLE).json()["status"] == "ok"
    assert len(loads) == 1


def test_openapi_includes_a_cvrp_example() -> None:
    spec = TestClient(create_app(ServingSettings())).get("/openapi.json").json()
    example = spec["paths"]["/solve"]["post"]["requestBody"]["content"]["application/json"]
    assert example["examples"]["cvrp"]["value"]["variant"] == "cvrp"
