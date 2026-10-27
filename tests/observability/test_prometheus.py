from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from vrp_service.serving.app import create_app
from vrp_service.serving.settings import ServingSettings
from vrp_service.serving.solve import CVRP_EXAMPLE


def _samples(client: TestClient) -> dict:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    }


def _solves(samples: dict, **labels) -> float:
    key = ("vrp_solves_total", tuple(sorted(labels.items())))
    return samples.get(key, 0.0)


def test_metrics_count_solves_and_time_the_successful_ones() -> None:
    application = create_app(ServingSettings(model_version="mtl/v7.0.0"))
    client = TestClient(application)
    ok = {"backend": "mtl", "variant": "cvrp", "status": "ok", "feasible": "true", "error_type": ""}
    missing = {**ok, "status": "error", "feasible": "false", "error_type": "model_unavailable"}
    before = _samples(client)

    client.post("/solve", json=CVRP_EXAMPLE)
    application.state.policy = lambda _i: {"reward": [-2.5], "tours": [[0, 1, 0]]}
    client.post("/solve", json=CVRP_EXAMPLE)
    after = _samples(client)

    assert _solves(after, **missing) - _solves(before, **missing) == 1
    assert _solves(after, **ok) - _solves(before, **ok) == 1
    count = ("vrp_solve_latency_seconds_count", (("backend", "mtl"), ("variant", "cvrp")))
    assert after[count] - before.get(count, 0.0) == 1
    assert after[("vrp_policy_loaded", ())] == 1
    assert after[("vrp_model_info", (("model_version", "mtl/v7.0.0"),))] == 1


def test_the_expected_metric_names_are_exposed() -> None:
    names = {name for name, _labels in _samples(TestClient(create_app(ServingSettings())))}
    assert {"vrp_policy_loaded", "vrp_model_info"} <= names
    assert all(name.startswith(("vrp_", "python_", "process_")) for name in names)
