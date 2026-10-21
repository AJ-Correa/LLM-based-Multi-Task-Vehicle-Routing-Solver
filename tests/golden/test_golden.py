"""Two checked-in solves that lock the loader and the policy.

The input snapshot runs everywhere. The solve needs torch and the n=100
checkpoint, so CI skips it and a local run checks the routes and cost.
"""

import json
import math
from pathlib import Path

import pytest

from vrp_service.contracts import SolveRequest
from vrp_service.env import load_instance
from vrp_service.mtl.decode import batch_fields, prompt_tag
from vrp_service.serving.settings import ServingSettings
from vrp_service.serving.solve import solve_mtl

GOLDEN = Path(__file__).parent
EXPECTED = json.loads((GOLDEN / "expected.json").read_text())
NAMES = sorted(EXPECTED)


def _request(name: str) -> SolveRequest:
    return SolveRequest.model_validate_json((GOLDEN / f"{name}.json").read_text())


def _route_length(request: SolveRequest, node_ids: list[int]) -> float:
    points = {node.id: (node.x, node.y) for node in request.instance.nodes}
    return sum(math.dist(points[a], points[b]) for a, b in zip(node_ids, node_ids[1:]))


@pytest.mark.parametrize("name", NAMES)
def test_loader_builds_the_snapshot_inputs(name: str) -> None:
    instance = load_instance(_request(name))
    assert batch_fields(instance) == EXPECTED[name]["inputs"]
    assert list(prompt_tag(instance)) == EXPECTED[name]["prompt_tag"]


@pytest.fixture(scope="module")
def policy():
    pytest.importorskip("torch")
    settings = ServingSettings.from_env()
    if not settings.weights_ready():
        pytest.skip("the n=100 checkpoint is not on disk")
    from vrp_service.mtl.decode import load_policy

    return load_policy(settings.checkpoint_path, settings.source_root, "cpu")


@pytest.mark.parametrize("name", NAMES)
def test_policy_reproduces_the_golden_solve(name: str, policy) -> None:
    request = _request(name)
    expected = EXPECTED[name]
    response = solve_mtl(request, policy)

    assert response.status == "ok"
    routes = [route.node_ids for route in response.routes]
    assert routes == expected["routes"]
    assert response.objective_value == pytest.approx(expected["cost"], rel=1e-4)

    customers = sorted(node for route in routes for node in route if node != 0)
    assert customers == [node.id for node in request.instance.nodes[1:]]
    length = sum(_route_length(request, route) for route in routes)
    assert length == pytest.approx(response.objective_value, rel=1e-4)
