"""Turn one solve request into the shared response.

``backend=mtl`` decodes with the supplied policy. Any other backend, or a
missing policy, stays on the same response schema with ``feasible`` false.
"""

import time
from collections.abc import Callable, Sequence

from vrp_service.contracts import Route, SolveRequest, SolveResponse
from vrp_service.env.instance import EnvironmentInstance
from vrp_service.mtl.decode import decode_request
from vrp_service.serving.settings import MODEL_VERSION

Policy = Callable[[EnvironmentInstance], dict]

CVRP_EXAMPLE = {
    "contract_version": "1.0.0",
    "backend": "mtl",
    "variant": "cvrp",
    "seed": 0,
    "instance": {
        "nodes": [
            {"id": 0, "x": 0.0, "y": 0.0, "demand_linehaul": 0, "demand_backhaul": 0},
            {"id": 1, "x": 0.2, "y": 0.3, "demand_linehaul": 1, "demand_backhaul": 0},
        ],
        "capacity": 10,
        "open_route": False,
    },
    "options": {"timeout_ms": 30000, "return_debug": False},
}


def routes_from_tour(node_ids: Sequence[int]) -> list[Route]:
    """Split a tour on each return to the depot. A trailing open leg stays one route."""
    routes: list[Route] = []
    current: list[int] = []
    nodes = [int(node) for node in node_ids]
    if nodes and nodes[0] != 0:
        nodes = [0, *nodes]
    for node in nodes:
        current.append(int(node))
        if node == 0 and len(current) > 1:
            routes.append(Route(vehicle_id=len(routes), node_ids=list(current)))
            current = [0]
    if current not in ([], [0]):
        routes.append(Route(vehicle_id=len(routes), node_ids=list(current)))
    return routes


def solve_mtl(
    request: SolveRequest,
    policy: Policy | None,
    model_version: str = MODEL_VERSION,
) -> SolveResponse:
    started = time.perf_counter()
    if request.backend != "mtl":
        return _error(request, model_version, "unsupported_backend", started)
    if policy is None:
        return _error(request, model_version, "model_unavailable", started)
    try:
        decoded = decode_request(request, policy)
    except (ValueError, FileNotFoundError, RuntimeError):
        return _error(request, model_version, "decode_failed", started)
    return SolveResponse(
        backend=request.backend,
        model_version=model_version,
        feasible=True,
        objective_value=decoded.cost,
        routes=routes_from_tour(decoded.node_ids),
        latency_ms=_latency(started),
        status="ok",
    )


def _error(
    request: SolveRequest,
    model_version: str,
    error_type: str,
    started: float,
) -> SolveResponse:
    return SolveResponse(
        backend=request.backend,
        model_version=model_version,
        feasible=False,
        routes=[],
        latency_ms=_latency(started),
        status="error",
        error_type=error_type,
    )


def _latency(started: float) -> float:
    return (time.perf_counter() - started) * 1000
