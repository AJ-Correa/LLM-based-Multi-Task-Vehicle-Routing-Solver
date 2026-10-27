"""Turn one solve request into the shared response.

``backend=mtl`` decodes with the supplied policy, and ``feasible`` comes from the
route checker. Any other backend, or a missing policy, stays on the same
response schema with ``feasible`` false.
"""

import time
from collections.abc import Callable, Sequence

from vrp_service.contracts import Route, SolveRequest, SolveResponse
from vrp_service.env import EnvironmentInstance, check_routes, load_instance
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


def routes_from_tour(node_ids: Sequence[int], open_route: bool = False) -> list[Route]:
    """Split a tour into one route per vehicle.

    A depot visit separates vehicles, and trailing zeros are padding. A closed
    route returns to the depot after its last customer; an open route ends there.
    """
    routes: list[Route] = []
    segment: list[int] = []
    for node in [*(int(node) for node in node_ids), 0]:
        if node != 0:
            segment.append(node)
            continue
        if segment:
            ending = [] if open_route else [0]
            routes.append(Route(vehicle_id=len(routes), node_ids=[0, *segment, *ending]))
            segment = []
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
    routes = routes_from_tour(decoded.node_ids, request.instance.open_route)
    violations = check_routes(load_instance(request), [route.node_ids for route in routes])
    return SolveResponse(
        backend=request.backend,
        model_version=model_version,
        feasible=not violations,
        objective_value=decoded.cost,
        routes=routes,
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
