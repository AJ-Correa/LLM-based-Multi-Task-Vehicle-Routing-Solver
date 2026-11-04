"""Turn one solve request into the shared response.

``backend=mtl`` decodes with the supplied policy; ``backend=llm`` renders the prompt,
generates, and parses the answer with no repair. ``feasible`` comes from the route
checker in both. A wrong backend, or a missing model, stays on the same response
schema with ``feasible`` false.
"""

import time
from collections.abc import Callable, Sequence

from vrp_service.contracts import Route, SolveRequest, SolveResponse
from vrp_service.env import EnvironmentInstance, check_routes, load_instance, route_length
from vrp_service.llm.generate import Generation
from vrp_service.llm.parse import ParsedRoutes, ParseError, parse_routes
from vrp_service.llm.prompt import PROMPT_VERSION, render
from vrp_service.mtl.decode import decode_request
from vrp_service.observability.tracing import tracer
from vrp_service.serving.settings import LLM_MODEL_VERSION, MODEL_VERSION

Policy = Callable[[EnvironmentInstance], dict]
Generate = Callable[[list[dict[str, str]]], Generation]

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
    return _traced(request, model_version, lambda: _solve(request, policy, model_version))


def solve_llm(
    request: SolveRequest,
    generate: Generate | None,
    model_version: str = LLM_MODEL_VERSION,
) -> SolveResponse:
    return _traced(request, model_version, lambda: _solve_llm(request, generate, model_version))


def _traced(
    request: SolveRequest, model_version: str, run: Callable[[], SolveResponse]
) -> SolveResponse:
    attributes = {
        "vrp.backend": request.backend,
        "vrp.variant": request.variant,
        "vrp.customers": len(request.instance.nodes) - 1,
        "vrp.model_version": model_version,
    }
    with tracer.start_as_current_span("solve", attributes=attributes) as span:
        response = run()
        span.set_attribute("vrp.status", response.status)
        span.set_attribute("vrp.feasible", response.feasible)
        span.set_attribute("vrp.vehicles", len(response.routes))
        if response.error_type is not None:
            span.set_attribute("vrp.error_type", response.error_type)
        if response.objective_value is not None:
            span.set_attribute("vrp.objective_value", response.objective_value)
        return response


def _solve(request: SolveRequest, policy: Policy | None, model_version: str) -> SolveResponse:
    started = time.perf_counter()
    if request.backend != "mtl":
        return _error(request, model_version, "unsupported_backend", started)
    if policy is None:
        return _error(request, model_version, "model_unavailable", started)
    try:
        with tracer.start_as_current_span("decode"):
            decoded = decode_request(request, policy)
    except (ValueError, FileNotFoundError, RuntimeError):
        return _error(request, model_version, "decode_failed", started)
    with tracer.start_as_current_span("split_routes"):
        routes = routes_from_tour(decoded.node_ids, request.instance.open_route)
    with tracer.start_as_current_span("check_routes") as span:
        violations = check_routes(load_instance(request), [route.node_ids for route in routes])
        span.set_attribute("vrp.violations", sorted({v.kind for v in violations}))
    return SolveResponse(
        backend=request.backend,
        model_version=model_version,
        feasible=not violations,
        objective_value=decoded.cost,
        routes=routes,
        latency_ms=_latency(started),
        status="ok",
    )


def _solve_llm(
    request: SolveRequest, generate: Generate | None, model_version: str
) -> SolveResponse:
    started = time.perf_counter()
    if request.backend != "llm":
        return _error(request, model_version, "unsupported_backend", started)
    if generate is None:
        return _error(request, model_version, "model_unavailable", started)
    with tracer.start_as_current_span("render"):
        messages = render(request)
    try:
        with tracer.start_as_current_span("generate") as span:
            generation = generate(messages)
            span.set_attribute("llm.prompt_tokens", generation.prompt_tokens)
            span.set_attribute("llm.completion_tokens", generation.completion_tokens)
    except Exception:  # noqa: BLE001 - any failure inside the model stack is one outcome
        return _error(request, model_version, "generation_failed", started)
    # Token counts ride on every LLM response, failed or not: the cost table needs them.
    debug = {
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "hit_token_limit": generation.hit_limit,
    }
    if request.options.return_debug:
        debug["raw_output"] = generation.text
    try:
        with tracer.start_as_current_span("parse_and_check") as span:
            parsed = parse_routes(generation.text, request)
            span.set_attribute("vrp.violations", sorted({v.kind for v in parsed.violations}))
    except ParseError as error:
        kind = "truncated" if generation.hit_limit else error.kind
        return _error(request, model_version, kind, started, debug)
    return SolveResponse(
        backend=request.backend,
        model_version=model_version,
        prompt_version=PROMPT_VERSION,
        feasible=not parsed.violations,
        objective_value=_llm_cost(request, parsed),
        routes=[
            Route(vehicle_id=number, node_ids=route) for number, route in enumerate(parsed.routes)
        ],
        latency_ms=_latency(started),
        status="ok",
        debug=debug,
    )


def _llm_cost(request: SolveRequest, parsed: ParsedRoutes) -> float | None:
    """No length for a route through a node that does not exist."""
    if any(violation.kind == "unknown_node" for violation in parsed.violations):
        return None
    return route_length(load_instance(request), parsed.routes)


def _error(
    request: SolveRequest,
    model_version: str,
    error_type: str,
    started: float,
    debug: dict | None = None,
) -> SolveResponse:
    return SolveResponse(
        backend=request.backend,
        model_version=model_version,
        prompt_version=PROMPT_VERSION if request.backend == "llm" else None,
        feasible=False,
        routes=[],
        latency_ms=_latency(started),
        status="error",
        error_type=error_type,
        debug=debug or {},
    )


def _latency(started: float) -> float:
    return (time.perf_counter() - started) * 1000
