"""Prometheus metrics for the HTTP solve path, served at ``GET /metrics``."""

from prometheus_client import Counter, Gauge, Histogram

from vrp_service.contracts import SolveRequest, SolveResponse

# Solves take about 0.5 s at 100 customers; the buckets resolve that range.
SOLVE_SECONDS = Histogram(
    "vrp_solve_latency_seconds",
    "Time inside the solve call (latency_ms), by backend and variant.",
    ["backend", "variant"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 2.5, 5.0, 10.0),
)
SOLVES = Counter(
    "vrp_solves",
    "Solve requests by outcome.",
    ["backend", "variant", "status", "feasible", "error_type"],
)
POLICY_LOADED = Gauge("vrp_policy_loaded", "1 when the routing policy is loaded in memory.")
MODEL_INFO = Gauge("vrp_model_info", "The model version this process serves.", ["model_version"])


def record_solve(request: SolveRequest, response: SolveResponse) -> None:
    SOLVES.labels(
        request.backend,
        request.variant,
        response.status,
        str(response.feasible).lower(),
        response.error_type or "",
    ).inc()
    if response.status == "ok":
        SOLVE_SECONDS.labels(request.backend, request.variant).observe(response.latency_ms / 1000)
