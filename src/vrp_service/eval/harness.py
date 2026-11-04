"""Score one backend on the frozen eval set: one JSONL row per instance and a markdown table.

    python -m vrp_service.eval.harness --out results/mtl.jsonl
    python -m vrp_service.eval.harness --smoke --out results/smoke.jsonl
    python -m vrp_service.eval.harness --mode http --url http://localhost:8000 --out ...

``latency_ms`` is the solve clock inside ``solve_mtl``: decode, route split, and the
feasibility check. ``wall_ms`` is what the harness waits for one solve: the same call
in process, or the full HTTP round trip. Reading the instance file is outside both.
Warm-up solves run first and are not recorded.
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from vrp_service.contracts import SolveRequest, SolveResponse
from vrp_service.env import (
    SOLVER_ROUNDING_TOLERANCE,
    check_routes,
    load_instance,
    route_length,
)
from vrp_service.eval.instance_set import (
    InstanceSet,
    load_instance_set,
    read_request,
    read_solution,
)
from vrp_service.eval.metrics import percentile
from vrp_service.serving.settings import ServingSettings, project_root

Solve = Callable[[SolveRequest], SolveResponse]
DEFAULT_CONFIG = "configs/eval/instance_set_v1.yaml"


def select(
    instance_set: InstanceSet, smoke: bool = False, limit: int | None = None
) -> list[tuple[str, int]]:
    if smoke:
        if limit is not None:
            raise ValueError("--smoke and --limit cannot be combined")
        return list(instance_set.smoke)
    count = instance_set.instances_per_variant if limit is None else limit
    if not 1 <= count <= instance_set.instances_per_variant:
        raise ValueError(f"limit must be between 1 and {instance_set.instances_per_variant}")
    return [(variant, index) for variant in instance_set.sha256 for index in range(count)]


def evaluate(
    instance_set: InstanceSet,
    items: Iterable[tuple[str, int]],
    solve: Solve,
    warmup: int = 0,
    backend: str = "mtl",
) -> Iterator[dict]:
    """Yield one row per item as it finishes, so a long run keeps what it has done."""
    items = list(items)
    for number in range(warmup if items else 0):
        variant, index = items[number % len(items)]
        _solve_safely(solve, read_request(instance_set, variant, index, backend))
    for variant, index in items:
        request = read_request(instance_set, variant, index, backend)
        started = time.perf_counter()
        response = _solve_safely(solve, request)
        wall_ms = (time.perf_counter() - started) * 1000
        row = score(variant, index, request, response, wall_ms)
        row.update(gaps(instance_set, variant, index, request, row))
        yield row


def _solve_safely(solve: Solve, request: SolveRequest) -> SolveResponse:
    """An exception becomes an error row instead of ending the run."""
    started = time.perf_counter()
    try:
        return solve(request)
    except Exception as error:  # noqa: BLE001
        return SolveResponse(
            backend=request.backend,
            model_version="unknown",
            feasible=False,
            routes=[],
            latency_ms=(time.perf_counter() - started) * 1000,
            status="error",
            error_type=f"harness_{type(error).__name__}",
        )


def score(
    variant: str, index: int, request: SolveRequest, response: SolveResponse, wall_ms: float
) -> dict:
    """``feasible`` is the harness's own check; the backend's claim is kept beside it."""
    ok = response.status == "ok"
    routes = [route.node_ids for route in response.routes]
    violations = check_routes(load_instance(request), routes) if ok else []
    return {
        "variant": variant,
        "index": index,
        "backend": response.backend,
        "model_version": response.model_version,
        "status": response.status,
        "error_type": response.error_type,
        "feasible": ok and not violations,
        "reported_feasible": response.feasible,
        "violations": sorted({violation.kind for violation in violations}),
        "objective_value": response.objective_value,
        "vehicles": len(routes),
        "routes": routes,
        "latency_ms": response.latency_ms,
        "wall_ms": wall_ms,
        "prompt_tokens": response.debug.get("prompt_tokens"),
        "completion_tokens": response.debug.get("completion_tokens"),
    }


def gaps(
    instance_set: InstanceSet, variant: str, index: int, request: SolveRequest, row: dict
) -> dict:
    """Gap to each stored reference, ``(cost - reference) / reference``.

    Reference costs are recomputed from the stored routes, because the stored PyVRP
    costs are rounded. A gap is ``None`` unless both solutions are feasible.
    """
    from vrp_service.serving.solve import routes_from_tour

    instance = load_instance(request)
    result = {}
    for solver in instance_set.solution_files:
        stored = read_solution(instance_set, variant, solver, index)
        routes = [r.node_ids for r in routes_from_tour(stored.actions, instance.open_route)]
        feasible = not check_routes(instance, routes, SOLVER_ROUNDING_TOLERANCE)
        reference = route_length(instance, routes)
        result[f"{solver}_cost"] = reference
        result[f"{solver}_feasible"] = feasible
        both = feasible and row["feasible"]
        result[f"gap_{solver}"] = (row["objective_value"] - reference) / reference if both else None
    return result


def markdown(rows: list[dict], header: dict[str, object]) -> str:
    """Cost and gaps average feasible rows only; latency covers rows that returned routes."""
    lines = [f"- {key}: {value}" for key, value in header.items()]
    lines += [
        "",
        "| Variant | n | Feasible | Mean cost (feasible) | Gap PyVRP | Gap OR-Tools | p50 ms |"
        " p95 ms | Wall p50 ms | Wall p95 ms | Errors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    groups = [(variant, [row for row in rows if row["variant"] == variant]) for variant in variants]
    if len(variants) > 1:
        groups.append(("all", rows))
    for name, group in groups:
        lines.append(_table_row(name, group))
    return "\n".join(lines) + "\n"


def _table_row(name: str, rows: list[dict]) -> str:
    feasible = sum(row["feasible"] for row in rows)
    costs = [row["objective_value"] for row in rows if row["feasible"]]
    solved = [row for row in rows if row["status"] == "ok"]
    latency = [row["latency_ms"] for row in solved]
    wall = [row["wall_ms"] for row in solved]
    cells = [
        name,
        str(len(rows)),
        f"{feasible / len(rows):.1%}",
        f"{sum(costs) / len(costs):.4f}" if costs else "-",
        *(_mean_gap(rows, f"gap_{solver}") for solver in ("pyvrp", "ortools")),
        *(_ms(values, q) for values in (latency, wall) for q in (50, 95)),
        str(len(rows) - len(solved)),
    ]
    return "| " + " | ".join(cells) + " |"


def _mean_gap(rows: list[dict], key: str) -> str:
    values = [row[key] for row in rows if row.get(key) is not None]
    return f"{sum(values) / len(values):+.2%}" if values else "-"


def _ms(values: list[float], q: float) -> str:
    return f"{percentile(values, q):.1f}" if values else "-"


def in_process(settings: ServingSettings, backend: str = "mtl") -> tuple[Solve, float | None]:
    """A solver that calls the model directly, and the load time when it loaded."""
    if backend == "llm":
        return _in_process_llm(settings)
    from vrp_service.serving.solve import solve_mtl

    policy = None
    load_ms = None
    if settings.weights_ready() and settings.source_root is not None:
        from vrp_service.mtl.decode import load_policy

        started = time.perf_counter()
        policy = load_policy(
            settings.checkpoint_path, settings.source_root, settings.device, settings.cpu_threads
        )
        load_ms = (time.perf_counter() - started) * 1000

    def solve(request: SolveRequest) -> SolveResponse:
        return solve_mtl(request, policy, settings.model_version)

    return solve, load_ms


def _in_process_llm(settings: ServingSettings) -> tuple[Solve, float | None]:
    from vrp_service.serving.solve import solve_llm

    generate = None
    load_ms = None
    if settings.llm_model_path:
        from vrp_service.llm.generate import generate as generate_answer
        from vrp_service.llm.generate import load_llm

        started = time.perf_counter()
        llm = load_llm(settings.llm_model_path, max_new_tokens=settings.llm_max_new_tokens)
        load_ms = (time.perf_counter() - started) * 1000

        def generate(messages: list[dict[str, str]]):
            return generate_answer(llm, messages)

    def solve(request: SolveRequest) -> SolveResponse:
        return solve_llm(request, generate, settings.llm_model_version)

    return solve, load_ms


def over_http(url: str, timeout_s: float = 60.0) -> Solve:
    endpoint = url.rstrip("/") + "/solve"

    def solve(request: SolveRequest) -> SolveResponse:
        body = request.model_dump_json().encode()
        call = urllib.request.Request(
            endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(call, timeout=timeout_s) as answer:
                return SolveResponse.model_validate_json(answer.read())
        except urllib.error.HTTPError as error:
            return SolveResponse(
                backend=request.backend,
                model_version="unknown",
                feasible=False,
                routes=[],
                latency_ms=0.0,
                status="error",
                error_type=f"http_{error.code}",
            )

    return solve


def server_ready(url: str) -> str:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/ready", timeout=5) as answer:
            return str(json.loads(answer.read())["ready"]).lower()
    except (OSError, ValueError, KeyError):
        return "unreachable"


def main(argv: list[str] | None = None) -> None:
    """Exit 1 when no instance was solved, for example when the policy is missing."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", required=True, help="JSONL path; the table goes next to it as .md")
    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--smoke", action="store_true", help="only the two CI smoke instances")
    parser.add_argument("--limit", type=int, help="first N instances per variant")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backend", choices=["mtl", "llm"], default="mtl")
    args = parser.parse_args(argv)

    root = project_root()
    instance_set = load_instance_set(root / args.config, root)
    try:
        items = select(instance_set, args.smoke, args.limit)
    except ValueError as error:
        parser.error(str(error))
    header: dict[str, object] = {
        "instance set": instance_set.name,
        "backend": args.backend,
        "mode": args.mode,
    }
    if args.mode == "http":
        solve = over_http(args.url)
        header["url"] = args.url
        header["server ready"] = server_ready(args.url)
    else:
        settings = ServingSettings.from_env()
        solve, load_ms = in_process(settings, args.backend)
        if args.backend == "mtl":
            header["device"] = settings.device
        if args.backend == "mtl" and settings.device == "cpu":
            header["cpu threads"] = settings.cpu_threads
        header["policy load ms"] = "not loaded" if load_ms is None else f"{load_ms:.0f}"
    header["warm-up solves"] = args.warmup

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as handle:
        for row in evaluate(instance_set, items, solve, args.warmup, args.backend):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
    table = markdown(rows, header)
    out.with_suffix(".md").write_text(table)
    print(table, end="")
    if not any(row["status"] == "ok" for row in rows):
        raise SystemExit("no instance was solved; see error_type in " + str(out))


if __name__ == "__main__":
    main()
