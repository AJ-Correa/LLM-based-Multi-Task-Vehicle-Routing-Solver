"""Solve generated instances with PyVRP and store the routes in the eval set's solution format.

    python -m vrp_service.data.label --root data/train/n100 --variant cvrp --seconds 20

The model mirrors ``check_routes``: an open route's legs back to the depot cost nothing, a
classical backhaul route may not go from a backhaul to a linehaul customer (that leg is
made prohibitively long), and travel time is distance / speed. Values are scaled by 1000
to integers, rounded so that every PyVRP-feasible route is feasible in exact arithmetic
(times and lengths up, window ends and limits down). Every stored route is re-checked with
``check_routes``; ``feasible`` records the result and training uses only feasible rows.
"""

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vrp_service.env import EnvironmentInstance, check_routes, load_instance, route_length
from vrp_service.eval.instance_set import InstanceSet, read_request

SCALE = 1000
# Far above any real route; PyVRP treats it as a penalised, never-chosen leg.
FORBIDDEN = 10**9


def solve(instance: EnvironmentInstance, seconds: float, seed: int = 0) -> list[list[int]]:
    """Routes in the response shape: depot first, and depot last unless the route is open."""
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime

    model = Model()
    capacity = round(instance.vehicle_capacity * SCALE)
    limit = instance.distance_limit
    model.add_vehicle_type(
        num_available=len(instance.node_ids) - 1,
        capacity=capacity,
        max_distance=math.floor(limit * SCALE) if math.isfinite(limit) else np.iinfo(np.int64).max,
    )
    locations = [model.add_location(x=x, y=y) for x, y in instance.locs]
    for position, location in enumerate(locations):
        start, end = instance.time_windows[position]
        window = {"tw_early": math.ceil(start * SCALE)}
        if math.isfinite(end) and not (position == 0 and instance.open_route):
            window["tw_late"] = math.floor(end * SCALE)
        if position == 0:
            model.add_depot(location, **window)
            continue
        model.add_client(
            location,
            delivery=round(instance.demand_linehaul[position] * SCALE),
            pickup=round(instance.demand_backhaul[position] * SCALE),
            service_duration=math.ceil(instance.service_time[position] * SCALE),
            **window,
        )
    for i, origin in enumerate(locations):
        for j, target in enumerate(locations):
            if i == j:
                continue
            if j == 0 and instance.open_route:
                distance = duration = 0
            elif instance.demand_backhaul[i] > 0 and instance.demand_linehaul[j] > 0:
                distance = duration = FORBIDDEN
            else:
                leg = math.dist(instance.locs[i], instance.locs[j])
                distance = math.ceil(leg * SCALE)
                duration = math.ceil(leg / instance.speed * SCALE)
            model.add_edge(origin, target, distance=distance, duration=duration)

    result = model.solve(MaxRuntime(seconds), seed=seed, display=False)
    ending = [] if instance.open_route else [0]
    return [
        # A client activity's idx counts clients from 0; position 0 is the depot.
        [0, *(instance.node_ids[act.idx + 1] for act in route if act.is_client()), *ending]
        for route in result.best.routes()
    ]


def _label(job: tuple[InstanceSet, str, int, float]) -> tuple[list[int], float, bool]:
    instance_set, variant, index, seconds = job
    instance = load_instance(read_request(instance_set, variant, index))
    routes = solve(instance, seconds)
    # The eval files' tour format: customers of each route, then a 0 separator.
    tour = [node for route in routes for node in [*(n for n in route if n != 0), 0]]
    return tour, route_length(instance, routes), not check_routes(instance, routes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="folder with <variant>/100.npz")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    with np.load(root / args.variant / "100.npz") as data:
        count = len(data["locs"])
    instance_set = InstanceSet("train", root, "100.npz", {}, count, [], {})
    jobs = [(instance_set, args.variant, index, args.seconds) for index in range(count)]
    with ProcessPoolExecutor(args.workers) as pool:
        labels = list(pool.map(_label, jobs))

    width = max(len(tour) for tour, _, _ in labels)
    np.savez(
        root / args.variant / "100_sol_pyvrp.npz",
        actions=np.array([tour + [0] * (width - len(tour)) for tour, _, _ in labels]),
        costs=-np.array([cost for _, cost, _ in labels]),
        feasible=np.array([ok for _, _, ok in labels]),
        seconds=np.array(args.seconds),
    )
    feasible = sum(ok for _, _, ok in labels)
    print(f"{args.variant}: {feasible}/{count} feasible -> {root / args.variant}")


if __name__ == "__main__":
    main()
