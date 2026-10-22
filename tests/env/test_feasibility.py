import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from vrp_service.contracts import TRAINING_VARIANTS, SolveRequest
from vrp_service.env import (
    SOLVER_ROUNDING_TOLERANCE,
    EnvironmentInstance,
    check_routes,
    load_instance,
)
from vrp_service.eval import load_instance_set, read_request, read_solution
from vrp_service.serving.settings import project_root
from vrp_service.serving.solve import routes_from_tour

GOLDEN = Path(__file__).parents[1] / "golden"
ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)

# Depot at the origin, customers 1-4 on the x axis.
LINE = EnvironmentInstance(
    node_ids=(0, 1, 2, 3, 4),
    locs=((0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (0.3, 0.0), (0.4, 0.0)),
    demand_linehaul=(0.0, 0.3, 0.3, 0.3, 0.3),
    demand_backhaul=(0.0, 0.0, 0.0, 0.0, 0.0),
    backhaul_class=1.0,
    distance_limit=math.inf,
    service_time=(0.0, 0.0, 0.0, 0.0, 0.0),
    open_route=False,
    time_windows=((0.0, math.inf),) * 5,
    vehicle_capacity=1.0,
    capacity_original=1.0,
    speed=1.0,
)
TWO_ROUTES = [[0, 1, 2, 0], [0, 3, 4, 0]]


def _kinds(instance: EnvironmentInstance, routes) -> list[str]:
    return [violation.kind for violation in check_routes(instance, routes)]


@pytest.mark.parametrize("name", ["cvrp_5", "ovrpbltw_5"])
def test_golden_routes_are_feasible(name: str) -> None:
    request = SolveRequest.model_validate_json((GOLDEN / f"{name}.json").read_text())
    routes = json.loads((GOLDEN / "expected.json").read_text())[name]["routes"]
    assert check_routes(load_instance(request), routes) == []


@pytest.mark.parametrize("solver", ["pyvrp", "ortools"])
@pytest.mark.parametrize("variant", sorted(TRAINING_VARIANTS))
def test_every_stored_solver_solution_is_feasible(variant: str, solver: str) -> None:
    open_route = TRAINING_VARIANTS[variant].open_route
    for index in range(INSTANCE_SET.instances_per_variant):
        instance = load_instance(read_request(INSTANCE_SET, variant, index))
        solution = read_solution(INSTANCE_SET, variant, solver, index)
        routes = [route.node_ids for route in routes_from_tour(solution.actions, open_route)]
        violations = check_routes(instance, routes, SOLVER_ROUNDING_TOLERANCE)
        assert violations == [], (variant, index)


def test_strict_checking_flags_the_rounded_pyvrp_routes() -> None:
    flagged = 0
    for index in range(INSTANCE_SET.instances_per_variant):
        instance = load_instance(read_request(INSTANCE_SET, "vrptw", index))
        solution = read_solution(INSTANCE_SET, "vrptw", "pyvrp", index)
        routes = [route.node_ids for route in routes_from_tour(solution.actions)]
        flagged += bool(check_routes(instance, routes))
    assert flagged > 0


def test_a_feasible_plan_has_no_violations() -> None:
    assert check_routes(LINE, TWO_ROUTES) == []


def test_a_missing_customer_and_a_repeated_one_are_reported() -> None:
    assert _kinds(LINE, [[0, 1, 2, 0], [0, 2, 3, 0]]) == ["visited_twice", "not_visited"]


def test_route_shape_follows_closed_or_open() -> None:
    assert _kinds(LINE, [[0, 1, 2], [0, 3, 4, 0]]) == ["depot_end"]
    assert _kinds(LINE, [[1, 2, 0], [0, 3, 4, 0]]) == ["depot_start"]
    assert _kinds(LINE, [[0, 1, 0, 2, 0], [0, 3, 4, 0]]) == ["depot_inside"]
    assert _kinds(LINE, [*TWO_ROUTES, [0, 0]]) == ["empty_route"]
    assert _kinds(LINE, [[0, 1, 2, 9, 0], [0, 3, 4, 0]]) == ["unknown_node"]
    opened = replace(LINE, open_route=True)
    assert _kinds(opened, [[0, 1, 2], [0, 3, 4]]) == []
    assert _kinds(opened, TWO_ROUTES) == ["depot_end", "depot_end"]


def test_capacity_is_checked_per_route_for_each_load() -> None:
    assert _kinds(LINE, [[0, 1, 2, 3, 4, 0]]) == ["capacity_linehaul"]
    backhaul = replace(LINE, demand_linehaul=(0.0,) * 5, demand_backhaul=LINE.demand_linehaul)
    assert _kinds(backhaul, [[0, 1, 2, 3, 4, 0]]) == ["capacity_backhaul"]


def test_the_capacity_tolerance_scales_with_capacity() -> None:
    large = replace(LINE, vehicle_capacity=1000.0, demand_linehaul=(0.0, 500.0, 500.0005, 1.0, 1.0))
    assert _kinds(large, [[0, 1, 2, 0], [0, 3, 4, 0]]) == []
    tiny = replace(LINE, vehicle_capacity=1e-4, demand_linehaul=(0.0, 5e-5, 5.5e-5, 1e-5, 1e-5))
    assert _kinds(tiny, [[0, 1, 2, 0], [0, 3, 4, 0]]) == ["capacity_linehaul"]


def test_linehaul_customers_come_before_backhaul_customers() -> None:
    mixed = replace(
        LINE,
        demand_linehaul=(0.0, 0.3, 0.0, 0.3, 0.0),
        demand_backhaul=(0.0, 0.0, 0.3, 0.0, 0.3),
    )
    assert _kinds(mixed, [[0, 1, 2, 0], [0, 3, 4, 0]]) == []
    assert _kinds(mixed, [[0, 2, 1, 0], [0, 3, 4, 0]]) == ["backhaul_order"]


def test_a_late_arrival_breaks_the_time_window() -> None:
    windows = ((0.0, 10.0), (0.0, 10.0), (0.0, 0.25), (0.0, 10.0), (0.0, 10.0))
    timed = replace(LINE, time_windows=windows)
    assert _kinds(timed, TWO_ROUTES) == []
    assert _kinds(timed, [[0, 1, 0], [0, 3, 4, 2, 0]]) == ["time_window"]


def test_waiting_and_service_delay_the_next_arrival() -> None:
    windows = ((0.0, 10.0), (0.5, 10.0), (0.0, 0.7), (0.0, 10.0), (0.0, 10.0))
    service = (0.0, 0.1, 0.1, 0.1, 0.1)
    timed = replace(LINE, time_windows=windows, service_time=service)
    # Arrive at 1 at 0.1, wait to 0.5, serve to 0.6, reach 2 at 0.7: just in time.
    assert _kinds(timed, TWO_ROUTES) == []
    late = replace(timed, service_time=(0.0, 0.2, 0.1, 0.1, 0.1))
    assert _kinds(late, TWO_ROUTES) == ["time_window"]


def test_a_closed_route_must_be_back_before_the_depot_closes() -> None:
    windows = ((0.0, 0.5), *((0.0, 10.0),) * 4)
    timed = replace(LINE, time_windows=windows)
    assert _kinds(timed, TWO_ROUTES) == ["time_window"]
    assert _kinds(replace(timed, open_route=True), [[0, 1, 2], [0, 3, 4]]) == []


def test_the_distance_limit_counts_the_return_leg_unless_the_route_is_open() -> None:
    limited = replace(LINE, distance_limit=0.5)
    assert _kinds(limited, [[0, 1, 2, 0], [0, 3, 4, 0]]) == ["distance_limit"]
    opened = replace(limited, open_route=True)
    assert _kinds(opened, [[0, 1, 2], [0, 3, 4]]) == []
