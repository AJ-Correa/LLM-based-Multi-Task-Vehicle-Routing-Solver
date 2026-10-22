"""Check returned routes against the variant's constraints, the way the environment masks them.

Routes use the response shape: each starts at the depot (node 0); a closed route
ends at the depot and an open route ends at its last customer. Features the
variant leaves off carry their inactive defaults (infinite window end and
distance limit, zero backhaul demand), so the same checks run for every variant.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from vrp_service.env.instance import EnvironmentInstance

# The environment allows 1e-6 in float32; a little more absorbs float32 inputs.
TOLERANCE = 1e-5
# Stored PyVRP routes were solved on distances rounded to 3 decimals, so each leg can
# be 5e-4 short and a route can overshoot a window or distance limit. The worst seen
# over the full 1,000-instance source sets is 0.0026; this covers about 20 legs.
SOLVER_ROUNDING_TOLERANCE = 1e-2


@dataclass(frozen=True)
class Violation:
    kind: str
    route: int | None = None
    node: int | None = None


def check_routes(
    instance: EnvironmentInstance,
    routes: Sequence[Sequence[int]],
    tolerance: float = TOLERANCE,
) -> list[Violation]:
    """Every broken constraint; an empty list means the routes are feasible."""
    index = {node: position for position, node in enumerate(instance.node_ids)}
    violations: list[Violation] = []
    visits: Counter[int] = Counter()
    for number, route in enumerate(routes):
        customers, shape = _customers(instance, index, number, list(route))
        violations += shape
        visits.update(customers)
        if not shape:
            violations += _check_route(instance, index, number, customers, tolerance)
    for node in instance.node_ids[1:]:
        if visits[node] == 0:
            violations.append(Violation("not_visited", node=node))
        elif visits[node] > 1:
            violations.append(Violation("visited_twice", node=node))
    return violations


def _customers(
    instance: EnvironmentInstance, index: dict[int, int], number: int, route: list[int]
) -> tuple[list[int], list[Violation]]:
    violations = []
    starts_at_depot = bool(route) and route[0] == 0
    if not starts_at_depot:
        violations.append(Violation("depot_start", number))
    body = route[1:] if starts_at_depot else route
    ends_at_depot = bool(body) and body[-1] == 0
    if ends_at_depot == instance.open_route:
        violations.append(Violation("depot_end", number))
    inner = body[:-1] if ends_at_depot else body
    customers = [node for node in inner if node != 0]
    if len(customers) != len(inner):
        violations.append(Violation("depot_inside", number))
    if not customers:
        violations.append(Violation("empty_route", number))
    unknown = [node for node in customers if node not in index]
    violations += [Violation("unknown_node", number, node) for node in unknown]
    return [node for node in customers if node in index], violations


def _check_route(
    instance: EnvironmentInstance,
    index: dict[int, int],
    number: int,
    customers: list[int],
    tolerance: float,
) -> list[Violation]:
    violations = []
    positions = [index[node] for node in customers]
    # Relative, like the environment, which compares demand / capacity with 1.
    capacity = instance.vehicle_capacity * (1 + tolerance)
    if sum(instance.demand_linehaul[p] for p in positions) > capacity:
        violations.append(Violation("capacity_linehaul", number))
    if sum(instance.demand_backhaul[p] for p in positions) > capacity:
        violations.append(Violation("capacity_backhaul", number))

    # Classical backhaul: no linehaul customer after a backhaul customer.
    seen_backhaul = False
    for node, p in zip(customers, positions):
        if instance.demand_backhaul[p] > 0:
            seen_backhaul = True
        elif seen_backhaul and instance.demand_linehaul[p] > 0:
            violations.append(Violation("backhaul_order", number, node))
            break

    stops = [0, *positions] if instance.open_route else [0, *positions, 0]
    length = 0.0
    clock = 0.0
    late = None
    for previous, current in zip(stops, stops[1:]):
        leg = math.dist(instance.locs[previous], instance.locs[current])
        length += leg
        clock += leg / instance.speed
        start, end = instance.time_windows[current]
        if late is None and clock > end + tolerance:
            late = instance.node_ids[current]
        clock = max(clock, start) + instance.service_time[current]
    if late is not None:
        violations.append(Violation("time_window", number, late))
    if length > instance.distance_limit + tolerance:
        violations.append(Violation("distance_limit", number))
    return violations
