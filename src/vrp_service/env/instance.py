"""Turn a solve request into the static fields the environment resets from.

Node 0 stays the depot. A feature that the variant leaves off uses the inactive
default: time-window end infinity, service time 0, and distance limit infinity.
Classical backhaul is class 1.
"""

import math
from dataclasses import dataclass

from vrp_service.contracts import SolveRequest
from vrp_service.contracts.variants import TRAINING_VARIANTS


@dataclass(frozen=True)
class EnvironmentInstance:
    node_ids: tuple[int, ...]
    locs: tuple[tuple[float, float], ...]
    demand_linehaul: tuple[float, ...]
    demand_backhaul: tuple[float, ...]
    backhaul_class: float
    distance_limit: float
    service_time: tuple[float, ...]
    open_route: bool
    time_windows: tuple[tuple[float, float], ...]
    vehicle_capacity: float
    capacity_original: float
    speed: float


def load_instance(request: SolveRequest) -> EnvironmentInstance:
    flags = TRAINING_VARIANTS[request.variant]
    instance = request.instance
    depot = instance.nodes[0]
    customers = sorted(instance.nodes[1:], key=lambda node: node.id)
    nodes = [depot, *customers]

    windows_by_id = {}
    service_by_id = {}
    if instance.time_windows is not None and instance.service_times is not None:
        for node, window, service in zip(
            instance.nodes, instance.time_windows, instance.service_times, strict=True
        ):
            windows_by_id[node.id] = (window.start, window.end)
            service_by_id[node.id] = service

    if flags.time_windows:
        time_windows = tuple(windows_by_id[node.id] for node in nodes)
        service_time = tuple(service_by_id[node.id] for node in nodes)
    else:
        time_windows = tuple((0.0, math.inf) for _ in nodes)
        service_time = tuple(0.0 for _ in nodes)

    distance_limit = instance.distance_limit if flags.distance_limit else math.inf
    if distance_limit is None:
        distance_limit = math.inf

    return EnvironmentInstance(
        node_ids=tuple(node.id for node in nodes),
        locs=tuple((node.x, node.y) for node in nodes),
        demand_linehaul=tuple(node.demand_linehaul for node in nodes),
        demand_backhaul=tuple(node.demand_backhaul for node in nodes),
        backhaul_class=1.0,
        distance_limit=distance_limit,
        service_time=service_time,
        open_route=instance.open_route,
        time_windows=time_windows,
        vehicle_capacity=instance.capacity,
        capacity_original=instance.capacity,
        speed=instance.speed,
    )
