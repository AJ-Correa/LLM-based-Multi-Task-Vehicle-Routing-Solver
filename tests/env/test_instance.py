import math

from vrp_service.contracts import TRAINING_VARIANTS, SolveRequest
from vrp_service.env import load_instance


def _request(variant: str) -> SolveRequest:
    flags = TRAINING_VARIANTS[variant]
    second_linehaul = 0.0 if flags.backhaul else 1.0
    second_backhaul = 1.0 if flags.backhaul else 0.0
    return SolveRequest.model_validate(
        {
            "backend": "mtl",
            "variant": variant,
            "seed": 0,
            "instance": {
                "nodes": [
                    {"id": 0, "x": 0.0, "y": 0.0, "demand_linehaul": 0, "demand_backhaul": 0},
                    {
                        "id": 2,
                        "x": 0.4,
                        "y": 0.5,
                        "demand_linehaul": second_linehaul,
                        "demand_backhaul": second_backhaul,
                    },
                    {
                        "id": 1,
                        "x": 0.2,
                        "y": 0.3,
                        "demand_linehaul": 1,
                        "demand_backhaul": 0,
                    },
                ],
                "capacity": 10,
                "open_route": flags.open_route,
                "time_windows": [
                    {"start": 0, "end": 10},
                    {"start": 1, "end": 4},
                    {"start": 2, "end": 6},
                ]
                if flags.time_windows
                else None,
                "service_times": [0, 0.2, 0.1] if flags.time_windows else None,
                "distance_limit": 3.0 if flags.distance_limit else None,
                "speed": 1.0,
            },
        }
    )


def test_cvrp_uses_inactive_defaults() -> None:
    loaded = load_instance(_request("cvrp"))
    assert loaded.node_ids == (0, 1, 2)
    assert loaded.locs[0] == (0.0, 0.0)
    assert loaded.demand_linehaul[0] == 0
    assert loaded.demand_backhaul == (0, 0, 0)
    assert loaded.backhaul_class == 1.0
    assert loaded.open_route is False
    assert loaded.vehicle_capacity == 10
    assert loaded.capacity_original == 10
    assert math.isinf(loaded.distance_limit)
    assert loaded.service_time == (0.0, 0.0, 0.0)
    assert all(window == (0.0, math.inf) for window in loaded.time_windows)


def test_ovrpbltw_keeps_active_features() -> None:
    loaded = load_instance(_request("ovrpbltw"))
    assert loaded.open_route is True
    assert loaded.distance_limit == 3.0
    assert loaded.backhaul_class == 1.0
    assert loaded.node_ids == (0, 1, 2)
    assert loaded.time_windows[2] == (1.0, 4.0)
    assert loaded.service_time[2] == 0.2
    assert loaded.demand_backhaul[2] == 1.0
    assert loaded.demand_linehaul[1] == 1.0
