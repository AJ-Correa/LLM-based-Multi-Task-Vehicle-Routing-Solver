import math

import pytest

from vrp_service.contracts import SolveRequest
from vrp_service.env import load_instance
from vrp_service.mtl.decode import (
    decode_model_params,
    decode_request,
    load_policy,
    prompt_tag,
    scaled_loads,
    select_route,
)


def _cvrp() -> SolveRequest:
    return SolveRequest.model_validate(
        {
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
        }
    )


def test_select_route_keeps_the_best_finite_cost():
    solution = select_route(
        node_ids=(0, 1, 2),
        tours=((0, 1, 2, 0), (0, 2, 1, 0)),
        rewards=(-4.0, -3.5),
    )
    assert solution.node_ids == (0, 2, 1, 0)
    assert solution.cost == 3.5


def test_select_route_rejects_a_non_finite_reward():
    with pytest.raises(ValueError, match="finite"):
        select_route((0, 1), ((0, 1),), (math.nan,))


def test_decode_request_maps_indices_through_a_fake_policy():
    def policy(instance):
        assert instance.node_ids[0] == 0
        return {"reward": [-2.0], "tours": [[0, 1, 0]]}

    solution = decode_request(_cvrp(), policy)
    assert solution.node_ids == (0, 1, 0)
    assert solution.cost == 2.0


def test_load_policy_requires_a_checkpoint_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(str(tmp_path / "missing.pt"), str(tmp_path))


def test_prompt_tag_marks_closed_cvrp_and_full_variant():
    cvrp = load_instance(_cvrp())
    assert prompt_tag(cvrp) == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / 2000)

    full = load_instance(
        SolveRequest.model_validate(
            {
                "backend": "mtl",
                "variant": "ovrpbltw",
                "seed": 0,
                "instance": {
                    "nodes": [
                        {"id": 0, "x": 0.0, "y": 0.0, "demand_linehaul": 0, "demand_backhaul": 0},
                        {"id": 1, "x": 0.2, "y": 0.3, "demand_linehaul": 1, "demand_backhaul": 0},
                        {"id": 2, "x": 0.4, "y": 0.5, "demand_linehaul": 0, "demand_backhaul": 1},
                    ],
                    "capacity": 10,
                    "open_route": True,
                    "time_windows": [
                        {"start": 0, "end": 10},
                        {"start": 1, "end": 6},
                        {"start": 1, "end": 4},
                    ],
                    "service_times": [0, 0.1, 0.2],
                    "distance_limit": 3.0,
                },
            }
        )
    )
    assert prompt_tag(full) == (0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2 / 2000)


def test_demands_are_divided_by_capacity():
    linehaul, backhaul, capacity = scaled_loads(load_instance(_cvrp()))
    assert linehaul == (0.0, 0.1)
    assert backhaul == (0.0, 0.0)
    assert capacity == 1.0


def test_logit_scale_is_the_square_root_of_the_embedding():
    params = decode_model_params({"embedding_dim": 128, "head_num": 8})
    assert params["sqrt_embedding_dim"] == 128**0.5
    assert params["head_num"] == 8
