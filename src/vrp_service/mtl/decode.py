"""Decode one instance with the multi-task policy.

The network and environment stay outside this package. ``load_policy`` reads a
checkpoint and calls that code. ``decode_request`` only maps the returned tours
back to node ids and keeps the best finite cost. Reward from the environment is
a negative length, so cost is its negation.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from vrp_service.contracts import SolveRequest
from vrp_service.env.instance import EnvironmentInstance, load_instance

Policy = Callable[[EnvironmentInstance], dict]


@dataclass(frozen=True)
class DecodedSolution:
    node_ids: tuple[int, ...]
    cost: float


def select_route(
    node_ids: Sequence[int],
    tours: Sequence[Sequence[int]],
    rewards: Sequence[float],
) -> DecodedSolution:
    if len(tours) == 0 or len(tours) != len(rewards):
        raise ValueError("each tour needs one reward")
    costs = [_cost(reward) for reward in rewards]
    best = min(range(len(costs)), key=lambda index: costs[index])
    route = tuple(node_ids[int(step)] for step in tours[best])
    if len(route) == 0:
        raise ValueError("best tour is empty")
    return DecodedSolution(node_ids=route, cost=costs[best])


def decode_request(request: SolveRequest, policy: Policy) -> DecodedSolution:
    instance = load_instance(request)
    result = policy(instance)
    return select_route(instance.node_ids, result["tours"], result["reward"])


def load_policy(checkpoint_path: str, source_root: str, device: str = "cpu") -> Policy:
    """Return a policy that runs one forward pass from a saved checkpoint."""
    checkpoint = Path(checkpoint_path)
    root = Path(source_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (root / "config.yaml").is_file():
        raise FileNotFoundError(root / "config.yaml")

    def policy(instance: EnvironmentInstance) -> dict:
        return _forward(instance, checkpoint, root, device)

    return policy


def _cost(reward: float) -> float:
    cost = -float(reward)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("cost must be finite and non-negative")
    return cost


def prompt_tag(instance: EnvironmentInstance) -> tuple[float, ...]:
    """Closed, open, windows, limit, backhaul, two zeros, then customers / 2000.

    Reset drops this field, so the caller writes it back onto the batch afterwards.
    """
    customers = len(instance.node_ids) - 1
    has_windows = any(math.isfinite(end) for _start, end in instance.time_windows)
    has_limit = math.isfinite(instance.distance_limit)
    has_backhaul = any(demand > 0 for demand in instance.demand_backhaul)
    return (
        float(not instance.open_route),
        float(instance.open_route),
        float(has_windows),
        float(has_limit),
        float(has_backhaul),
        0.0,
        0.0,
        customers / 2000.0,
    )


def scaled_loads(
    instance: EnvironmentInstance,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Demands divided by capacity, with vehicle capacity set to 1."""
    capacity = instance.vehicle_capacity
    linehaul = tuple(value / capacity for value in instance.demand_linehaul)
    backhaul = tuple(value / capacity for value in instance.demand_backhaul)
    return linehaul, backhaul, 1.0


def decode_model_params(raw: dict) -> dict:
    params = dict(raw)
    params["sqrt_embedding_dim"] = float(params["embedding_dim"]) ** 0.5
    return params


def _forward(
    instance: EnvironmentInstance,
    checkpoint: Path,
    root: Path,
    device: str,
) -> dict:
    import sys

    import torch
    import yaml
    from tensordict import TensorDict

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from envs.mtvrp import MTVRPEnv
    from models.model import VRPModel

    with (root / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    args = SimpleNamespace(
        model_params=decode_model_params(config["model_params"]),
        trainer_params=config.get("trainer_params", {}),
        log=lambda *_args, **_kwargs: None,
    )
    env = MTVRPEnv(
        generator_params=config["env"]["generator_params"],
        device=device,
        check_solution=False,
    )
    model = VRPModel(args)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    td = env.reset(_static_batch(instance, torch, TensorDict, device))
    td["p_s_tag"] = torch.tensor([prompt_tag(instance)], dtype=torch.float32, device=device)
    with torch.inference_mode():
        output = model(td, env)
    reward = output["reward"].detach().cpu().reshape(-1).tolist()
    tours = output["tours"].detach().cpu().tolist()
    return {"reward": reward, "tours": tours}


def _static_batch(instance: EnvironmentInstance, torch, tensor_dict, device: str):
    count = len(instance.node_ids)
    locations = torch.tensor(instance.locs, dtype=torch.float32, device=device).view(1, count, 2)
    windows = torch.tensor(instance.time_windows, dtype=torch.float32, device=device).view(
        1, count, 2
    )

    linehaul, backhaul, capacity = scaled_loads(instance)

    def column(values: Sequence[float]):
        return torch.tensor(values, dtype=torch.float32, device=device).view(1, count)

    def scalar(value: float):
        return torch.tensor([[value]], dtype=torch.float32, device=device)

    return tensor_dict(
        {
            "locs": locations,
            "demand_linehaul": column(linehaul),
            "demand_backhaul": column(backhaul),
            "backhaul_class": scalar(instance.backhaul_class),
            "distance_limit": scalar(instance.distance_limit),
            "service_time": column(instance.service_time),
            "open_route": torch.tensor([[instance.open_route]], dtype=torch.bool, device=device),
            "time_windows": windows,
            "vehicle_capacity": scalar(capacity),
            "capacity_original": scalar(instance.capacity_original),
            "speed": scalar(instance.speed),
        },
        batch_size=[1],
        device=device,
    )
