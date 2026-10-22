"""Read the frozen eval set and turn one stored instance into a solve request.

The files keep the arrays of the original test sets: demands are already divided
by capacity, so capacity is 1, and stored solution costs are rewards (negative
lengths).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from vrp_service.contracts import TRAINING_VARIANTS, SolveRequest


@dataclass(frozen=True)
class InstanceSet:
    root: Path
    instance_file: str
    solution_files: dict[str, str]
    instances_per_variant: int
    smoke: list[tuple[str, int]]
    sha256: dict[str, dict[str, str]]

    def path(self, variant: str, file_name: str) -> Path:
        return self.root / variant / file_name


@dataclass(frozen=True)
class StoredSolution:
    actions: list[int]
    cost: float


def load_instance_set(config_path: str | Path, project_root: str | Path) -> InstanceSet:
    """``root`` in the YAML is relative to the project root."""
    config = yaml.safe_load(Path(config_path).read_text())
    return InstanceSet(
        root=Path(project_root) / config["root"],
        instance_file=config["instance_file"],
        solution_files=dict(config["solution_files"]),
        instances_per_variant=int(config["instances_per_variant"]),
        smoke=[(entry["variant"], int(entry["index"])) for entry in config["smoke"]],
        sha256={variant: dict(files) for variant, files in config["sha256"].items()},
    )


def read_request(
    instance_set: InstanceSet, variant: str, index: int, backend: str = "mtl"
) -> SolveRequest:
    flags = TRAINING_VARIANTS[variant]
    with np.load(instance_set.path(variant, instance_set.instance_file)) as data:
        locs = data["locs"][index].tolist()
        linehaul = [0.0, *data["demand_linehaul"][index].tolist()]
        backhaul = (
            [0.0, *data["demand_backhaul"][index].tolist()]
            if "demand_backhaul" in data.files
            else [0.0] * len(locs)
        )
        instance = {
            "nodes": [
                {
                    "id": node,
                    "x": x,
                    "y": y,
                    "demand_linehaul": linehaul[node],
                    "demand_backhaul": backhaul[node],
                }
                for node, (x, y) in enumerate(locs)
            ],
            "capacity": float(data["vehicle_capacity"][index][0]),
            "open_route": flags.open_route,
            "speed": float(data["speed"][index][0]),
        }
        if flags.time_windows:
            instance["time_windows"] = [
                {"start": start, "end": end} for start, end in data["time_windows"][index].tolist()
            ]
            instance["service_times"] = data["service_time"][index].tolist()
        if flags.distance_limit:
            instance["distance_limit"] = float(data["distance_limit"][index][0])
    return SolveRequest.model_validate(
        {"backend": backend, "variant": variant, "seed": index, "instance": instance}
    )


def read_solution(
    instance_set: InstanceSet, variant: str, solver: str, index: int
) -> StoredSolution:
    with np.load(instance_set.path(variant, instance_set.solution_files[solver])) as data:
        return StoredSolution(
            actions=data["actions"][index].tolist(),
            cost=-float(data["costs"][index]),
        )
