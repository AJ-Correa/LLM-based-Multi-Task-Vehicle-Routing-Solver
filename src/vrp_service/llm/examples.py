"""Pick few-shot examples from the PyVRP-labelled training set, never from the eval set."""

from pathlib import Path

import numpy as np

from vrp_service.contracts import TRAINING_VARIANTS
from vrp_service.eval.instance_set import InstanceSet, read_request
from vrp_service.llm.prompt import Example
from vrp_service.serving.solve import routes_from_tour


def load_examples(root: str | Path, count: int | None = None) -> dict[str, list[Example]]:
    """The first ``count`` (default: all) feasible labelled instances of every variant."""
    root = Path(root)
    examples: dict[str, list[Example]] = {}
    for variant, flags in TRAINING_VARIANTS.items():
        with np.load(root / variant / "100_sol_pyvrp.npz") as solved:
            indices = np.flatnonzero(solved["feasible"])[:count].tolist()
            tours = [solved["actions"][index].tolist() for index in indices]
        if count is not None and len(indices) < count:
            raise ValueError(f"{variant}: only {len(indices)} feasible labelled instances")
        training = InstanceSet("train", root, "100.npz", {}, 0, [], {})
        examples[variant] = [
            (
                read_request(training, variant, index, "llm"),
                [
                    [node for node in route.node_ids if node != 0]
                    for route in routes_from_tour(tour, flags.open_route)
                ],
            )
            for index, tour in zip(indices, tours)
        ]
    return examples
