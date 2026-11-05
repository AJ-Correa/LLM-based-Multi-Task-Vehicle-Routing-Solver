"""Generate training instances for the 16 single-depot variants, in the eval set's file format.

    python -m vrp_service.data.generate --per-variant 500 --seed 1 --out data/train/n100

Uses the multi-task environment's generator (``runtime/envs/mtvrp``), the one the eval
instances come from, so both follow the same distribution. Each variant gets its own
seed, ``seed * 100 + its position in TRAINING_VARIANTS``, so a variant can be regenerated
alone. No generated instance may share coordinates with an eval instance; that is checked
before anything is written.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

from vrp_service.contracts import TRAINING_VARIANTS
from vrp_service.eval.instance_set import load_instance_set
from vrp_service.serving.settings import project_root

CUSTOMERS = 100
# Arrays the eval files store per node without the depot.
CUSTOMER_ONLY = ("demand_linehaul", "demand_backhaul")


def variant_seed(seed: int, variant: str) -> int:
    return seed * 100 + list(TRAINING_VARIANTS).index(variant)


def generate(variant: str, count: int, seed: int) -> dict[str, np.ndarray]:
    import torch

    runtime = str(project_root() / "runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    from envs.mtvrp.generator import MTVRPGenerator

    torch.manual_seed(variant_seed(seed, variant))
    batch = MTVRPGenerator(num_loc=CUSTOMERS, variant_preset=variant)(count)
    arrays = {key: value.cpu().numpy() for key, value in batch.items()}
    for key in CUSTOMER_ONLY:
        arrays[key] = arrays[key][:, 1:]
    return arrays


def fingerprints(locs: np.ndarray) -> set[str]:
    return {hashlib.sha256(np.ascontiguousarray(one).tobytes()).hexdigest() for one in locs}


def eval_fingerprints() -> set[str]:
    root = project_root()
    instance_set = load_instance_set(root / "configs/eval/instance_set_v1.yaml", root)
    found: set[str] = set()
    for variant in instance_set.sha256:
        with np.load(instance_set.path(variant, instance_set.instance_file)) as data:
            found |= fingerprints(data["locs"])
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-variant", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", nargs="+", choices=list(TRAINING_VARIANTS))
    args = parser.parse_args(argv)

    held_out = eval_fingerprints()
    batches = {
        variant: generate(variant, args.per_variant, args.seed)
        for variant in args.variants or TRAINING_VARIANTS
    }
    for variant, arrays in batches.items():
        shared = fingerprints(arrays["locs"]) & held_out
        if shared:
            raise SystemExit(f"{variant}: {len(shared)} instances match eval instances")
    for variant, arrays in batches.items():
        folder = Path(args.out) / variant
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"{CUSTOMERS}.npz", **arrays)
        print(f"{variant}: {args.per_variant} instances -> {folder}")


if __name__ == "__main__":
    main()
