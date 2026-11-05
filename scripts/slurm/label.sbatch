#!/bin/bash
# Label generated training instances with PyVRP: one array task per variant.
#
#   sbatch -A <cpu account> -p <cpu partition> scripts/slurm/label.sbatch data/train/n100
#
# Submit from the repo root with the environment set up, after
# `python -m vrp_service.data.generate` has written <root>/<variant>/100.npz.
# Each task solves its variant's instances in parallel on its CPUs (20 s each).
#SBATCH --array=0-15
#SBATCH --cpus-per-task=32
#SBATCH -t 02:00:00
#SBATCH -J vrp-label
#SBATCH -o slurm-%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_OFFLINE=1
VARIANTS=(cvrp ovrp vrpb vrpl vrptw ovrptw ovrpb ovrpl vrpbl vrpbtw vrpltw ovrpbl ovrpbtw ovrpltw vrpbltw ovrpbltw)
uv run --frozen --group data --group mtl --extra gpu python -m vrp_service.data.label \
  --root "$1" --variant "${VARIANTS[$SLURM_ARRAY_TASK_ID]}" --seconds 20
