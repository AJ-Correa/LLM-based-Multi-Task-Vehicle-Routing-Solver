#!/bin/bash
# Run the eval harness as a Slurm job; every argument goes to the harness.
#
#   sbatch -A <account> -p <gpu partition> scripts/slurm/eval.sbatch \
#     --backend llm --smoke --warmup 1 --out results/llm_smoke.jsonl
#
# Submit from the repo root with your environment set up (uv on PATH, LLM_MODEL_PATH, caches):
# sbatch passes the submitting shell's environment to the job. Compute nodes may have no
# internet, so the job runs uv and Hugging Face offline. The log is slurm-<job>.out here.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -J vrp-eval
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 UV_OFFLINE=1
nvidia-smi --query-gpu=name,memory.used --format=csv -l 30 > "slurm-$SLURM_JOB_ID.gpu.csv" &
uv run --frozen --group llm --group mtl --extra gpu python -m vrp_service.eval.harness "$@"
