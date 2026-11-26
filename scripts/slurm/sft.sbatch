#!/bin/bash
# QLoRA SFT on one GPU; every argument goes to vrp_service.train.sft.
#
#   sbatch -A <account> -p <gpu partition> scripts/slurm/sft.sbatch \
#     --data data/train/n100 --model "$LLM_MODEL_PATH" --out artifacts/llm/adapters/sft-v1
#
# Submit from the repo root with the environment set up. Checkpoints are saved every 100
# steps; resubmitting the same command resumes from the newest one after a time limit.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 2-00:00:00
#SBATCH -J vrp-sft
#SBATCH -o slurm-%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1 UV_OFFLINE=1
nvidia-smi --query-gpu=name,memory.used --format=csv -l 60 > "slurm-$SLURM_JOB_ID.gpu.csv" &
uv run --frozen --group train --group llm --group data --extra gpu python -m vrp_service.train.sft "$@"
