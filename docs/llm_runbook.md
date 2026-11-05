# LLM backend runbook

How to run `backend=llm`: Qwen2.5-7B-Instruct in bf16 (quantization optional), greedy decoding,
output constrained to compact `{"routes": [[int, ...], ...]}` by lm-format-enforcer. Two enforcer
defaults are overridden: the schema sets `maxItems` (without it every list stops at 20 items),
and tokens containing whitespace are banned (otherwise the model can spend its token budget on
whitespace between JSON tokens). The prompt format is in
`docs/prompt_format.md`; the scope and the no-repair rule are in `docs/paper_notes.md`.

## Requirements

- An NVIDIA GPU with CUDA. About 15 GB of disk for the weights; GPU memory is about 16 GB in
  bf16 and 9.8 GB in 8-bit (measured).
- The `llm` dependency group and the CUDA torch build:

```bash
uv sync --group llm --extra gpu
```

transformers is pinned below 5 because lm-format-enforcer 0.11 imports 4.x internals.

## Download the weights

Pinned to the revision the token counts in `docs/prompt_format.md` were measured with:

```bash
hf download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir artifacts/llm/qwen2.5-7b-instruct
```

The weights stay out of git. On a cluster where compute nodes have no internet, run this on a
login node and point `LLM_MODEL_PATH` at the folder.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL_PATH` | unset | Model folder or cached Hub id. Unset, or a load that fails (logged), keeps the LLM backend off; the MTL backend still serves |
| `LLM_MODEL_VERSION` | `llm/v0.1.0` | Version string on responses, logs, spans, and metrics |
| `LLM_MAX_NEW_TOKENS` | `1024` | Generation cap; hitting it is reported as `truncated` |
| `LLM_QUANTIZATION` | `none` | `none` (bf16), `8bit`, or `4bit` (NF4). Quantization saves memory but decodes more slowly; 4-bit is meant for QLoRA training |
| `LLM_FEW_SHOT` | `0` | Solved examples per prompt (harness only); 0 is zero-shot. Each adds about 2,450 prompt tokens: mean 2,350 / 4,800 / 7,300 / 9,700 tokens for k = 0 / 1 / 2 / 3 |
| `LLM_FEW_SHOT_ROOT` | `data/train/n100` | Labelled training set the examples come from: the first k PyVRP-feasible instances of the request's variant |
| `LLM_ADAPTER_PATH` | unset | LoRA adapter folder merged into the bf16 base at load (needs the `train` group). The adapter is trained on a 4-bit base and scored merged |

## Score it on the smoke set

```bash
LLM_MODEL_PATH=artifacts/llm/qwen2.5-7b-instruct \
  uv run --group llm --extra gpu python -m vrp_service.eval.harness \
  --backend llm --smoke --warmup 1 --out results/llm_smoke.jsonl
```

Each JSONL row has `prompt_tokens` and `completion_tokens` beside the cost, gaps, feasibility,
and latency. `--limit N` runs the first N instances of every variant. LLM runs add an outcome
table (parsed, feasible, infeasible, each error kind, mean tokens);
`--report results/llm_smoke.jsonl` prints the tables again from an earlier run.

On a Slurm cluster, from the repo root with the environment set up:

```bash
sbatch -A <account> -p <gpu partition> scripts/slurm/eval.sbatch \
  --backend llm --smoke --warmup 1 --out results/llm_smoke.jsonl
```

GPU memory is sampled every 30 s into `slurm-<job>.gpu.csv`.

## Training data and few-shot examples

Few-shot examples and the later SFT data come from generated instances, labelled by PyVRP.
They are never eval instances: generation stops if any instance shares coordinates with one.

```bash
uv sync --group data --group mtl --extra gpu
uv run python -m vrp_service.data.generate --per-variant 500 --seed 1 --out data/train/n100
sbatch -A <cpu account> -p <cpu partition> scripts/slurm/label.sbatch data/train/n100
```

- `generate` uses the multi-task environment's generator, the source of the eval instances,
  and writes `<variant>/100.npz` in the eval file format. Variant seed: `seed * 100 + index`.
- `label` solves every instance with PyVRP (20 s, one array task per variant, one instance
  per CPU) into `<variant>/100_sol_pyvrp.npz`: `actions`, `costs`, and `feasible`, the
  result of `check_routes` at the model's tolerance. The PyVRP model mirrors the checker; on
  two eval instances per variant at 5 s, all 32 routes were feasible and within 2% of the
  stored 20 s PyVRP cost.
- `data/train/` is git-ignored; the seed regenerates it.

## Fine-tune (QLoRA SFT)

```bash
uv sync --group train --group llm --group data --extra gpu
sbatch -A <account> -p <gpu partition> scripts/slurm/sft.sbatch \
  --data data/train/n100 --model "$LLM_MODEL_PATH" --out artifacts/llm/adapters/sft-v1
```

- Rows: the eval prompt (`render`) and the PyVRP routes as the compact JSON answer, loss on
  the answer only. The last 10 feasible instances of each variant are held out for eval loss.
- 4-bit NF4 base (double quantization), LoRA rank 64, alpha 64, dropout 0.05, on every
  attention and MLP projection. Defaults: 1 epoch, learning rate 2e-4 (cosine, 3% warm-up),
  batch 4 × 4 accumulation steps, sequences up to 4,096 tokens.
- `--limit N` trains on N rows per variant, for a quick check. Checkpoints every 100 steps;
  resubmitting resumes.
- Score it with `LLM_ADAPTER_PATH=artifacts/llm/adapters/sft-v1` and the eval job script.

## Serve it

```bash
LLM_MODEL_PATH=artifacts/llm/qwen2.5-7b-instruct \
  uv run --group llm --extra gpu uvicorn vrp_service.serving.app:app --port 8000
```

A request with `"backend": "llm"` goes to the LLM; `"mtl"` still goes to the policy. The
response carries `prompt_version`, and `debug` holds `prompt_tokens`, `completion_tokens`, and
`hit_token_limit` (plus `raw_output` with `options.return_debug`).

## Outcomes

| Outcome | `status` | `error_type` | `feasible` |
| --- | --- | --- | --- |
| Valid JSON, routes pass every check | `ok` | – | true |
| Valid JSON, a route breaks a constraint | `ok` | – | false |
| Valid JSON, a node id that does not exist | `ok` (no `objective_value`) | – | false |
| Output stopped at the token cap | `error` | `truncated` | false |
| Not JSON | `error` | `invalid_json` | false |
| JSON of the wrong shape | `error` | `schema` | false |
| Generation raised anything (for example out of memory) | `error` | `generation_failed` | false |
| `LLM_MODEL_PATH` unset | `error` | `model_unavailable` | false |

## Measured

### 8-bit, 2026-11-04

Deucalion, one A100-SXM4-40GB, smoke set (2 instances), 1 warm-up solve.

| | cvrp | ovrpbltw |
| --- | --- | --- |
| Outcome | parsed, infeasible (missed and repeated customers) | parsed, infeasible (one route, five violation kinds) |
| Prompt / output tokens | 1,600 / 348 | 3,094 / 298 |
| Solve time | 40.7 s | 37.1 s |

- Load: 115 s (12 s reading shards; the rest is quantizing and the whitespace-token scan).
- About 8.5 output tokens per second; peak GPU memory 9.8 GB.
- bitsandbytes logs one `MatMul8bitLt` warning per 8-bit matmul (about 195,000 lines here).
- At this speed the full set (1,600 solves) would take about 17 hours. This is why inference
  defaults to bf16.

### bf16, 2026-11-05

Same GPU and smoke set, `LLM_QUANTIZATION=none`.

| | cvrp | ovrpbltw |
| --- | --- | --- |
| Outcome | parsed, infeasible | parsed, infeasible |
| Prompt / output tokens | 1,600 / 325 | 3,094 / 298 |
| Solve time | 8.4 s | 8.0 s |

- About 38 output tokens per second (4.7× the 8-bit run); peak GPU memory 15.7 GB.
- Load: about 105 s, of which importing torch and transformers from the cluster's shared
  filesystem is about 70–85 s and the weights 8–29 s. Once per job, so left as is.
- Full set at this speed: about 3.6 hours.
