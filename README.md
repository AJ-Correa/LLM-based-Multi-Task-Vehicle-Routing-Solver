# LLM-based VRP decision support

[![ci](https://github.com/AJ-Correa/LLM-based-Multi-Task-Vehicle-Routing-Solver/actions/workflows/ci.yml/badge.svg)](https://github.com/AJ-Correa/LLM-based-Multi-Task-Vehicle-Routing-Solver/actions/workflows/ci.yml)

A FastAPI service that solves vehicle routing problems behind one HTTP contract, with two
backends scored on the same frozen benchmark:

- **`mtl`**: a multi-task neural routing policy that covers 16 single-depot variants with one
  model. 100% feasible, +2.9% to PyVRP, about 0.55 s per 100-customer instance.
- **`llm`**: Qwen2.5-7B-Instruct, scored before and after QLoRA fine-tuning, with and without
  feasibility-masked decoding. An engineering study inspired by LLMCoSolver [1].

## Results

1,600 instances (16 variants × 100, 100 customers each). Gaps are averaged over feasible
answers against PyVRP run for 20 s per instance. LLM outputs are never repaired.

| Backend | Feasible | Gap to PyVRP | Time per solve |
| --- | --- | --- | --- |
| MTL policy | 100% | +2.9% | 0.55 s |
| Qwen2.5-7B zero-shot | 0.1% | – | 8.0 s |
| Qwen2.5-7B zero-shot + masked decoding | 100% | +241.5% | 7.9 s |
| Qwen2.5-7B QLoRA SFT (80k instances) | 0.4% | +11.3% (7 answers) | 8.2 s |
| Qwen2.5-7B QLoRA SFT + masked decoding | 100% | +24.1% | 7.5 s |

- Fine-tuning helps a lot: with masked decoding, the gap to PyVRP falls from +241.5%
  (zero-shot) to +24.1%, ten times smaller. Going from 500 to 5,000 training instances per
  variant, with nearest-customer hints in the prompt, halved it from +51.9%.
- Masked decoding (an action mask over tokens, not a repair step) makes every answer feasible
  and isolates the quality of the model's routing choices. Unmasked, the fine-tuned model
  still almost never writes a feasible plan: capacity and time windows break most answers.
- Used as an end-to-end solver, a 7B LLM remains far from classical optimisation algorithms
  (PyVRP, OR-Tools) and neural routing policies, both on solution quality and on speed.

Every row, per-variant tables, and the error breakdown are in
[`docs/results.md`](docs/results.md) (MTL) and [`docs/llm_runbook.md`](docs/llm_runbook.md)
(LLM).

## Variants

Each variant is one combination of open routes (O), time windows (TW), a distance limit (L), and
classical backhaul (B), with one depot:

`cvrp`, `ovrp`, `vrpb`, `vrpl`, `vrptw`, `ovrptw`, `ovrpb`, `ovrpl`, `vrpbl`, `vrpbtw`,
`vrpltw`, `ovrpbl`, `ovrpbtw`, `ovrpltw`, `vrpbltw`, `ovrpbltw`

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Docker is optional.

The model weights are not in this repository. Place the checkpoint at
`artifacts/checkpoints/n100/checkpoint-300.pt` and check it against the manifest:

```bash
sha256sum artifacts/checkpoints/n100/checkpoint-300.pt
# f77a8e334e84d2ad1e2ea8ce9ca46778d61db5339175c56a08dddf63ca890749
```

Without the checkpoint the API still starts, `/ready` returns `false`, and `/solve` answers with
`"error_type": "model_unavailable"`.

## Run

Locally, with the CUDA build of torch:

```bash
make api                        # uv run --group mtl --extra gpu uvicorn ... on port 8000
```

Use `--extra cpu` instead of `--extra gpu` for a CPU-only torch. The policy runs on CPU unless
`MTL_DEVICE=cuda` is set.

With Docker (CPU image; the checkpoint is mounted read-only from `artifacts/checkpoints/`):

```bash
make docker                     # docker compose -f docker/docker-compose.yml up --build
```

## Solve an instance

```bash
curl -X POST http://localhost:8000/solve \
  -H 'Content-Type: application/json' \
  -d @tests/golden/cvrp_5.json
```

```json
{
  "contract_version": "1.0.0",
  "backend": "mtl",
  "model_version": "mtl/v0.1.0",
  "feasible": true,
  "objective_value": 1.856235384941101,
  "routes": [
    {"vehicle_id": 0, "node_ids": [0, 3, 4, 1, 5, 0]},
    {"vehicle_id": 1, "node_ids": [0, 2, 0]}
  ],
  "latency_ms": 25.2,
  "status": "ok",
  "error_type": null
}
```

Trimmed; the full response also carries `prompt_version`, `objective_sense`, `gap_to_baseline`,
`baseline_name`, and `debug`. Latency depends on the machine.

In a request, node 0 is the depot. Each customer has `demand_linehaul` and `demand_backhaul`.
The variant decides which other fields are required: `time_windows` and `service_times` (one
per node) for TW, `distance_limit` for L, and `open_route` must match O. A request whose fields
do not match its variant is rejected with `422`. `tests/golden/ovrpbltw_5.json` is a request
with every feature on. The OpenAPI page at `/docs` has the schema and an example.

Closed routes end at the depot. Open routes end at their last customer. `feasible` comes from a
separate checker (`vrp_service.env.check_routes`) that tests every route for capacity, visiting
each customer once, the depot start and end, time windows, the distance limit, and linehaul
before backhaul.

## Endpoints

| Method | Path      | Returns                                         |
| ------ | --------- | ----------------------------------------------- |
| GET    | `/health` | `{"status": "ok"}` once the process is up       |
| GET    | `/ready`  | `{"ready": true}` when the checkpoint is on disk |
| POST   | `/solve`  | Routes, cost, model version, latency            |
| GET    | `/models` | Registered versions, the active one, file checks |
| GET    | `/metrics`| Prometheus metrics                              |

## Configuration

| Variable            | Default                                       |
| ------------------- | --------------------------------------------- |
| `MTL_CHECKPOINT`    | `artifacts/checkpoints/n100/checkpoint-300.pt` |
| `MTL_SOURCE`        | `runtime/` (environment, network, config)     |
| `MTL_MODEL_VERSION` | `mtl/v0.1.0`                                  |
| `MTL_DEVICE`        | `cpu`                                         |
| `MTL_CPU_THREADS`   | `4`                                           |
| `MODEL_MANIFESTS`   | `artifacts/manifests`                         |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset (no trace export)             |
| `OTEL_TRACES_EXPORTER` | unset; `console` prints spans              |

The policy loads once at startup.

## Eval set

`data/eval/n100/` holds 100 instances for each of the 16 variants, 100 customers each, with
reference PyVRP and OR-Tools solutions (20 seconds per instance).
`configs/eval/instance_set_v1.yaml` lists every file with its SHA-256, and a test checks them.
`data/eval/n100/README.md` describes the arrays.

To score the policy on it:

```bash
MTL_DEVICE=cuda uv run --group mtl --extra gpu python -m vrp_service.eval.harness --out results/mtl.jsonl
```

This writes one JSONL row per instance (routes, cost, feasibility, violations, gap to PyVRP and
OR-Tools, latency) and a markdown table next to it. `--smoke` runs two instances, `--limit N`
the first N per variant, and `--mode http --url …` scores a running server. Current numbers are
in [`docs/results.md`](docs/results.md): 100% feasible, +2.9% to PyVRP, about 0.55 s per solve.

## Observability and tracking

Each solve emits a trace, Prometheus metrics, and a JSON log line. Optional compose profiles:

```bash
# Prometheus on :9090 and Jaeger on :16686
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318 docker compose -f docker/docker-compose.yml --profile obs up --build
# MLflow on :5000 with its own Postgres
docker compose -f docker/docker-compose.yml --profile mlflow up -d --build
```

[`docs/design.md`](docs/design.md) has a sample trace, the metric names, and how to add a model
version.

## Model registry

`artifacts/manifests/mtl/v0.1.0.json` records the checkpoint and config with their SHA-256.
`GET /models` lists every valid manifest, marks the served version, and checks the files on
disk. A test fails if the config, or the checkpoint when present, no longer matches.

## Tests

```bash
make test
make lint
```

`tests/golden/` holds two 5-customer instances. CI checks the exact policy inputs they produce.
With torch and the checkpoint installed, the same tests also check the stored routes and cost:

```bash
uv run --group mtl --extra gpu pytest -q
```

## LLM backend

Set `LLM_MODEL_PATH` to a local Qwen2.5-7B-Instruct folder (and `LLM_ADAPTER_PATH` to a LoRA
adapter) and `POST /solve` accepts `"backend": "llm"`. The prompt format, training data
generation, QLoRA training, masked decoding, and Slurm scripts are documented in
[`docs/llm_runbook.md`](docs/llm_runbook.md) and [`docs/prompt_format.md`](docs/prompt_format.md).
Weights and adapters stay out of git.

## Full LLM experiment

Needs an NVIDIA GPU and the Qwen2.5-7B-Instruct weights (download command in
[`docs/llm_runbook.md`](docs/llm_runbook.md#download-the-weights)). Run from the repo root:

```bash
uv sync --group train --group llm --group data --group mtl --extra gpu
export LLM_MODEL_PATH=artifacts/llm/qwen2.5-7b-instruct
EVAL="uv run python -m vrp_service.eval.harness --backend llm --warmup 1"

# 1. Training data: 5,000 instances per variant (80,000), labelled by PyVRP (20 s each)
uv run python -m vrp_service.data.generate --per-variant 5000 --seed 2 --out data/train/n100_5k
for v in cvrp ovrp vrpb vrpl vrptw ovrptw ovrpb ovrpl vrpbl vrpbtw vrpltw ovrpbl ovrpbtw ovrpltw vrpbltw ovrpbltw; do
  uv run python -m vrp_service.data.label --root data/train/n100_5k --variant $v --seconds 20
done

# 2. QLoRA SFT (rerunning the same command resumes from the last checkpoint)
uv run python -m vrp_service.train.sft \
  --data data/train/n100_5k --model "$LLM_MODEL_PATH" --out artifacts/llm/adapters/sft-v2

# 3. Evaluation on all 16 variants x 100 instances, one row per command
$EVAL --out results/llm_zero_shot.jsonl
LLM_FEW_SHOT=2 $EVAL --out results/llm_few_shot_2.jsonl
LLM_MASKED=1 $EVAL --out results/llm_zero_shot_masked.jsonl
LLM_FEW_SHOT=2 LLM_MASKED=1 $EVAL --out results/llm_few_shot_2_masked.jsonl
LLM_ADAPTER_PATH=artifacts/llm/adapters/sft-v2 $EVAL --out results/llm_lora.jsonl
LLM_ADAPTER_PATH=artifacts/llm/adapters/sft-v2 LLM_MASKED=1 $EVAL --out results/llm_lora_masked.jsonl

# 4. Tables
uv run python -m vrp_service.eval.harness --report results/llm_lora_masked.jsonl
```

Labelling and fine-tuning are the long steps; each eval row is 1,600 generations.
`scripts/slurm/` wraps the same commands as Slurm jobs (labelling as a 16-task array).

## Key hyper-parameters

| | Setting | Where to change it |
| --- | --- | --- |
| Base model | Qwen2.5-7B-Instruct, revision `a09a354` | `LLM_MODEL_PATH`, `--model` |
| Quantisation (training) | 4-bit NF4, double quantisation, bf16 compute | `src/vrp_service/train/sft.py` |
| LoRA | rank 64, alpha 64, dropout 0.05, every attention and MLP projection | `src/vrp_service/train/sft.py` |
| Optimisation | 1 epoch, learning rate 2e-4, cosine schedule, 3% warm-up | `--epochs`, `--lr`; schedule in `sft.py` |
| Batch | 4 per device × 4 gradient accumulation = 16 | `--batch`, `--grad-accum` |
| Sequence length | up to 5,120 tokens; loss on the answer only | `--max-length` |
| Data | 5,000 PyVRP-labelled instances per variant, 10 per variant held out | `generate --per-variant`, `label --seconds`, `sft --holdout` |
| Prompt | v2: integer node table with the two nearest customers per node | `PROMPT_VERSION` in `src/vrp_service/llm/prompt.py`, text in `configs/prompts/` |
| Inference | bf16, adapter merged, greedy decoding, up to 1,024 new tokens | `LLM_QUANTIZATION`, `LLM_ADAPTER_PATH`, `LLM_MAX_NEW_TOKENS` |
| Few-shot / masking | k = 2 solved examples of the same variant; masked decoding on or off | `LLM_FEW_SHOT`, `LLM_MASKED` |

`--…` flags belong to `vrp_service.train.sft` (`--help` lists them); `LLM_…` are environment
variables read by the eval harness and the API ([`docs/llm_runbook.md`](docs/llm_runbook.md#settings)).

## References

[1] X. Jiang, Y. Wu, M. Li, Z. Cao, and Y. Zhang. *Large Language Models as End-to-end
Combinatorial Optimization Solvers.* NeurIPS 2025.
[arXiv:2509.16865](https://arxiv.org/abs/2509.16865)
