# LLM backend runbook

How to run `backend=llm`: Qwen2.5-7B-Instruct in 8-bit (bitsandbytes), greedy decoding, output
constrained to compact `{"routes": [[int, ...], ...]}` by lm-format-enforcer. Two enforcer
defaults are overridden: the schema sets `maxItems` (without it every list stops at 20 items),
and tokens containing whitespace are banned (otherwise the model can spend its token budget on
whitespace between JSON tokens). The prompt format is in
`docs/prompt_format.md`; the scope and the no-repair rule are in `docs/paper_notes.md`.

## Requirements

- An NVIDIA GPU with CUDA. 8-bit loading does not run on CPU.
- About 15 GB of disk for the weights and roughly 8–9 GB of GPU memory for them (to be
  replaced by the measured peak after the first smoke run).
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

## Score it on the smoke set

```bash
LLM_MODEL_PATH=artifacts/llm/qwen2.5-7b-instruct \
  uv run --group llm --extra gpu python -m vrp_service.eval.harness \
  --backend llm --smoke --warmup 1 --out results/llm_smoke.jsonl
```

Each JSONL row has `prompt_tokens` and `completion_tokens` beside the cost, gaps, feasibility,
and latency. `--limit N` runs the first N instances of every variant.

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

Not yet run on the 7B weights. The smoke run records load time, seconds per instance, output
tokens per second, and peak GPU memory here.
