# Paper notes: the LLM backend's scope

The `llm` backend is an engineering demo inspired by Jiang et al., *Large Language Models as
End-to-end Combinatorial Optimization Solvers* (NeurIPS 2025,
[arXiv:2509.16865](https://arxiv.org/abs/2509.16865)), called LLMCoSolver here. These notes say
what this project takes from the paper and what it leaves out, so the LLM results are read as
a demo, not a solver claim.

## What the paper does

A 7B LLM writes complete solutions to combinatorial optimization problems as text, with no
code generation and no call to an external solver. Training has two stages: supervised
fine-tuning (SFT) on solutions from domain-specific solvers, then feasibility-and-optimality-aware
reinforcement learning (FOARL), which the authors add because SFT alone gives over-greedy
solutions that break constraints. Across seven NP-hard problems they report a high feasible
rate and an average optimality gap of 1.03–8.20%.

## What this project takes

- One base model, Qwen2.5-7B-Instruct, and one QLoRA adapter (4-bit NF4 base, rank 64,
  alpha 64) trained on a mixture of the 16 single-depot variants. Inference runs in bf16, with
  the adapter merged for evaluation; quantized decoding was slower on the A100.
- The instance goes in as text; the routes come out as JSON.
- SFT on PyVRP solutions (20 s per instance), then one small FOARL pass.
- Every output goes through the same `check_routes` as the MTL policy.

## What this project leaves out

- The paper's training scale. The dataset here is sized for one L40S and is set in F5.
- The other combinatorial problems. Only the 16 single-depot VRP variants are covered; the
  multi-depot and mixed-backhaul variants are excluded.
- Any claim that the LLM beats PyVRP, OR-Tools, HGS, RouteFinder, or the MTL policy. The point
  is the change from the base model to the fine-tuned one.
- Solving instances with GPT or Claude. Their costs are estimated from token counts and list
  prices at the end.

## Repair policy

None. An output that does not parse, or a route that breaks a constraint, counts as
infeasible. Repairing it (dropping duplicate visits, inserting missing customers) would put a
heuristic's work into the model's score.

## Masked decoding is a separate baseline

The base model writes almost no feasible answer: on the full set, 0.1% zero-shot and 0% with
two examples, mostly because routes overload the vehicle or miss time windows (forcing every
customer to appear exactly once would have made at most 2 more answers feasible). So each
row also runs with feasibility-masked decoding, the LLM counterpart of the MTL policy's
action mask: at each token, only customers that keep the answer feasible can be written.
It constrains decoding; nothing is changed afterwards, so the no-repair rule holds. The
masked rows measure how good the model's choices are among feasible ones (cost and gaps);
the unmasked rows remain the measure of the model alone.

## How the LLM is measured

The same frozen protocol as `docs/results.md`: 1,600 instances (100 per variant, 100
customers), the same checker and harness, gaps to the stored PyVRP and OR-Tools routes. Rows:
zero-shot, few-shot (solved training examples in the prompt), QLoRA SFT (QLoRA SFT +
FOARL, each also with masked decoding, with extra columns: parse success, answers that visit
every customer exactly once, and an error breakdown. The smoke set runs first to measure speed; the full run is decided from that.

## Results

Full table: `docs/llm_runbook.md`, "Full eval set". Fine-tuning is the gain that matters:
with masked decoding, the gap to PyVRP falls from +241.5% (base, zero-shot) to +51.9%
(sft-v1, 500 instances per variant) and +24.1% (sft-v2, 5,000 per variant with the
nearest-customer hints). Unmasked, the fine-tuned model writes a feasible answer 0.4% of the
time, against 59% for the paper's CVRP-only SFT; the paper trains one problem on 500,000
instances, this project 16 variants on 80,000 with one adapter. FOARL was not run, and the
MTL policy (+2.9%) stays far ahead, as expected.
