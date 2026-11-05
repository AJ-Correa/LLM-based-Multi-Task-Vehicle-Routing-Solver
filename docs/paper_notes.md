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

## How the LLM is measured

The same frozen protocol as `docs/results.md`: 1,600 instances (100 per variant, 100
customers), the same checker and harness, gaps to the stored PyVRP and OR-Tools routes. Four
rows, zero-shot, few-shot (solved training examples in the prompt), QLoRA SFT, and QLoRA SFT +
FOARL, with two extra columns: parse success and an
error breakdown. The smoke set runs first to measure speed; the full run is decided from that.
