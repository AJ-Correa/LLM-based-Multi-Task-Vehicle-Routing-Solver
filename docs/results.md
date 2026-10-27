# Results

## MTL policy on the frozen eval set

Measured 2026-10-27 with `mtl/v0.1.0` on `instance_set_v1`: 16 single-depot variants × 100
instances, 100 customers each, 1,600 solves per run.

### Setup

- GPU: NVIDIA L40S (46 GB). CPU runs use 4 torch threads (`MTL_CPU_THREADS=4`).
- In-process harness, 5 warm-up solves before timing. Policy load (cold start) is reported once.
- References: PyVRP and OR-Tools, 20 s per instance. Their costs are recomputed from the stored
  routes in float64 (the stored PyVRP costs are rounded to 3 decimals). Every reference route
  passes the feasibility checker at the solver rounding tolerance (1e-2); the model's routes are
  checked at 1e-5.

```bash
MTL_DEVICE=cuda uv run --group mtl --extra gpu python -m vrp_service.eval.harness --out results/mtl_gpu.jsonl
MTL_DEVICE=cpu  uv run --group mtl --extra gpu python -m vrp_service.eval.harness --out results/mtl_cpu.jsonl
```

### Definitions

- **Feasible:** the harness's own route check passes (capacity, each customer once, depot start
  and end, time windows, distance limit, linehaul before backhaul). The backend's claim is stored
  separately and not trusted.
- **Gap:** `(cost − reference) / reference` per instance, averaged over instances where both
  solutions are feasible. Negative means the policy found a shorter plan.
- **p50 / p95 ms:** percentiles (linear interpolation) of `latency_ms`, the clock inside the
  solve call: decode, route split, and the feasibility check. **Wall** is the harness's clock
  around the same call; in process the two match, over HTTP wall adds the round trip (~3 ms
  locally). Only solved rows count.
- **Cold start:** loading the checkpoint and building the network, once per process.

### GPU (L40S), per variant

Policy load: 3.5 s.

| Variant | n | Feasible | Mean cost (feasible) | Gap PyVRP | Gap OR-Tools | p50 ms | p95 ms | Wall p50 ms | Wall p95 ms | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cvrp | 100 | 100.0% | 16.0564 | +1.22% | -3.13% | 521.2 | 549.1 | 521.2 | 549.1 | 0 |
| ovrp | 100 | 100.0% | 10.1904 | +3.62% | +0.80% | 524.0 | 548.5 | 524.1 | 548.6 | 0 |
| vrpb | 100 | 100.0% | 15.0991 | +3.98% | -0.00% | 524.1 | 554.4 | 524.1 | 554.5 | 0 |
| vrpl | 100 | 100.0% | 16.2205 | +1.62% | -2.54% | 520.0 | 538.6 | 520.0 | 538.7 | 0 |
| vrptw | 100 | 100.0% | 26.5220 | +2.83% | +1.17% | 556.0 | 582.0 | 556.0 | 582.0 | 0 |
| ovrptw | 100 | 100.0% | 17.4758 | +2.00% | +1.44% | 574.6 | 618.2 | 574.6 | 618.2 | 0 |
| ovrpb | 100 | 100.0% | 10.8504 | +3.73% | +1.05% | 549.5 | 574.5 | 549.5 | 574.5 | 0 |
| ovrpl | 100 | 100.0% | 10.1914 | +3.63% | +0.69% | 534.4 | 549.4 | 534.4 | 549.5 | 0 |
| vrpbl | 100 | 100.0% | 15.6831 | +5.18% | +0.17% | 524.5 | 556.5 | 524.5 | 556.5 | 0 |
| vrpbtw | 100 | 100.0% | 30.8185 | +2.33% | +0.68% | 567.0 | 596.5 | 567.1 | 596.5 | 0 |
| vrpltw | 100 | 100.0% | 26.9775 | +3.39% | +1.21% | 553.5 | 585.0 | 553.6 | 585.0 | 0 |
| ovrpbl | 100 | 100.0% | 10.8642 | +3.86% | +1.32% | 546.9 | 573.5 | 546.9 | 573.5 | 0 |
| ovrpbtw | 100 | 100.0% | 19.8399 | +1.82% | +1.16% | 598.5 | 631.5 | 598.6 | 631.6 | 0 |
| ovrpltw | 100 | 100.0% | 17.4741 | +1.99% | +1.41% | 576.0 | 610.4 | 576.0 | 610.4 | 0 |
| vrpbltw | 100 | 100.0% | 31.2237 | +2.69% | +0.79% | 574.5 | 599.4 | 574.5 | 599.4 | 0 |
| ovrpbltw | 100 | 100.0% | 19.8312 | +1.77% | +1.13% | 598.3 | 630.5 | 598.4 | 630.5 | 0 |
| all | 1600 | 100.0% | 18.4574 | +2.85% | +0.46% | 552.2 | 610.4 | 552.2 | 610.5 | 0 |

### CPU (4 threads), all variants

Policy load: 2.5 s.

| Variant | n | Feasible | Mean cost (feasible) | Gap PyVRP | Gap OR-Tools | p50 ms | p95 ms | Wall p50 ms | Wall p95 ms | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1600 | 100.0% | 18.4574 | +2.85% | +0.46% | 596.9 | 648.5 | 596.9 | 648.5 | 0 |

### Reading the numbers

- Every one of the 1,600 plans is feasible, with no errors.
- The policy is 2.9% above PyVRP on average and 0.5% above OR-Tools, in about half a second per
  instance, against 20 s per instance for both solvers. It beats PyVRP on 8 of 1,600 instances.
- A solve decodes 100 tours in parallel (one per starting customer, about 110 steps each) and
  keeps the best. The step-by-step decode dominates the time, so the GPU is only about 8% faster
  than 4 CPU threads at batch size 1. Repeated GPU runs varied between about 440 and 550 ms p50.
- Each L variant uses the same instances as its partner without the limit (for example `ovrp`
  and `ovrpl`); only the distance limit is added. On closed routes the limit binds (the longest
  PyVRP routes reach 100% of it). On open routes it barely does: the longest PyVRP route reaches
  72% of the limit in `ovrpl`, 98% in `ovrpbl`, and 96% in `ovrpltw`. So `ovrpl`, `ovrpbl`, and
  `ovrpltw` score almost the same as `ovrp`, `ovrpb`, and `ovrptw`, and add little information.
