# Eval set, n=100

16 single-depot variants, 100 instances each, 100 customers plus one depot. These are the first
100 instances of the 1,000-instance test sets the multi-task policy was evaluated on. No values
were changed; every array keeps its key, dtype, and order.

Each `<variant>/` folder holds:

| File | Keys |
| --- | --- |
| `100.npz` | `locs` (100, 101, 2), `demand_linehaul` (100, 100), `vehicle_capacity`, `speed`, `num_depots`, plus the variant's own fields: `demand_backhaul`, `backhaul_class`, `open_route`, `time_windows` (100, 101, 2), `service_time` (100, 101), `distance_limit` |
| `100_sol_pyvrp.npz` | `actions`, `costs` (100,), `time` |
| `100_sol_ortools.npz` | same keys as PyVRP |

Notes:

- Demands are already divided by vehicle capacity, so `vehicle_capacity` is 1.
- `costs` are rewards, the negative of route length. The cost is `-costs`.
- PyVRP `costs` are the solver's objective rounded to 3 decimals (it worked on rounded
  distances), so they differ from the exact route length by up to about 0.009. OR-Tools costs
  match the route length to within 3e-4.
- `actions` is the visit order with `0` for the depot, padded with zeros.
- Each instance was solved with a 20-second limit by PyVRP and by OR-Tools.
- `time` is a single number carried over unchanged. It is the wall-clock time of the original
  run over all 1,000 instances solved in parallel, not a per-instance time.
