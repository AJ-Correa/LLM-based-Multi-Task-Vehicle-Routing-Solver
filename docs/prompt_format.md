# Prompt format and token cost

The `llm` backend renders each solve request as two chat messages from
`configs/prompts/<version>/` (`src/vrp_service/llm/prompt.py`). This page records why the
instance is written as integers and what that saves.

## Format (prompt `v1`)

The system message names the variant and lists only the rules the variant has, worded to
match the feasibility checker. The user message is the instance as a table of integers:

```text
capacity: 1000
distance limit: 2492
id x y demand open close service
0 192 622 0 0 4600 0
1 438 785 -100 2468 2666 150
2 780 273 160 1946 2141 159
```

- Coordinates, times, service times, and the distance limit are multiplied by 1000 and
  rounded. Times are also multiplied by the vehicle speed (1 on the eval set), so they are in
  distance units and travel time equals distance.
- Demand is `(linehaul − backhaul) / capacity × 1000`, so the capacity is always 1000 and a
  pickup is negative. On the eval set every demand is a multiple of capacity/50, so this is
  exact.
- Columns that the variant does not use (time windows, distance limit) are left out.
- The model answers `{"routes": [[3, 17, 42], [8, 1]]}`: customer ids only; the parser
  adds the depot.

Every rendered variant is in `tests/llm/snapshots/` and is checked on each test run.

## Why integers

Qwen2.5's tokenizer splits numbers into single digits: `0.438` is five tokens
(`0 . 4 3 8`), and `438` is three. The earlier draft wrote decimals with labels
(`1 x=0.438 y=0.785 bh=0.100 tw=[2.468,2.666] s=0.150`). The integer table carries the same
values at the same 3-decimal precision without the `0.` prefix or the labels.

## Measured saving

Tokenizer: `Qwen/Qwen2.5-7B-Instruct` at revision `a09a354`, `tokenizers` library. Count:
system + user message content, without the chat template's fixed tokens (the same for
both formats). All 1,600 eval instances (100 per variant, 100 customers). Measured on
2026-10-28.

| Variant | Labeled decimals | Integer table | Saved |
| --- | ---: | ---: | ---: |
| cvrp | 2,580 | 1,584 | 38.6% |
| ovrp | 2,584 | 1,587 | 38.6% |
| vrpb | 2,624 | 1,610 | 38.6% |
| vrpl | 2,615 | 1,616 | 38.2% |
| vrptw | 4,775 | 3,030 | 36.5% |
| ovrptw | 4,761 | 3,021 | 36.6% |
| ovrpb | 2,628 | 1,613 | 38.6% |
| ovrpl | 2,607 | 1,609 | 38.3% |
| vrpbl | 2,660 | 1,643 | 38.2% |
| vrpbtw | 4,820 | 3,057 | 36.6% |
| vrpltw | 4,810 | 3,062 | 36.3% |
| ovrpbl | 2,652 | 1,636 | 38.3% |
| ovrpbtw | 4,806 | 3,048 | 36.6% |
| ovrpltw | 4,784 | 3,043 | 36.4% |
| vrpbltw | 4,855 | 3,089 | 36.4% |
| ovrpbltw | 4,829 | 3,070 | 36.4% |
| **All 1,600** | **3,712** | **2,333** | **37.2%** |

Mean input tokens per prompt. Over the whole eval set: 5,939,000 → 3,732,060 tokens
(2.2 M fewer per full evaluation pass). The same saving applies to every SFT and FOARL
training sequence, since they use this format.

What did not change: the information in the prompt, the rules, and the output format. The
model reads different text, so its generations are not token-for-token the same as they
would be with the labeled draft; no generation was run with that draft.
