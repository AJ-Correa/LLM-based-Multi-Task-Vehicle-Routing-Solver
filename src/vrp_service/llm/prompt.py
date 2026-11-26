"""Render a solve request as chat messages from the versioned templates in ``configs/prompts``.

Every number is an integer scaled by ``SCALE`` (3-decimal precision), because Qwen splits
numbers into single digits and a leading ``0.`` costs two tokens. Coordinates, times, and
the distance limit share one scale, so travel time still equals distance.
The model answers with customer ids only; the parser adds the depot.

Few-shot examples go before the question as earlier user and assistant turns, so the question
itself reads the same with or without them.
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path
from string import Template

from vrp_service.contracts import TRAINING_VARIANTS, SolveRequest
from vrp_service.serving.settings import project_root

PROMPT_VERSION = "v2"
# v2 adds each node's nearest customers, as LLMCoSolver's prompt does; v1 is kept for the
# results measured with it.
NEIGHBOURS = {"v1": 0, "v2": 2}
SCALE = 1000
# A solved example: the request and its routes as customer ids, without the depot.
Example = tuple[SolveRequest, list[list[int]]]


def render(
    request: SolveRequest, version: str = PROMPT_VERSION, examples: Sequence[Example] = ()
) -> list[dict[str, str]]:
    folder = project_root() / "configs/prompts" / version
    system = Template(_read(folder / "system.txt")).substitute(
        variant_upper=request.variant.upper(),
        constraints="\n".join(
            _read(folder / "constraints" / f"{name}.txt").strip() for name in _constraints(request)
        ),
    )
    user = Template(_read(folder / "user.txt"))
    messages = [{"role": "system", "content": system.strip()}]
    for example, routes in examples:
        messages += [
            {
                "role": "user",
                "content": user.substitute(instance=_instance(example, version)).strip(),
            },
            {"role": "assistant", "content": answer(routes)},
        ]
    question = user.substitute(instance=_instance(request, version)).strip()
    messages.append({"role": "user", "content": question})
    return messages


def answer(routes: list[list[int]]) -> str:
    """The answer exactly as constrained decoding writes it: compact JSON."""
    return json.dumps({"routes": routes}, separators=(",", ":"))


def _constraints(request: SolveRequest) -> list[str]:
    flags = TRAINING_VARIANTS[request.variant]
    closed = not flags.open_route
    names = ["visits", "open" if flags.open_route else "closed"]
    names.append("backhaul" if flags.backhaul else "capacity")
    if flags.time_windows:
        names += ["time_windows"] + (["time_windows_closed"] if closed else [])
    if flags.distance_limit:
        names += ["distance_limit"] + (["distance_limit_closed"] if closed else [])
    return names


def _instance(request: SolveRequest, version: str = PROMPT_VERSION) -> str:
    instance = request.instance
    flags = TRAINING_VARIANTS[request.variant]
    lines = [f"capacity: {SCALE}"]
    if flags.distance_limit:
        lines.append(f"distance limit: {_int(instance.distance_limit)}")
    near = NEIGHBOURS[version]
    columns = "id x y demand" + (" open close service" if flags.time_windows else "")
    columns += "".join(f" n{k} d{k}" for k in range(1, near + 1))
    lines.append(columns)
    for position, node in enumerate(instance.nodes):
        # Pickups are negative; demands are relative to capacity so the capacity is always SCALE.
        demand = (node.demand_linehaul - node.demand_backhaul) / instance.capacity
        row = [node.id, _int(node.x), _int(node.y), _int(demand)]
        if flags.time_windows:
            # Times in distance units, so "travel time equals distance" holds at any speed.
            window = instance.time_windows[position]
            times = [window.start, window.end, instance.service_times[position]]
            row += [_int(value * instance.speed) for value in times]
        if near:
            row += _nearest(instance.nodes, position, near)
        lines.append(" ".join(map(str, row)))
    return "\n".join(lines)


def _nearest(nodes, position: int, count: int) -> list[int]:
    """The ``count`` nearest customers to node ``position`` as id, distance pairs."""
    here = nodes[position]
    found = sorted(
        (math.dist((here.x, here.y), (node.x, node.y)), node.id)
        for other, node in enumerate(nodes)
        if other not in (0, position)
    )[:count]
    return [value for distance, node_id in found for value in (node_id, _int(distance))]


def _int(value: float) -> int:
    return round(value * SCALE)


def _read(path: Path) -> str:
    return path.read_text()
