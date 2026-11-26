import math

import pytest

from vrp_service.contracts import TRAINING_VARIANTS
from vrp_service.eval import load_instance_set, read_request
from vrp_service.llm import PROMPT_VERSION, render
from vrp_service.llm.prompt import SCALE
from vrp_service.serving.settings import project_root

ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)
SNAPSHOTS = ROOT / "tests/llm/snapshots"


def _text(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"### {m['role']}\n{m['content']}" for m in messages) + "\n"


def _rows(user: str) -> list[list[int]]:
    lines = user.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("id x y"))
    return [[int(value) for value in line.split()] for line in lines[start + 1 :]]


@pytest.mark.parametrize("version", ["v1", PROMPT_VERSION])
@pytest.mark.parametrize("variant", sorted(TRAINING_VARIANTS))
def test_render_matches_the_snapshot(variant: str, version: str) -> None:
    # Regenerate on purpose with a new prompt version, never to make this pass.
    text = _text(render(read_request(INSTANCE_SET, variant, 0, backend="llm"), version))
    assert text == (SNAPSHOTS / f"{version}_{variant}_0.txt").read_text()


def test_v2_lists_the_two_nearest_customers_with_distances() -> None:
    request = read_request(INSTANCE_SET, "cvrp", 0)
    nodes = request.instance.nodes
    rows = _rows(render(request, "v2")[1]["content"])
    for position in (0, 5):
        here = nodes[position]
        distance = {n.id: math.dist((here.x, here.y), (n.x, n.y)) for n in nodes[1:]}
        distance.pop(here.id, None)
        nearest = sorted(distance, key=distance.get)[:2]
        assert rows[position][4:] == [v for n in nearest for v in (n, round(distance[n] * SCALE))]


def test_cvrp_prompt_has_only_the_base_constraints() -> None:
    system, user = (m["content"] for m in render(read_request(INSTANCE_SET, "cvrp", 0)))
    assert "return to it" in system
    for text in ("pickup", "close time", "distance limit", "do not return"):
        assert text not in system
    assert "open close service" not in user and "distance limit" not in user


def test_ovrpbltw_prompt_has_every_feature_and_no_return() -> None:
    system, user = (m["content"] for m in render(read_request(INSTANCE_SET, "ovrpbltw", 0)))
    for text in ("do not return", "pickup", "close time", "distance limit"):
        assert text in system
    assert "return to the depot" not in system and "Return to the depot" not in system
    assert "open close service" in user and "distance limit:" in user


@pytest.mark.parametrize("variant", ["vrpbltw", "ovrpltw", "cvrp"])
def test_rows_round_trip_to_the_instance_at_three_decimals(variant: str) -> None:
    request = read_request(INSTANCE_SET, variant, 3)
    instance = request.instance
    rows = _rows(render(request)[1]["content"])
    assert [row[0] for row in rows] == [node.id for node in instance.nodes]
    half_step = 0.5 / SCALE + 1e-9
    for position, (node, row) in enumerate(zip(instance.nodes, rows)):
        assert abs(row[1] / SCALE - node.x) <= half_step
        assert abs(row[2] / SCALE - node.y) <= half_step
        # Eval demands are multiples of capacity/50, so the integer demand is exact.
        demand = (node.demand_linehaul - node.demand_backhaul) / instance.capacity
        assert abs(row[3] / SCALE - demand) < 1e-6
        if instance.time_windows:
            window = instance.time_windows[position]
            assert abs(row[4] / SCALE - window.start) <= half_step
            assert abs(row[5] / SCALE - window.end) <= half_step
            assert abs(row[6] / SCALE - instance.service_times[position]) <= half_step


def test_rendering_is_deterministic() -> None:
    request = read_request(INSTANCE_SET, "vrpltw", 7)
    assert render(request) == render(request)


def test_times_are_written_in_distance_units() -> None:
    request = read_request(INSTANCE_SET, "vrptw", 0)
    request.instance.speed = 2.0
    window = request.instance.time_windows[1]
    row = _rows(render(request)[1]["content"])[1]
    assert row[4:7] == [
        round(v * 2 * SCALE) for v in (window.start, window.end, request.instance.service_times[1])
    ]
