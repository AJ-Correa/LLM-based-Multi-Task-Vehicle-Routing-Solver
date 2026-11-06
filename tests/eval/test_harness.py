import json

import pytest

from vrp_service.contracts import TRAINING_VARIANTS
from vrp_service.env import load_instance
from vrp_service.eval import load_instance_set, read_solution
from vrp_service.eval.harness import evaluate, main, markdown, route_length, select
from vrp_service.serving.settings import project_root
from vrp_service.serving.solve import solve_mtl

ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)


def _replay_ortools(request):
    """A stand-in policy that returns the stored OR-Tools tour.

    ``read_request`` sets ``seed`` to the instance index, which is how this finds it.
    """
    solution = read_solution(INSTANCE_SET, request.variant, "ortools", request.seed)

    def policy(_instance):
        return {"reward": [-solution.cost], "tours": [solution.actions]}

    return solve_mtl(request, policy)


def test_select_covers_every_variant_or_only_the_smoke_set() -> None:
    assert len(select(INSTANCE_SET)) == 16 * INSTANCE_SET.instances_per_variant
    assert select(INSTANCE_SET, limit=2)[:3] == [("cvrp", 0), ("cvrp", 1), ("ovrp", 0)]
    assert select(INSTANCE_SET, smoke=True) == INSTANCE_SET.smoke


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_a_limit_outside_the_set_is_rejected(limit: int) -> None:
    with pytest.raises(ValueError):
        select(INSTANCE_SET, limit=limit)
    with pytest.raises(ValueError):
        select(INSTANCE_SET, smoke=True, limit=1)


def test_rows_record_cost_feasibility_and_both_clocks() -> None:
    items = [(variant, 0) for variant in TRAINING_VARIANTS] + [("cvrp", 7)]
    rows = list(evaluate(INSTANCE_SET, items, _replay_ortools, warmup=1))
    assert [(row["variant"], row["index"]) for row in rows] == items
    for row in rows:
        stored = read_solution(INSTANCE_SET, row["variant"], "ortools", row["index"])
        assert row["status"] == "ok"
        assert row["feasible"] is True and row["reported_feasible"] is True
        assert row["violations"] == []
        assert row["objective_value"] == stored.cost
        assert row["vehicles"] == len(row["routes"]) > 0
        assert row["latency_ms"] >= 0 and row["wall_ms"] >= 0
        json.dumps(row)


def test_the_harness_checks_routes_instead_of_trusting_the_backend() -> None:
    def drop_last_route(request):
        response = _replay_ortools(request)
        return response.model_copy(update={"routes": response.routes[:-1]})

    [row] = evaluate(INSTANCE_SET, [("cvrp", 0)], drop_last_route)
    assert row["reported_feasible"] is True
    assert row["feasible"] is False
    assert row["violations"] == ["not_visited"]


def test_an_exception_becomes_an_error_row_and_the_run_goes_on() -> None:
    def flaky(request):
        if request.seed == 0:
            raise ConnectionError("down")
        return _replay_ortools(request)

    rows = list(evaluate(INSTANCE_SET, [("cvrp", 0), ("cvrp", 1)], flaky, warmup=2))
    assert [row["status"] for row in rows] == ["error", "ok"]
    assert rows[0]["error_type"] == "harness_ConnectionError"
    assert rows[0]["feasible"] is False and rows[0]["routes"] == []


def test_the_table_has_one_row_per_variant_and_an_overall_row() -> None:
    items = [("cvrp", 0), ("cvrp", 1), ("vrptw", 0), ("vrptw", 1)]
    rows = list(evaluate(INSTANCE_SET, items, _replay_ortools))
    rows[1]["feasible"] = False
    rows[3].update(status="error", feasible=False, latency_ms=1e9, wall_ms=1e9)
    table = markdown(rows, {"mode": "inprocess"})
    lines = table.splitlines()
    assert lines[0] == "- mode: inprocess"
    body = [line for line in lines if line.startswith("| ") and "---" not in line][1:]
    assert [line.split(" | ")[0] for line in body] == ["| cvrp", "| vrptw", "| all"]
    cvrp, vrptw, overall = (line.split(" | ") for line in body)
    assert cvrp[1:4] == ["2", "50.0%", f"{rows[0]['objective_value']:.4f}"]
    assert vrptw[1:4] == ["2", "50.0%", f"{rows[2]['objective_value']:.4f}"]
    assert overall[1:3] == ["4", "50.0%"]
    assert float(overall[7]) < 1e9, "an error row's latency must not enter p95"
    assert overall[-1] == "1 |"


def test_the_smoke_command_writes_jsonl_and_markdown(tmp_path, monkeypatch) -> None:
    # No checkpoint here, as in CI: every row is a well-formed model_unavailable error.
    monkeypatch.setenv("MTL_CHECKPOINT", str(tmp_path / "missing.pt"))
    out = tmp_path / "smoke.jsonl"
    with pytest.raises(SystemExit) as stopped:
        main(["--smoke", "--warmup", "0", "--out", str(out)])
    assert stopped.value.code != 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(row["variant"], row["index"]) for row in rows] == INSTANCE_SET.smoke
    assert {row["error_type"] for row in rows} == {"model_unavailable"}
    table = out.with_suffix(".md").read_text()
    assert "- instance set: instance_set_v1" in table
    assert "- policy load ms: not loaded" in table


def _replay(solver: str, scale: float = 1.0):
    def solve(request):
        stored = read_solution(INSTANCE_SET, request.variant, solver, request.seed)
        response = solve_mtl(request, lambda _i: {"reward": [-1.0], "tours": [stored.actions]})
        return response

    def scaled(request):
        response = solve(request)
        row_cost = route_length(load_instance(request), [r.node_ids for r in response.routes])
        return response.model_copy(update={"objective_value": row_cost * scale})

    return scaled


def test_replaying_a_reference_route_has_zero_gap_to_it() -> None:
    items = [(variant, 3) for variant in TRAINING_VARIANTS]
    for row in evaluate(INSTANCE_SET, items, _replay("ortools")):
        assert row["ortools_feasible"] is True
        assert row["gap_ortools"] == pytest.approx(0, abs=1e-12)
        assert row["gap_pyvrp"] is not None or row["pyvrp_feasible"] is False


def test_a_ten_percent_longer_cost_is_a_ten_percent_gap() -> None:
    [row] = evaluate(INSTANCE_SET, [("cvrp", 0)], _replay("pyvrp", 1.1))
    assert row["gap_pyvrp"] == pytest.approx(0.1)
    stored = read_solution(INSTANCE_SET, "cvrp", "pyvrp", 0)
    assert row["pyvrp_cost"] == pytest.approx(stored.cost, abs=0.01)


def test_an_infeasible_answer_has_no_gap() -> None:
    def drop_last_route(request):
        response = _replay("ortools")(request)
        return response.model_copy(update={"routes": response.routes[:-1]})

    [row] = evaluate(INSTANCE_SET, [("cvrp", 0)], drop_last_route)
    assert row["gap_pyvrp"] is None and row["gap_ortools"] is None


def _llm_row(
    variant: str, status: str, feasible: bool, error_type: str | None, violations=()
) -> dict:
    return {
        "violations": list(violations),
        "variant": variant,
        "status": status,
        "feasible": feasible,
        "error_type": error_type,
        "objective_value": 1.0 if feasible else None,
        "latency_ms": 10.0,
        "wall_ms": 10.0,
        "prompt_tokens": 2000,
        "completion_tokens": 300,
    }


def test_llm_rows_get_an_outcome_table_and_report_rebuilds_it(tmp_path, capsys) -> None:
    rows = [
        _llm_row("cvrp", "ok", True, None),
        _llm_row("cvrp", "ok", False, None, ["capacity_linehaul"]),
        _llm_row("cvrp", "ok", False, None, ["not_visited", "capacity_linehaul"]),
        _llm_row("vrptw", "error", False, "truncated"),
        _llm_row("vrptw", "error", False, "harness_KeyError"),
    ]
    table = markdown(rows, {})
    header, separator, *body = table.split("\n\n")[-1].strip().splitlines()
    assert header.count("|") == separator.count("|")
    overall = [cell.strip() for cell in body[-1].strip("|").split("|")]
    # all, n, parsed, each customer once, feasible, infeasible, truncated, invalid_json, schema,
    # failed, other, tokens
    assert overall == [
        "all",
        "5",
        "60.0%",
        "40.0%",
        "1",
        "2",
        "1",
        "0",
        "0",
        "0",
        "1",
        "2,000",
        "300",
    ]
    assert "Parsed" not in markdown([{**row, "prompt_tokens": None} for row in rows], {})

    out = tmp_path / "llm.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    main(["--report", str(out)])
    assert capsys.readouterr().out.strip().endswith(body[-1].strip())
