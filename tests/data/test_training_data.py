import numpy as np
import pytest

from vrp_service.env import check_routes, load_instance, route_length
from vrp_service.eval.instance_set import read_request, read_solution
from vrp_service.serving.settings import project_root

pytest.importorskip("torch")
pyvrp = pytest.importorskip("pyvrp")

from vrp_service.data import generate, label  # noqa: E402
from vrp_service.eval.instance_set import load_instance_set  # noqa: E402

INSTANCE_SET = load_instance_set(
    project_root() / "configs/eval/instance_set_v1.yaml", project_root()
)


def test_generated_files_match_the_eval_format_and_are_reproducible() -> None:
    with np.load(INSTANCE_SET.path("vrpbltw", "100.npz")) as reference:
        expected = {key: reference[key].shape[1:] for key in reference.files}
    arrays = generate.generate("vrpbltw", 3, seed=7)
    for key, shape in expected.items():
        if key != "num_depots":
            assert arrays[key].shape[1:] == shape, key
    again = generate.generate("vrpbltw", 3, seed=7)
    assert np.array_equal(arrays["locs"], again["locs"])
    assert not generate.fingerprints(arrays["locs"]) & generate.eval_fingerprints()


@pytest.mark.parametrize("variant", ["cvrp", "ovrpbltw", "vrpbl"])
def test_pyvrp_labels_are_feasible_and_near_the_stored_reference(variant: str) -> None:
    instance = load_instance(read_request(INSTANCE_SET, variant, 0))
    routes = label.solve(instance, seconds=1.0)
    assert check_routes(instance, routes) == []
    reference = read_solution(INSTANCE_SET, variant, "pyvrp", 0).cost
    assert route_length(instance, routes) < reference * 1.05


def test_generate_then_label_writes_a_readable_solution_file(tmp_path) -> None:
    generate.main(
        ["--per-variant", "2", "--seed", "3", "--out", str(tmp_path), "--variants", "ovrptw"]
    )
    label.main(
        ["--root", str(tmp_path), "--variant", "ovrptw", "--seconds", "0.5", "--workers", "2"]
    )
    with np.load(tmp_path / "ovrptw" / "100_sol_pyvrp.npz") as solved:
        assert solved["feasible"].all()
        assert solved["actions"].shape[0] == 2 and (solved["costs"] < 0).all()


def test_few_shot_examples_come_from_labelled_training_data(tmp_path) -> None:
    from vrp_service.contracts import TRAINING_VARIANTS
    from vrp_service.llm.examples import load_examples
    from vrp_service.llm.parse import parse_routes
    from vrp_service.llm.prompt import render

    generate.main(["--per-variant", "2", "--seed", "5", "--out", str(tmp_path)])
    for variant in TRAINING_VARIANTS:
        label.main(["--root", str(tmp_path), "--variant", variant, "--seconds", "0.2"])
    examples = load_examples(tmp_path, 2)
    assert set(examples) == set(TRAINING_VARIANTS)

    request = read_request(INSTANCE_SET, "ovrpbltw", 0, "llm")
    messages = render(request, examples=examples["ovrpbltw"])
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[-1] == render(request)[-1]
    for (example, _), reply in zip(examples["ovrpbltw"], messages[2:5:2]):
        assert " " not in reply["content"]
        assert parse_routes(reply["content"], example).violations == []
    with pytest.raises(ValueError, match="feasible"):
        load_examples(tmp_path, 3)


def test_sft_rows_hold_out_the_last_instances_of_every_variant(tmp_path) -> None:
    from vrp_service.contracts import TRAINING_VARIANTS
    from vrp_service.llm.prompt import render
    from vrp_service.train.sft import rows

    generate.main(["--per-variant", "3", "--seed", "8", "--out", str(tmp_path)])
    for variant in TRAINING_VARIANTS:
        label.main(["--root", str(tmp_path), "--variant", variant, "--seconds", "0.2"])
    train, held_out = rows(str(tmp_path), holdout=1)
    assert len(train) == 2 * len(TRAINING_VARIANTS) and len(held_out) == len(TRAINING_VARIANTS)
    assert not {str(row["prompt"]) for row in train} & {str(row["prompt"]) for row in held_out}
    request = read_request(INSTANCE_SET, "cvrp", 0, "llm")
    assert [m["role"] for m in train[0]["prompt"]] == [m["role"] for m in render(request)]
    assert len(rows(str(tmp_path), holdout=1, limit=1)[0]) == len(TRAINING_VARIANTS)
