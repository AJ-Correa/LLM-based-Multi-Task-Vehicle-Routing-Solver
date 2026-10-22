import hashlib

import numpy as np
import pytest

from vrp_service.contracts import TRAINING_VARIANTS
from vrp_service.env import load_instance
from vrp_service.eval import load_instance_set, read_request, read_solution
from vrp_service.mtl.decode import batch_fields
from vrp_service.serving.settings import project_root

ROOT = project_root()
INSTANCE_SET = load_instance_set(ROOT / "configs/eval/instance_set_v1.yaml", ROOT)
VARIANTS = sorted(TRAINING_VARIANTS)


def test_every_variant_has_an_instance_file_and_both_solutions() -> None:
    expected = {INSTANCE_SET.instance_file, *INSTANCE_SET.solution_files.values()}
    assert sorted(INSTANCE_SET.sha256) == VARIANTS
    for files in INSTANCE_SET.sha256.values():
        assert set(files) == expected


@pytest.mark.parametrize("variant", VARIANTS)
def test_files_match_their_checksums(variant: str) -> None:
    for name, digest in INSTANCE_SET.sha256[variant].items():
        data = INSTANCE_SET.path(variant, name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name


@pytest.mark.parametrize("variant", VARIANTS)
def test_files_hold_the_expected_number_of_instances(variant: str) -> None:
    count = INSTANCE_SET.instances_per_variant
    with np.load(INSTANCE_SET.path(variant, INSTANCE_SET.instance_file)) as data:
        assert data["locs"].shape == (count, 101, 2)
    for name in INSTANCE_SET.solution_files.values():
        with np.load(INSTANCE_SET.path(variant, name)) as data:
            assert data["costs"].shape == (count,)
            assert len(data["actions"]) == count


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_request_rebuilds_the_stored_arrays(variant: str) -> None:
    index = INSTANCE_SET.instances_per_variant - 1
    fields = batch_fields(load_instance(read_request(INSTANCE_SET, variant, index)))
    with np.load(INSTANCE_SET.path(variant, INSTANCE_SET.instance_file)) as data:
        np.testing.assert_allclose(fields["locs"][0], data["locs"][index], rtol=1e-6)
        np.testing.assert_allclose(
            fields["demand_linehaul"][0][1:], data["demand_linehaul"][index], rtol=1e-6
        )
        if "demand_backhaul" in data.files:
            np.testing.assert_allclose(
                fields["demand_backhaul"][0][1:], data["demand_backhaul"][index], rtol=1e-6
            )
        if "time_windows" in data.files:
            np.testing.assert_allclose(
                fields["time_windows"][0], data["time_windows"][index], rtol=1e-6
            )
            np.testing.assert_allclose(
                fields["service_time"][0], data["service_time"][index], rtol=1e-6
            )
        if "distance_limit" in data.files:
            assert fields["distance_limit"][0][0] == pytest.approx(
                float(data["distance_limit"][index][0]), rel=1e-6
            )
        if "open_route" in data.files:
            assert fields["open_route"][0][0] == bool(data["open_route"][index][0])


def test_stored_costs_are_positive_lengths() -> None:
    solution = read_solution(INSTANCE_SET, "cvrp", "pyvrp", 0)
    assert solution.cost > 0
    assert any(solution.actions)


def test_smoke_instances_are_in_the_set() -> None:
    assert len(INSTANCE_SET.smoke) == 2
    for variant, index in INSTANCE_SET.smoke:
        assert variant in INSTANCE_SET.sha256
        assert 0 <= index < INSTANCE_SET.instances_per_variant
