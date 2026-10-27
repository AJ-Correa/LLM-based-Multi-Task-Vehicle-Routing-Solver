import numpy as np
import pytest

from vrp_service.eval.metrics import percentile


def test_percentiles_of_one_to_ten() -> None:
    values = [float(value) for value in range(10, 0, -1)]
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 5.5
    assert percentile(values, 95) == pytest.approx(9.55)
    assert percentile(values, 100) == 10


@pytest.mark.parametrize("q", [50, 90, 95, 99])
def test_percentiles_match_numpy(q: float) -> None:
    values = np.random.default_rng(0).lognormal(size=257).tolist()
    assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))


def test_a_single_value_is_every_percentile() -> None:
    assert percentile([3.0], 95) == 3.0


def test_bad_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)
