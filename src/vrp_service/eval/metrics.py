"""Latency percentiles for the eval tables."""

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile with linear interpolation between ranks (numpy's default)."""
    if not values:
        raise ValueError("no values")
    if not 0 <= q <= 100:
        raise ValueError("q must be between 0 and 100")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
