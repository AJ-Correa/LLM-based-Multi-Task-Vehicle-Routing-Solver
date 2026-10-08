"""The 16 single-depot variants POLAR trains on.

Each name is one on/off combination of open routes, time windows, a distance
limit, and classical backhaul. Mixed backhaul and multi-depot names are absent.
"""

from typing import Literal, NamedTuple


class VariantFlags(NamedTuple):
    open_route: bool
    time_windows: bool
    distance_limit: bool
    backhaul: bool


TRAINING_VARIANTS: dict[str, VariantFlags] = {
    "cvrp": VariantFlags(False, False, False, False),
    "ovrp": VariantFlags(True, False, False, False),
    "vrpb": VariantFlags(False, False, False, True),
    "vrpl": VariantFlags(False, False, True, False),
    "vrptw": VariantFlags(False, True, False, False),
    "ovrptw": VariantFlags(True, True, False, False),
    "ovrpb": VariantFlags(True, False, False, True),
    "ovrpl": VariantFlags(True, False, True, False),
    "vrpbl": VariantFlags(False, False, True, True),
    "vrpbtw": VariantFlags(False, True, False, True),
    "vrpltw": VariantFlags(False, True, True, False),
    "ovrpbl": VariantFlags(True, False, True, True),
    "ovrpbtw": VariantFlags(True, True, False, True),
    "ovrpltw": VariantFlags(True, True, True, False),
    "vrpbltw": VariantFlags(False, True, True, True),
    "ovrpbltw": VariantFlags(True, True, True, True),
}

VariantName = Literal[
    "cvrp",
    "ovrp",
    "vrpb",
    "vrpl",
    "vrptw",
    "ovrptw",
    "ovrpb",
    "ovrpl",
    "vrpbl",
    "vrpbtw",
    "vrpltw",
    "ovrpbl",
    "ovrpbtw",
    "ovrpltw",
    "vrpbltw",
    "ovrpbltw",
]
