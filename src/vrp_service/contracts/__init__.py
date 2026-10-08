"""Shared solve contracts for LLM-based VRP decision support."""

from vrp_service.contracts.solve import (
    Instance,
    Node,
    Route,
    SolveOptions,
    SolveRequest,
    SolveResponse,
)
from vrp_service.contracts.variants import TRAINING_VARIANTS, VariantFlags, VariantName

__all__ = [
    "Instance",
    "Node",
    "Route",
    "SolveOptions",
    "SolveRequest",
    "SolveResponse",
    "TRAINING_VARIANTS",
    "VariantFlags",
    "VariantName",
]
