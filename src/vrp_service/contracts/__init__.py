"""Shared solve contracts for LLM-based VRP decision support."""

from vrp_service.contracts.solve import (
    Instance,
    Route,
    SolveOptions,
    SolveRequest,
    SolveResponse,
)

__all__ = [
    "Instance",
    "Route",
    "SolveOptions",
    "SolveRequest",
    "SolveResponse",
]
