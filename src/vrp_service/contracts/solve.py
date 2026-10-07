"""Stub request and response shapes from the API contract sketch.

Field names are fixed so later backends can share one JSONL schema.
Validation of variants and required instance attributes arrives with the solve path.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: int
    x: float
    y: float
    demand: float | None = None


class Instance(BaseModel):
    depot_ids: list[int]
    nodes: list[Node]
    capacity: float | None = None
    time_windows: list[list[float]] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class SolveOptions(BaseModel):
    timeout_ms: int = 30_000
    return_debug: bool = False


class SolveRequest(BaseModel):
    contract_version: str = "1.0.0"
    backend: str
    variant: str
    seed: int
    instance: Instance
    options: SolveOptions = Field(default_factory=SolveOptions)


class Route(BaseModel):
    vehicle_id: int
    node_ids: list[int]


class SolveResponse(BaseModel):
    contract_version: str = "1.0.0"
    backend: str
    model_version: str
    prompt_version: str | None = None
    feasible: bool
    objective_value: float | None = None
    objective_sense: Literal["min"] = "min"
    gap_to_baseline: float | None = None
    baseline_name: str | None = None
    routes: list[Route]
    latency_ms: float
    status: str
    error_type: str | None = None
    debug: dict[str, Any] = Field(default_factory=dict)
