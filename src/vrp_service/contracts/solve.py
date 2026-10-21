"""Solve request and response shapes for the 16 training variants."""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vrp_service.contracts.variants import TRAINING_VARIANTS, VariantName


class Node(BaseModel):
    id: int
    x: float
    y: float
    demand_linehaul: float = Field(ge=0)
    demand_backhaul: float = Field(ge=0)


class TimeWindow(BaseModel):
    start: float
    end: float

    @model_validator(mode="after")
    def start_before_end(self) -> "TimeWindow":
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("a time window must be finite")
        if self.start > self.end:
            raise ValueError("a time window must start at or before it ends")
        return self


class Instance(BaseModel):
    nodes: list[Node] = Field(min_length=2)
    capacity: float = Field(gt=0)
    open_route: bool
    time_windows: list[TimeWindow] | None = None
    service_times: list[float] | None = None
    distance_limit: float | None = None
    speed: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def depot_is_first_node(self) -> "Instance":
        depot = self.nodes[0]
        if depot.id != 0:
            raise ValueError("node 0 is the depot")
        if depot.demand_linehaul != 0 or depot.demand_backhaul != 0:
            raise ValueError("the depot has no demand")
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node ids must be unique")
        if self.service_times is not None and any(value < 0 for value in self.service_times):
            raise ValueError("service times must be non-negative")
        return self


class SolveOptions(BaseModel):
    timeout_ms: int = 30_000
    return_debug: bool = False


class SolveRequest(BaseModel):
    contract_version: str = "1.0.0"
    backend: str
    variant: VariantName
    seed: int
    instance: Instance
    options: SolveOptions = Field(default_factory=SolveOptions)

    @field_validator("variant", mode="before")
    @classmethod
    def reject_unknown_variant(cls, value: object) -> object:
        if isinstance(value, str) and value not in TRAINING_VARIANTS:
            raise ValueError(f"variant must be one of the 16 training variants, got {value!r}")
        return value

    @model_validator(mode="after")
    def flags_match_variant(self) -> "SolveRequest":
        flags = TRAINING_VARIANTS[self.variant]
        instance = self.instance
        if instance.open_route != flags.open_route:
            raise ValueError("open_route does not match the variant")
        self._check_time_windows(flags.time_windows)
        self._check_distance_limit(flags.distance_limit)
        self._check_demands(flags.backhaul)
        return self

    def _check_time_windows(self, required: bool) -> None:
        windows = self.instance.time_windows
        service_times = self.instance.service_times
        if not required:
            if windows is not None or service_times is not None:
                raise ValueError("this variant has no time windows")
            return
        node_count = len(self.instance.nodes)
        if windows is None or len(windows) != node_count:
            raise ValueError("time windows are required for this variant")
        if service_times is None or len(service_times) != node_count:
            raise ValueError("service times are required for this variant")

    def _check_distance_limit(self, required: bool) -> None:
        limit = self.instance.distance_limit
        if not required:
            if limit is not None:
                raise ValueError("this variant has no distance limit")
            return
        if limit is None or not math.isfinite(limit) or limit <= 0:
            raise ValueError("a finite distance limit is required for this variant")

    def _check_demands(self, backhaul: bool) -> None:
        for node in self.instance.nodes[1:]:
            linehaul = node.demand_linehaul
            backhaul_demand = node.demand_backhaul
            if backhaul:
                if linehaul > 0 and backhaul_demand > 0:
                    raise ValueError("a customer cannot carry both linehaul and backhaul demand")
                if linehaul == 0 and backhaul_demand == 0:
                    raise ValueError("a customer needs linehaul or backhaul demand")
            else:
                if backhaul_demand != 0:
                    raise ValueError("this variant has no backhaul demand")
                if linehaul <= 0:
                    raise ValueError("a customer needs linehaul demand")


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
