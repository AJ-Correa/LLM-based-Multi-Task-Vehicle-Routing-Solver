import copy

import pytest
from pydantic import ValidationError

from vrp_service.contracts import TRAINING_VARIANTS, SolveRequest, SolveResponse

RESPONSE = {
    "contract_version": "1.0.0",
    "backend": "mtl",
    "model_version": "mtl/v0.1.0",
    "prompt_version": None,
    "feasible": True,
    "objective_value": 123.4,
    "objective_sense": "min",
    "gap_to_baseline": 0.08,
    "baseline_name": "pyvrp_tlim",
    "routes": [{"vehicle_id": 0, "node_ids": [0, 1, 0]}],
    "latency_ms": 42.5,
    "status": "ok",
    "error_type": None,
    "debug": {},
}


def _request(variant: str) -> dict:
    flags = TRAINING_VARIANTS[variant]
    customer_linehaul = 0.0 if flags.backhaul else 1.0
    customer_backhaul = 1.0 if flags.backhaul else 0.0
    return {
        "contract_version": "1.0.0",
        "backend": "mtl",
        "variant": variant,
        "seed": 0,
        "instance": {
            "nodes": [
                {"id": 0, "x": 0.0, "y": 0.0, "demand_linehaul": 0, "demand_backhaul": 0},
                {
                    "id": 1,
                    "x": 0.2,
                    "y": 0.3,
                    "demand_linehaul": customer_linehaul,
                    "demand_backhaul": customer_backhaul,
                },
            ],
            "capacity": 10,
            "open_route": flags.open_route,
            "time_windows": [{"start": 0, "end": 10}, {"start": 1, "end": 5}]
            if flags.time_windows
            else None,
            "service_times": [0, 0.1] if flags.time_windows else None,
            "distance_limit": 3.0 if flags.distance_limit else None,
            "speed": 1.0,
        },
        "options": {"timeout_ms": 30000, "return_debug": False},
    }


def test_contracts_roundtrip() -> None:
    request = SolveRequest.model_validate(_request("cvrp"))
    assert SolveRequest.model_validate(request.model_dump()).model_dump() == request.model_dump()

    response = SolveResponse.model_validate(RESPONSE)
    assert SolveResponse.model_validate(response.model_dump()).model_dump() == response.model_dump()


def test_all_training_variants_are_accepted() -> None:
    assert len(TRAINING_VARIANTS) == 16
    for name in TRAINING_VARIANTS:
        parsed = SolveRequest.model_validate(_request(name))
        assert parsed.variant == name


def test_missing_capacity_is_rejected() -> None:
    payload = _request("cvrp")
    del payload["instance"]["capacity"]
    with pytest.raises(ValidationError):
        SolveRequest.model_validate(payload)


def test_unknown_variant_is_rejected() -> None:
    for name in ("mdcvrp", "vrpmb"):
        payload = _request("cvrp")
        payload["variant"] = name
        with pytest.raises(ValidationError):
            SolveRequest.model_validate(payload)


def test_backhaul_demand_is_rejected_on_cvrp() -> None:
    payload = _request("cvrp")
    payload["instance"]["nodes"][1]["demand_backhaul"] = 1
    with pytest.raises(ValidationError):
        SolveRequest.model_validate(payload)


def test_vrptw_requires_time_windows() -> None:
    payload = _request("vrptw")
    payload["instance"]["time_windows"] = None
    payload["instance"]["service_times"] = None
    with pytest.raises(ValidationError):
        SolveRequest.model_validate(payload)


def test_roundtrip_keeps_a_fully_constrained_variant() -> None:
    original = _request("ovrpbltw")
    parsed = SolveRequest.model_validate(original)
    again = SolveRequest.model_validate(parsed.model_dump())
    assert again.model_dump() == parsed.model_dump()
    assert copy.deepcopy(original)["variant"] == "ovrpbltw"
