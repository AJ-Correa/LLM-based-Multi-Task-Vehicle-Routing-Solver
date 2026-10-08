from vrp_service.contracts import SolveRequest, SolveResponse

REQUEST = {
    "contract_version": "1.0.0",
    "backend": "mtl",
    "variant": "cvrp",
    "seed": 0,
    "instance": {
        "depot_ids": [0],
        "nodes": [{"id": 1, "x": 0.1, "y": 0.2, "demand": 5}],
        "capacity": 50,
        "time_windows": None,
        "extra": {},
    },
    "options": {"timeout_ms": 30000, "return_debug": False},
}

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
    "routes": [{"vehicle_id": 0, "node_ids": [0, 1, 2, 0]}],
    "latency_ms": 42.5,
    "status": "ok",
    "error_type": None,
    "debug": {},
}


def test_contracts_roundtrip() -> None:
    request = SolveRequest.model_validate(REQUEST)
    assert SolveRequest.model_validate(request.model_dump()).model_dump() == request.model_dump()

    response = SolveResponse.model_validate(RESPONSE)
    assert SolveResponse.model_validate(response.model_dump()).model_dump() == response.model_dump()
