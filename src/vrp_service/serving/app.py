"""HTTP application for LLM-based VRP decision support.

When the checkpoint is present, startup loads the policy once so no request pays
for the load. Each solve writes one JSON line to stdout.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Body, FastAPI
from pydantic import BaseModel

from vrp_service import __version__
from vrp_service.contracts import SolveRequest, SolveResponse
from vrp_service.mtl.decode import load_policy
from vrp_service.serving.settings import ServingSettings
from vrp_service.serving.solve import CVRP_EXAMPLE, solve_mtl

app_title = "LLM-based VRP decision support"
solve_log = logging.getLogger("vrp_service.solve")
STDOUT_HANDLER = "vrp_service.solve.stdout"


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    ready: bool


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else ServingSettings.from_env()
    _log_to_stdout()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        _policy(application, settings)
        yield

    application = FastAPI(title=app_title, version=__version__, lifespan=lifespan)
    application.state.settings = settings
    application.state.policy = None

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @application.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        return ReadyResponse(ready=settings.weights_ready())

    @application.post("/solve", response_model=SolveResponse)
    def solve(
        request: Annotated[
            SolveRequest,
            Body(openapi_examples={"cvrp": {"summary": "One customer", "value": CVRP_EXAMPLE}}),
        ],
    ) -> SolveResponse:
        response = solve_mtl(request, _policy(application, settings), settings.model_version)
        _log_solve(request, response)
        return response

    return application


def _log_to_stdout() -> None:
    if any(handler.get_name() == STDOUT_HANDLER for handler in solve_log.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(STDOUT_HANDLER)
    handler.setFormatter(logging.Formatter("%(message)s"))
    solve_log.addHandler(handler)
    solve_log.setLevel(logging.INFO)
    solve_log.propagate = False


def _log_solve(request: SolveRequest, response: SolveResponse) -> None:
    record = {
        "event": "solve",
        "backend": request.backend,
        "variant": request.variant,
        "customers": len(request.instance.nodes) - 1,
        "model_version": response.model_version,
        "status": response.status,
        "error_type": response.error_type,
        "feasible": response.feasible,
        "objective_value": response.objective_value,
        "vehicles": len(response.routes),
        "latency_ms": round(response.latency_ms, 3),
    }
    solve_log.info(json.dumps(record))


def _policy(application: FastAPI, settings: ServingSettings):
    if application.state.policy is not None:
        return application.state.policy
    if not settings.weights_ready() or settings.source_root is None:
        return None
    application.state.policy = load_policy(
        settings.checkpoint_path,
        settings.source_root,
        settings.device,
        settings.cpu_threads,
    )
    return application.state.policy


app = create_app()
