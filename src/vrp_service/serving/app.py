"""HTTP application for LLM-based VRP decision support.

When the checkpoint is present, startup loads the policy once so no request pays
for the load.
"""

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


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    ready: bool


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else ServingSettings.from_env()

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
        return solve_mtl(request, _policy(application, settings), settings.model_version)

    return application


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
