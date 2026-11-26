"""HTTP application for LLM-based VRP decision support.

When the checkpoint is present, startup loads the policy once so no request pays
for the load; with ``LLM_MODEL_PATH`` set it also loads the LLM. Each solve writes
one JSON line to stdout.
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Body, FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from vrp_service import __version__
from vrp_service.contracts import SolveRequest, SolveResponse
from vrp_service.llm.generate import generate as generate_answer
from vrp_service.mtl.decode import load_policy
from vrp_service.observability.metrics import MODEL_INFO, POLICY_LOADED, record_solve
from vrp_service.observability.tracing import configure_tracing
from vrp_service.registry import ModelEntry, Registry, load_registry
from vrp_service.serving.settings import ServingSettings
from vrp_service.serving.solve import CVRP_EXAMPLE, solve_llm, solve_mtl

app_title = "LLM-based VRP decision support"
solve_log = logging.getLogger("vrp_service.solve")
STDOUT_HANDLER = "vrp_service.solve.stdout"


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    ready: bool


class ModelsResponse(BaseModel):
    active: str
    active_registered: bool
    models: list[ModelEntry]
    errors: list[str]


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings if settings is not None else ServingSettings.from_env()
    _log_to_stdout()
    configure_tracing()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        _policy(application, settings)
        _llm(application, settings)
        yield

    application = FastAPI(title=app_title, version=__version__, lifespan=lifespan)
    application.state.settings = settings
    application.state.policy = None
    application.state.llm = None
    application.state.llm_failed = False
    application.state.registry = None
    MODEL_INFO.clear()
    MODEL_INFO.labels(settings.model_version).set(1)
    if settings.llm_model_path:
        MODEL_INFO.labels(settings.llm_model_version).set(1)

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
        if request.backend == "llm":
            llm = _llm(application, settings)
            generate = partial(generate_answer, llm) if llm is not None else None
            response = solve_llm(request, generate, settings.llm_model_version)
        else:
            response = solve_mtl(request, _policy(application, settings), settings.model_version)
        _log_solve(request, response)
        record_solve(request, response)
        return response

    @application.get("/models", response_model=ModelsResponse)
    def models() -> ModelsResponse:
        """Registered versions, the one this process serves, and any broken manifest."""
        if application.state.registry is None:
            application.state.registry = _registry(settings)
        registry = application.state.registry
        active = settings.model_version
        return ModelsResponse(
            active=active,
            active_registered=any(entry.model_version == active for entry in registry.models),
            models=registry.models,
            errors=registry.errors,
        )

    @application.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        POLICY_LOADED.set(application.state.policy is not None)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

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


def _registry(settings: ServingSettings):
    """Manifest paths are relative to the project root, two levels above the manifests."""
    root = Path(settings.manifests_root) if settings.manifests_root else None
    if root is None or not root.is_dir():
        return Registry(models=[], errors=[f"no manifests directory: {root}"])
    return load_registry(root, root.parents[1])


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


def _llm(application: FastAPI, settings: ServingSettings):
    """A failed load leaves the LLM off for this process instead of stopping the MTL backend."""
    if (
        application.state.llm is None
        and settings.llm_model_path
        and not application.state.llm_failed
    ):
        from vrp_service.llm.generate import load_llm

        try:
            application.state.llm = load_llm(
                settings.llm_model_path,
                settings.llm_quantization,
                max_new_tokens=settings.llm_max_new_tokens,
                adapter_path=settings.llm_adapter_path,
            )
        except Exception as error:  # noqa: BLE001
            application.state.llm_failed = True
            logging.getLogger(__name__).error("LLM load failed: %r", error)
    return application.state.llm


app = create_app()
