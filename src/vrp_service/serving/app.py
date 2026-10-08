"""Process checks. No model is loaded in this build, so readiness stays false."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from vrp_service import __version__

app = FastAPI(title="LLM-based VRP decision support", version=__version__)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    ready: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    return ReadyResponse(ready=False)
