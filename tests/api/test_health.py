from pathlib import Path

from fastapi.testclient import TestClient

from vrp_service.serving.app import create_app
from vrp_service.serving.settings import ServingSettings


def test_health_ok() -> None:
    response = TestClient(create_app(ServingSettings())).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_follows_the_checkpoint_file(monkeypatch) -> None:
    monkeypatch.delenv("MTL_CHECKPOINT", raising=False)
    response = TestClient(create_app()).get("/ready")
    checkpoint = ServingSettings.from_env().checkpoint_path
    assert response.status_code == 200
    assert response.json() == {"ready": Path(checkpoint).is_file()}


def test_default_paths_stay_inside_the_project(monkeypatch) -> None:
    monkeypatch.delenv("MTL_CHECKPOINT", raising=False)
    monkeypatch.delenv("MTL_SOURCE", raising=False)
    settings = ServingSettings.from_env()
    root = Path(__file__).resolve().parents[2]
    assert Path(settings.checkpoint_path).resolve().is_relative_to(root)
    assert Path(settings.source_root).resolve().is_relative_to(root)
    assert Path(settings.checkpoint_path).name == "checkpoint-300.pt"
