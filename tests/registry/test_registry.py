import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vrp_service.registry import load_manifest, load_registry
from vrp_service.serving.app import create_app
from vrp_service.serving.settings import ServingSettings, project_root
from vrp_service.serving.solve import CVRP_EXAMPLE

ROOT = project_root()
MANIFESTS = ROOT / "artifacts" / "manifests"
GOOD = json.loads((MANIFESTS / "mtl" / "v0.1.0.json").read_text())


def _write(root: Path, name: str, version: str, body: dict) -> Path:
    path = root / name / f"v{version}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))
    return path


def test_the_shipped_manifest_loads() -> None:
    manifest = load_manifest(MANIFESTS / "mtl" / "v0.1.0.json")
    assert manifest.model_version == "mtl/v0.1.0"


@pytest.mark.parametrize(
    "change",
    [
        {"version": "0.1"},
        {"backend": "other"},
        {"files": []},
        {"files": [{**GOOD["files"][0], "sha256": "abc"}]},
        {"files": [GOOD["files"][1], GOOD["files"][1]]},
        {"name": "MTL"},
    ],
)
def test_a_malformed_manifest_is_rejected(tmp_path, change) -> None:
    body = {**GOOD, **change}
    path = _write(tmp_path, "mtl", "0.1.0", body)
    with pytest.raises(ValueError):
        load_manifest(path)


def test_a_manifest_filed_under_the_wrong_version_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        load_manifest(_write(tmp_path, "mtl", "0.2.0", GOOD))


def test_invalid_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "mtl" / "v0.1.0.json"
    path.parent.mkdir()
    path.write_text("{")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_the_registry_lists_good_manifests_and_reports_broken_ones(tmp_path) -> None:
    project = tmp_path / "project"
    manifests = project / "artifacts" / "manifests"
    (project / "runtime").mkdir(parents=True)
    shutil.copy(ROOT / "runtime" / "config.yaml", project / "runtime" / "config.yaml")
    _write(manifests, "mtl", "0.1.0", GOOD)
    _write(manifests, "mtl", "0.2.0", {**GOOD, "version": "0.2.0", "backend": "nope"})
    registry = load_registry(manifests, project)
    [entry] = registry.models
    assert entry.model_version == "mtl/v0.1.0"
    files = {status.role: status for status in entry.files}
    assert files["config"].present and files["config"].verified is True
    assert not files["checkpoint"].present and files["checkpoint"].verified is None
    [error] = registry.errors
    assert "v0.2.0.json" in error


def test_a_changed_file_fails_verification(tmp_path) -> None:
    project = tmp_path / "project"
    (project / "runtime").mkdir(parents=True)
    (project / "runtime" / "config.yaml").write_text("changed: true\n")
    manifests = project / "artifacts" / "manifests"
    _write(manifests, "mtl", "0.1.0", GOOD)
    files = {status.role: status for status in load_registry(manifests, project).models[0].files}
    assert files["config"].verified is False


@pytest.mark.parametrize("version", ["mtl/v0.1.0", "mtl/v9.0.0"])
def test_models_reports_the_served_version(version: str) -> None:
    settings = ServingSettings(model_version=version, manifests_root=str(MANIFESTS))
    application = create_app(settings)
    application.state.policy = lambda _i: {"reward": [-2.5], "tours": [[0, 1, 0]]}
    client = TestClient(application)
    body = client.get("/models").json()
    assert body["active"] == version
    assert body["active_registered"] is (version == "mtl/v0.1.0")
    assert [model["model_version"] for model in body["models"]] == ["mtl/v0.1.0"]
    assert body["errors"] == []
    assert client.post("/solve", json=CVRP_EXAMPLE).json()["model_version"] == version


def test_models_without_a_manifests_directory_reports_an_error() -> None:
    body = TestClient(create_app(ServingSettings())).get("/models").json()
    assert body["models"] == [] and len(body["errors"]) == 1
