import hashlib
import json
from pathlib import Path

import pytest

from vrp_service.serving.settings import MODEL_VERSION

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "artifacts" / "manifests" / "mtl" / "v0.1.0.json").read_text())
FILES = {entry["role"]: entry for entry in MANIFEST["files"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_manifest_names_the_served_version() -> None:
    assert f"{MANIFEST['name']}/v{MANIFEST['version']}" == MODEL_VERSION


def test_config_matches_the_manifest() -> None:
    entry = FILES["config"]
    assert _sha256(ROOT / entry["path"]) == entry["sha256"]


def test_checkpoint_matches_the_manifest() -> None:
    entry = FILES["checkpoint"]
    path = ROOT / entry["path"]
    if not path.is_file():
        pytest.skip("the n=100 checkpoint is not on disk")
    assert path.stat().st_size == entry["bytes"]
    assert _sha256(path) == entry["sha256"]
