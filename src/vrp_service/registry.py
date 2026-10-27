"""Model versions recorded under ``artifacts/manifests/<name>/v<version>.json``.

A manifest names its files with their SHA-256. Loading validates the schema and that
the file name matches the version inside; a malformed manifest is reported as an
error instead of being listed. Checksums are compared only for files on disk.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class ManifestFile(BaseModel):
    role: str
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(default=None, ge=0)
    in_git: bool
    url: str | None = None


class Manifest(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    backend: Literal["mtl", "llm"]
    created_at: str
    description: str
    files: list[ManifestFile] = Field(min_length=1)
    metrics_ref: str | None = None

    @property
    def model_version(self) -> str:
        return f"{self.name}/v{self.version}"

    @field_validator("files")
    @classmethod
    def roles_are_unique(cls, files: list[ManifestFile]) -> list[ManifestFile]:
        roles = [entry.role for entry in files]
        if len(roles) != len(set(roles)):
            raise ValueError("each file role appears once")
        return files


class FileStatus(BaseModel):
    role: str
    path: str
    present: bool
    verified: bool | None = None


class ModelEntry(BaseModel):
    model_version: str
    backend: str
    created_at: str
    description: str
    files: list[FileStatus]


class Registry(BaseModel):
    models: list[ModelEntry]
    errors: list[str]


def load_manifest(path: Path) -> Manifest:
    """Raise ``ValueError`` when the manifest is malformed or misnamed."""
    try:
        manifest = Manifest.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(f"{path.name}: {error}") from error
    if path.stem != f"v{manifest.version}" or path.parent.name != manifest.name:
        raise ValueError(f"{path}: file name does not match {manifest.model_version}")
    return manifest


def load_registry(manifests_root: Path, project_root: Path) -> Registry:
    models, errors = [], []
    for path in sorted(manifests_root.glob("*/v*.json")):
        try:
            manifest = load_manifest(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        models.append(
            ModelEntry(
                model_version=manifest.model_version,
                backend=manifest.backend,
                created_at=manifest.created_at,
                description=manifest.description,
                files=[_status(entry, project_root) for entry in manifest.files],
            )
        )
    return Registry(models=models, errors=errors)


def _status(entry: ManifestFile, project_root: Path) -> FileStatus:
    path = project_root / entry.path
    if not path.is_file():
        return FileStatus(role=entry.role, path=entry.path, present=False)
    return FileStatus(
        role=entry.role, path=entry.path, present=True, verified=sha256(path) == entry.sha256
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
