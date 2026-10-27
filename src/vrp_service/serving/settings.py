"""Paths the API reads at startup. No checkpoint file means the process is not ready."""

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_VERSION = "mtl/v0.1.0"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ServingSettings:
    checkpoint_path: str | None = None
    source_root: str | None = None
    model_version: str = MODEL_VERSION
    device: str = "cpu"
    cpu_threads: int = 4
    manifests_root: str | None = None

    @classmethod
    def from_env(cls) -> "ServingSettings":
        root = project_root()
        checkpoint = root / "artifacts" / "checkpoints" / "n100" / "checkpoint-300.pt"
        source = root / "runtime"
        return cls(
            checkpoint_path=os.environ.get("MTL_CHECKPOINT") or str(checkpoint),
            source_root=os.environ.get("MTL_SOURCE") or str(source),
            model_version=os.environ.get("MTL_MODEL_VERSION") or MODEL_VERSION,
            device=os.environ.get("MTL_DEVICE") or "cpu",
            cpu_threads=int(os.environ.get("MTL_CPU_THREADS") or 4),
            manifests_root=os.environ.get("MODEL_MANIFESTS")
            or str(root / "artifacts" / "manifests"),
        )

    def weights_ready(self) -> bool:
        if self.checkpoint_path is None:
            return False
        return Path(self.checkpoint_path).is_file()
