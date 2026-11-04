"""Paths the API reads at startup. No checkpoint file means the process is not ready."""

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_VERSION = "mtl/v0.1.0"
LLM_MODEL_VERSION = "llm/v0.1.0"


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
    # A local folder or a cached Hub id; unset keeps the LLM backend off.
    llm_model_path: str | None = None
    llm_model_version: str = LLM_MODEL_VERSION
    llm_max_new_tokens: int = 1024

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
            llm_model_path=os.environ.get("LLM_MODEL_PATH") or None,
            llm_model_version=os.environ.get("LLM_MODEL_VERSION") or LLM_MODEL_VERSION,
            llm_max_new_tokens=int(os.environ.get("LLM_MAX_NEW_TOKENS") or 1024),
        )

    def weights_ready(self) -> bool:
        if self.checkpoint_path is None:
            return False
        return Path(self.checkpoint_path).is_file()
