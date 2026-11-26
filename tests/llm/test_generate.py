import pytest

from vrp_service.llm.generate import load_llm
from vrp_service.serving.settings import ServingSettings


def test_inference_defaults_to_unquantized(monkeypatch) -> None:
    monkeypatch.delenv("LLM_QUANTIZATION", raising=False)
    assert ServingSettings.from_env().llm_quantization == "none"
    monkeypatch.setenv("LLM_QUANTIZATION", "4bit")
    assert ServingSettings.from_env().llm_quantization == "4bit"


def test_an_unknown_quantization_is_refused_before_loading() -> None:
    with pytest.raises(ValueError, match="quantization"):
        load_llm("no/such/model", "int3")
