"""Load Qwen2.5-7B-Instruct (bf16 by default) and generate one JSON answer per request.

Decoding is greedy, so the baseline is reproducible, and the tokens are restricted to
``ROUTES_SCHEMA`` by lm-format-enforcer, with no whitespace, so the answer is compact JSON.
Imports torch and transformers lazily, so the API image without the ``llm`` group starts.
"""

from dataclasses import dataclass
from typing import Any

from vrp_service.env import EnvironmentInstance
from vrp_service.llm.parse import ROUTES_SCHEMA

MAX_NEW_TOKENS = 1024
# bitsandbytes saves memory but decodes more slowly than bf16 on an A100 (8-bit: ~8.5 tokens/s on
# the smoke run), so inference defaults to "none"; 4-bit is for QLoRA training.
QUANTIZATIONS = ("none", "8bit", "4bit")


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    hit_limit: bool


@dataclass(frozen=True)
class LoadedLlm:
    model: Any
    tokenizer: Any
    tokenizer_data: Any
    max_new_tokens: int
    # The enforcer allows runs of whitespace between JSON tokens, and its config is replaced
    # inside the transformers integration, so whitespace tokens are banned here instead.
    whitespace_tokens: frozenset[int]
    # Every token's text, for the feasibility mask.
    vocab: tuple[tuple[int, str], ...]


def load_llm(
    model_path: str,
    quantization: str = "none",
    max_new_tokens: int = MAX_NEW_TOKENS,
    adapter_path: str | None = None,
) -> LoadedLlm:
    """``model_path`` is a local folder or a Hub id already in the cache. A LoRA adapter, if
    given, is merged into the base (it was trained on a 4-bit base; it is scored merged)."""
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS}, not {quantization!r}")
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = {
        "none": None,
        "8bit": BitsAndBytesConfig(load_in_8bit=True),
        "4bit": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="bfloat16"
        ),
    }[quantization]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=config, dtype="auto", device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    model.eval()
    vocab = tuple((token, tokenizer.decode([token])) for token in range(len(tokenizer)))
    whitespace = frozenset(token for token, text in vocab if any(c.isspace() for c in text))
    return LoadedLlm(
        model,
        tokenizer,
        build_token_enforcer_tokenizer_data(tokenizer),
        max_new_tokens,
        whitespace,
        vocab,
    )


def generate(
    llm: LoadedLlm, messages: list[dict[str, str]], mask_for: EnvironmentInstance | None = None
) -> Generation:
    """``mask_for`` switches to feasibility-masked decoding (``vrp_service.llm.mask``),
    which replaces the schema enforcer: its grammar is the same JSON, narrowed further."""
    import torch
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    tokenizer = llm.tokenizer
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(llm.model.device)
    prompt_tokens = inputs["input_ids"].shape[1]
    if mask_for is None:
        schema_tokens = build_transformers_prefix_allowed_tokens_fn(
            llm.tokenizer_data, JsonSchemaParser(ROUTES_SCHEMA)
        )

        def allowed(batch: int, prefix: Any) -> list[int]:
            return [t for t in schema_tokens(batch, prefix) if t not in llm.whitespace_tokens]
    else:
        from vrp_service.llm.mask import RouteMask

        mask = RouteMask(mask_for, llm.vocab, tokenizer.eos_token_id)

        def allowed(batch: int, prefix: Any) -> list[int]:
            return mask.allowed(tokenizer.decode(prefix[prompt_tokens:]))

    with torch.inference_mode():
        output = llm.model.generate(
            **inputs,
            max_new_tokens=llm.max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=allowed,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    new = output[0, prompt_tokens:]
    ended = bool((new == tokenizer.eos_token_id).any())
    return Generation(
        text=tokenizer.decode(new, skip_special_tokens=True),
        prompt_tokens=prompt_tokens,
        completion_tokens=int(new.shape[0]),
        hit_limit=not ended and new.shape[0] >= llm.max_new_tokens,
    )
