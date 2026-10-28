"""The local LLM backend: prompts, output parsing, and generation."""

from vrp_service.llm.prompt import PROMPT_VERSION, render

__all__ = ["PROMPT_VERSION", "render"]
