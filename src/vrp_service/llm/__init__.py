"""The local LLM backend: prompts, output parsing, and generation."""

from vrp_service.llm.parse import ROUTES_SCHEMA, ParsedRoutes, ParseError, parse_routes
from vrp_service.llm.prompt import PROMPT_VERSION, render

__all__ = [
    "PROMPT_VERSION",
    "ROUTES_SCHEMA",
    "ParseError",
    "ParsedRoutes",
    "parse_routes",
    "render",
]
