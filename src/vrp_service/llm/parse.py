"""Turn the model's answer into routes, or say why it cannot be used.

No repair: an answer that is not ``{"routes": [[int, ...], ...]}`` raises ``ParseError``,
and a parsed answer is judged by the same ``check_routes`` as the MTL policy.
"""

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError

from vrp_service.contracts import SolveRequest
from vrp_service.env import Violation, check_routes, load_instance

# An opening fence with an optional language tag, and a closing fence if generation got there.
FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*(?:```)?$", re.DOTALL)


class RoutesAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routes: list[list[StrictInt]]


# Constrains generation in the backend, so the model can only write this shape.
ROUTES_SCHEMA = RoutesAnswer.model_json_schema()


class ParseError(ValueError):
    """``kind`` is ``truncated``, ``invalid_json``, or ``schema``."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


@dataclass(frozen=True)
class ParsedRoutes:
    routes: list[list[int]]
    violations: list[Violation]


def parse_routes(text: str, request: SolveRequest) -> ParsedRoutes:
    text = text.strip()
    fenced = FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        # The decoder failing at the very end means the text stopped mid-answer.
        kind = "truncated" if error.pos >= len(text) else "invalid_json"
        raise ParseError(kind, error.msg) from error
    except (RecursionError, ValueError) as error:
        # Pathological nesting, or an integer past Python's digit limit.
        raise ParseError("invalid_json", type(error).__name__) from error
    try:
        answer = RoutesAnswer.model_validate(data)
    except ValidationError as error:
        raise ParseError("schema", str(error.errors()[0]["msg"])) from error

    ending = [] if request.instance.open_route else [0]
    routes = [[0, *customers, *ending] for customers in answer.routes]
    return ParsedRoutes(routes, check_routes(load_instance(request), routes))
