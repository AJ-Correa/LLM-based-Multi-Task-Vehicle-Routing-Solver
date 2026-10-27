import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vrp_service.contracts import SolveRequest
from vrp_service.serving.solve import CVRP_EXAMPLE, solve_mtl

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider():
    # The global provider can be set once per process; tracing.py leaves it unset without env.
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(provider)


@pytest.fixture(autouse=True)
def clear():
    EXPORTER.clear()


def _spans():
    return {span.name: span for span in EXPORTER.get_finished_spans()}


def test_a_solve_span_carries_the_request_and_the_outcome() -> None:
    request = SolveRequest.model_validate(CVRP_EXAMPLE)
    solve_mtl(request, lambda _i: {"reward": [-2.5], "tours": [[0, 1, 0]]}, "mtl/v9.9.9")
    spans = _spans()
    assert set(spans) == {"solve", "decode", "split_routes", "check_routes"}
    root = spans["solve"].attributes
    assert root["vrp.backend"] == "mtl"
    assert root["vrp.variant"] == "cvrp"
    assert root["vrp.customers"] == 1
    assert root["vrp.model_version"] == "mtl/v9.9.9"
    assert root["vrp.status"] == "ok"
    assert root["vrp.feasible"] is True
    assert root["vrp.objective_value"] == 2.5
    for child in ("decode", "split_routes", "check_routes"):
        assert spans[child].parent.span_id == spans["solve"].context.span_id
    assert tuple(spans["check_routes"].attributes["vrp.violations"]) == ()


def test_an_error_is_recorded_on_the_span() -> None:
    solve_mtl(SolveRequest.model_validate(CVRP_EXAMPLE), None)
    spans = _spans()
    assert set(spans) == {"solve"}
    assert spans["solve"].attributes["vrp.status"] == "error"
    assert spans["solve"].attributes["vrp.error_type"] == "model_unavailable"
