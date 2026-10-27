"""Send solve traces where the standard OpenTelemetry variables say.

``OTEL_EXPORTER_OTLP_ENDPOINT`` set: spans go to that collector (for example Jaeger).
``OTEL_TRACES_EXPORTER=console``: spans print to stdout as JSON.
Neither: no provider is installed and spans cost almost nothing.
"""

import os

from opentelemetry import trace

SERVICE_NAME = "vrp-decision-service"
tracer = trace.get_tracer("vrp_service")
_configured = False


def configure_tracing() -> None:
    global _configured
    if _configured:
        return
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    console = os.environ.get("OTEL_TRACES_EXPORTER") == "console"
    if not endpoint and not console:
        return

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
