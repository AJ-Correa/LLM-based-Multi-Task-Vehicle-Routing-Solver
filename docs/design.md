# Design notes

## Observability

Each `POST /solve` leaves three signals: a trace, Prometheus metrics, and one JSON log line.
All three carry the backend, variant, and model version.

### Traces

`solve_mtl` opens a `solve` span with child spans for each stage. The exporter follows the
standard OpenTelemetry variables:

| Setting | Where spans go |
| --- | --- |
| `OTEL_EXPORTER_OTLP_ENDPOINT=http://host:4318` | An OTLP collector, for example Jaeger |
| `OTEL_TRACES_EXPORTER=console` | stdout, one JSON object per span |
| neither | Nowhere; no provider is installed |

FastAPI adds the HTTP span (`POST /solve`) itself, so the solve spans nest under it. A sample
from a real request (a 100-customer `vrptw` instance on the L40S, the first request after
startup, so the decode includes CUDA warm-up):

```text
POST /solve                     971.4 ms  http.route=/solve, http.response.status_code=200
└─ fastapi.endpoint             967.6 ms
   └─ solve                     967.1 ms  vrp.backend=mtl, vrp.variant=vrptw, vrp.customers=100,
      │                                   vrp.model_version=mtl/v0.1.0, vrp.status=ok,
      │                                   vrp.feasible=true, vrp.vehicles=18,
      │                                   vrp.objective_value=24.197
      ├─ decode                 966.0 ms
      ├─ split_routes             0.1 ms
      └─ check_routes             0.4 ms  vrp.violations=[]
```

The decode is the whole cost, as in `docs/results.md`. On an error the `solve` span has
`vrp.status=error` and `vrp.error_type` (for example `model_unavailable`) and no children.

### Metrics

`GET /metrics` serves the Prometheus text format:

| Metric | Type | Labels |
| --- | --- | --- |
| `vrp_solves_total` | counter | backend, variant, status, feasible, error_type |
| `vrp_solve_latency_seconds` | histogram | backend, variant (successful solves only) |
| `vrp_policy_loaded` | gauge | — |
| `vrp_model_info` | gauge, always 1 | model_version |

The latency is the same clock as `latency_ms` in the response. Buckets are dense between
0.25 s and 1 s, where solves at 100 customers fall. Sample after the request above:

```text
vrp_solves_total{backend="mtl",error_type="",feasible="true",status="ok",variant="vrptw"} 1.0
vrp_solve_latency_seconds_count{backend="mtl",variant="vrptw"} 1.0
vrp_solve_latency_seconds_sum{backend="mtl",variant="vrptw"} 0.967
vrp_policy_loaded 1.0
vrp_model_info{model_version="mtl/v0.1.0"} 1.0
```

The scrape config is `docker/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: vrp-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

To run the API with Prometheus (http://localhost:9090) and Jaeger (http://localhost:16686):

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318 \
  docker compose -f docker/docker-compose.yml --profile obs up --build
```

### Logs

One JSON line per solve on stdout, written by the HTTP path only (the eval harness does not
log): backend, variant, customers, model version, status, error type, feasible, cost, vehicles,
latency.

## Model registry

Every served version has a manifest at `artifacts/manifests/<name>/v<version>.json`. The model
version string is `<name>/v<version>`, for example `mtl/v0.1.0`. `GET /models` lists the valid
manifests, marks which version this process serves (`MTL_MODEL_VERSION`), and reports each file
as present and verified (SHA-256 matches) or absent. A manifest that fails validation is listed
under `errors` instead of `models`. The checksums are computed on the first `/models` call.

To add a version:

1. Put the files in place (weights stay out of git; publish them as a release asset).
2. Write `artifacts/manifests/<name>/v<version>.json` with `name`, `version`, `backend` (`mtl` or
   `llm`), `created_at`, `description`, and `files`: one entry per role with `path`, `sha256`,
   `in_git`, and optionally `bytes` and `url`. Get the hash with `sha256sum <file>`.
3. Serve it with `MTL_MODEL_VERSION=<name>/v<version>`. `/models` then shows it as `active`, and
   every solve response, log line, span, and the `vrp_model_info` metric carry that version.
4. Run `make test`; the manifest tests check the schema, the file name, and the hashes of the
   files on disk.

## Contract version

Requests and responses carry `contract_version` (now `1.0.0`). It changes when a field is
added, removed, or changes meaning: a minor version for an optional addition, a major version
for anything an existing client would break on. Changes are recorded here.

| Version | Change |
| --- | --- |
| 1.0.0 | First contract: `POST /solve` with the 16 single-depot variants |

## Experiment tracking

The LoRA runs (F5) log to a self-hosted MLflow server with its own Postgres database:

```bash
docker compose -f docker/docker-compose.yml --profile mlflow up -d --build
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Postgres (`mlflow-db`) stores experiments, runs, parameters, and metrics; artifacts go to the
`mlflow-artifacts` volume and are served through the tracking server. Both volumes survive
`down`; `down -v` deletes them. Set `MLFLOW_DB_PASSWORD` to change the default password.
