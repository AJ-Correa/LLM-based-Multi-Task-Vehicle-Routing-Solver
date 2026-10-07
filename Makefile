.PHONY: test lint api

test:
	uv run pytest -q

lint:
	uv run ruff check

# The application module arrives in the next commit. This target is the command CI and the README will use.
api:
	uv run uvicorn vrp_service.serving.app:app --host 0.0.0.0 --port 8000
