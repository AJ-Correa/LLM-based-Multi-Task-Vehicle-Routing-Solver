.PHONY: test lint api

test:
	uv run pytest -q

lint:
	uv run ruff check

api:
	uv run uvicorn vrp_service.serving.app:app --host 0.0.0.0 --port 8000
