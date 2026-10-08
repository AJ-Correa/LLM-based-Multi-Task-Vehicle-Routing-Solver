.PHONY: test lint api docker

test:
	uv run pytest -q

lint:
	uv run ruff check

api:
	uv run uvicorn vrp_service.serving.app:app --host 0.0.0.0 --port 8000

docker:
	docker compose -f docker/docker-compose.yml up --build
