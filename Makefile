run:
	uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest -q

docker:
	docker compose up --build
