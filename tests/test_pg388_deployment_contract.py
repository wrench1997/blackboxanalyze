from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pg388_backend_image_and_run_contract_are_explicit() -> None:
    dockerfile = (ROOT / "fixtures/pg388/Dockerfile").read_text(encoding="utf-8")
    readme = (ROOT / "fixtures/pg388/README.md").read_text(encoding="utf-8")
    assert "PYTHON_IMAGE_DIGEST" in dockerfile
    assert "USER nobody" in dockerfile
    assert "network=none" in readme
    assert "127.0.0.1:8088" in readme
    assert "safe_to_send=false" in readme


def test_pg388_two_container_display_is_loopback_published_and_internal() -> None:
    compose = (ROOT / "docker-compose.pg388.yml").read_text(encoding="utf-8")
    frontend_dockerfile = (ROOT / "frontend/Dockerfile").read_text(encoding="utf-8")
    next_config = (ROOT / "frontend/next.config.mjs").read_text(encoding="utf-8")
    assert "internal: true" in compose
    assert '127.0.0.1:3000:3000' in compose
    assert "PG388_PYTHON_IMAGE_DIGEST" in compose
    assert "PG388_NODE_BASE_IMAGE" in compose
    assert "pg388-logic-backend:local" in compose
    assert "pg388-frontend:local" in compose
    assert "pg388_display" in compose
    assert "pg388_internal" in compose
    assert "NODE_BASE_IMAGE" in frontend_dockerfile
    assert "standalone" in frontend_dockerfile
    assert 'output: "standalone"' in next_config
    proxy = (ROOT / "frontend/app/pg388-api/[...path]/route.ts").read_text(encoding="utf-8")
    assert "ALLOWED_PATHS" in proxy
    assert "pg388-backend" in proxy
    assert "MAX_REQUEST_BYTES" in proxy
    assert "MAX_RESPONSE_BYTES" in proxy
    assert "blocked_backend_route" in proxy
    assert "payload" not in proxy.lower()
    dockerignore = (ROOT / "frontend/.dockerignore").read_text(encoding="utf-8")
    assert "node_modules" in dockerignore
    assert ".next" in dockerignore
