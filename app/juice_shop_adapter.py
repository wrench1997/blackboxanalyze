from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:3100"
SPLIT_SEED = "sift-js-loop12-family-split-v1"
PINNED_IMAGE = "bkimminich/juice-shop@sha256:28870b9d2bec49e605d6ebbf4b22ed1ec1ca0a72347ef19217bbbb21ea44e3fe"
TARGET_CONTAINER = "sift-loop12-juice-v20"
TARGET_NETWORK = "sift-loop12-internal"
PROXY_CONTAINER = "sift-loop12-proxy"
BLOCKED_AGENT_PATHS = ("/api/challenges", "/snippets")
ALLOWED_AGENT_METHODS = {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH"}
MAXIMUM_AGENT_BODY_BYTES = 1_048_576
MAXIMUM_EVIDENCE_PREVIEW_BYTES = 65_536

CATEGORY_FAMILIES = {
    "Broken Access Control": "access_control",
    "Broken Authentication": "authentication",
    "Improper Input Validation": "input_validation",
    "Injection": "injection",
    "Observability Failures": "observability",
    "Security Misconfiguration": "misconfiguration",
    "Sensitive Data Exposure": "information_exposure",
    "Unvalidated Redirects": "url_redirect",
    "XSS": "xss",
}

FORBIDDEN_KEY_PREFIXES = {
    "chatbot", "dlppasswordspraying", "dlppastebin", "nft", "oauth", "rce", "ssrf", "web3", "xssbonus", "xxe",
}

FORBIDDEN_TEXT = {
    "ai debugging",
    "arbitrary file write",
    "blocked rce",
    "chatbot",
    "data erasure",
    "denial of service",
    "file read",
    "gmail",
    "google account",
    "internet",
    "memory bomb",
    "mint the honey pot",
    "password spraying",
    "wallet depletion",
    "soundcloud",
}

RULE_FAMILY_TEMPLATES: dict[str, dict[str, Any]] = {
    "access_control": {
        "op": "and",
        "args": [
            {"op": "policy_slot", "name": "subject_authenticated"},
            {"op": "policy_slot", "name": "subject_authorized_for_resource"},
        ],
    },
    "authentication": {
        "op": "and",
        "args": [
            {"op": "policy_slot", "name": "identity_proof_valid"},
            {"op": "policy_slot", "name": "credential_policy_satisfied"},
        ],
    },
    "input_validation": {
        "op": "and",
        "args": [
            {"op": "policy_slot", "name": "representation_is_canonical"},
            {"op": "policy_slot", "name": "value_is_in_declared_domain"},
        ],
    },
    "injection": {
        "op": "policy_slot",
        "name": "untrusted_data_cannot_change_interpreter_structure",
    },
    "observability": {
        "op": "not",
        "arg": {"op": "policy_slot", "name": "sensitive_operational_artifact_is_public"},
    },
    "misconfiguration": {
        "op": "not",
        "arg": {"op": "policy_slot", "name": "unsafe_or_deprecated_surface_is_enabled"},
    },
    "information_exposure": {
        "op": "not",
        "arg": {"op": "policy_slot", "name": "sensitive_data_visible_without_need"},
    },
    "url_redirect": {
        "op": "origin_eq",
        "left": {"op": "policy_slot", "name": "candidate_url"},
        "right": {"op": "policy_slot", "name": "trusted_origin"},
    },
    "xss": {
        "op": "not",
        "arg": {
            "op": "html_creates_nodes",
            "arg": {"op": "policy_slot", "name": "untrusted_text"},
        },
    },
}


def _assert_local_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Juice Shop adapter only permits a local HTTP target")
    if parsed.port != 3100:
        raise ValueError("Juice Shop adapter is pinned to the isolated Loop 12 port 3100")
    return base_url.rstrip("/")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def catalog_sha256(rows: Iterable[dict[str, Any]]) -> str:
    projection = [
        {
            "id": row.get("id"),
            "key": row.get("key"),
            "name": row.get("name"),
            "category": row.get("category"),
            "difficulty": row.get("difficulty"),
            "disabledEnv": row.get("disabledEnv"),
        }
        for row in rows
    ]
    return hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()


def is_safe_challenge(row: dict[str, Any]) -> bool:
    category = str(row.get("category", ""))
    if category not in CATEGORY_FAMILIES:
        return False
    if row.get("disabledEnv") not in {None, ""}:
        return False
    key = str(row.get("key", "")).casefold()
    if any(key.startswith(prefix) for prefix in FORBIDDEN_KEY_PREFIXES):
        return False
    text = f"{row.get('name', '')} {row.get('description', '')}".casefold()
    return not any(term in text for term in FORBIDDEN_TEXT)


def select_safe_catalog(rows: Iterable[dict[str, Any]], per_family: int = 3) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not is_safe_challenge(row):
            continue
        family = CATEGORY_FAMILIES[str(row["category"])]
        selected = {
            "id": int(row["id"]),
            "key": str(row["key"]),
            "name": str(row["name"]),
            "category": str(row["category"]),
            "family": family,
            "difficulty": int(row["difficulty"]),
            "rule_ir_template": RULE_FAMILY_TEMPLATES[family],
            "rule_ir_executable_after_episode_binding": False,
        }
        grouped.setdefault(family, []).append(selected)

    result: list[dict[str, Any]] = []
    for family in sorted(grouped):
        result.extend(sorted(grouped[family], key=lambda row: row["key"])[:per_family])
    return result


def split_families(rows: Iterable[dict[str, Any]], seed: str = SPLIT_SEED) -> dict[str, list[str]]:
    families = sorted({str(row["family"]) for row in rows})
    if len(families) < 4:
        raise ValueError("at least four safe families are required")
    ordered = sorted(families, key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest())
    return {"train": ordered[:1], "validation": ordered[1:2], "hidden_test": ordered[2:]}


def attach_splits(rows: Iterable[dict[str, Any]], splits: dict[str, list[str]]) -> list[dict[str, Any]]:
    family_to_split = {family: split for split, families in splits.items() for family in families}
    return [{**row, "split": family_to_split[str(row["family"])]} for row in rows]


def agent_observation(
    *,
    action: dict[str, Any],
    status_code: int,
    response_headers: dict[str, str] | None = None,
    response_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a challenge-label-free observation suitable for an agent episode."""
    return {
        "action": action,
        "observation": {
            "status_code": int(status_code),
            "headers": dict(response_headers or {}),
            "summary": dict(response_summary or {}),
        },
    }


def _json_shape(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {str(key): _json_shape(item, depth + 1) for key, item in list(value.items())[:32]}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "items": [_json_shape(item, depth + 1) for item in value[:3]]}
    if value is None:
        return "null"
    return type(value).__name__


NONDETERMINISTIC_JSON_KEYS = {
    "createdat", "updatedat", "deletedat", "timestamp", "issuedat", "iat", "exp", "nonce", "requestid"
}


def stable_json_projection(value: Any) -> Any:
    """Remove run-specific metadata while preserving behaviorally meaningful values."""
    if isinstance(value, dict):
        return {
            str(key): stable_json_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).casefold().replace("_", "") not in NONDETERMINISTIC_JSON_KEYS
        }
    if isinstance(value, list):
        return [stable_json_projection(item) for item in value]
    return value


def _validate_agent_action(action: dict[str, Any]) -> tuple[str, str]:
    method = str(action.get("method", "GET")).upper()
    path = str(action.get("path", ""))
    if method not in ALLOWED_AGENT_METHODS:
        raise ValueError(f"method {method} is not enabled in the safe Loop 12 action space")
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise ValueError("agent path must be an origin-relative local path")
    normalized_path = path.casefold().split("?", 1)[0].rstrip("/")
    if any(normalized_path == blocked or normalized_path.startswith(f"{blocked}/") for blocked in BLOCKED_AGENT_PATHS):
        raise ValueError("evaluator-only endpoint is hidden from the agent")
    return method, path


class EvidenceLedger:
    """Append-only hash-chained evidence for local target-range episodes."""

    def __init__(self, path: Path, workspace_root: Path) -> None:
        resolved_root = workspace_root.resolve()
        resolved_path = path.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError("evidence ledger must stay inside the workspace")
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved_path
        self.previous_hash = "0" * 64
        if self.path.exists():
            lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                self.previous_hash = str(json.loads(lines[-1])["record_hash"])

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        envelope = {"previous_hash": self.previous_hash, **record}
        record_hash = hashlib.sha256(canonical_json(envelope).encode("utf-8")).hexdigest()
        stored = {**envelope, "record_hash": record_hash}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stored, sort_keys=True, ensure_ascii=False) + "\n")
        self.previous_hash = record_hash
        return stored


class JuiceShopEpisode:
    """Stateful local-only HTTP episode. Evaluator metadata never enters observations."""

    def __init__(self, adapter: "JuiceShopAdapter", ledger: EvidenceLedger | None = None) -> None:
        self.adapter = adapter
        self.ledger = ledger
        self.client = httpx.Client(
            base_url=adapter.base_url,
            timeout=adapter.timeout_seconds,
            follow_redirects=False,
        )
        self.step = 0

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "JuiceShopEpisode":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def act(self, action: dict[str, Any]) -> dict[str, Any]:
        method, path = _validate_agent_action(action)
        headers = {
            str(key): str(value)
            for key, value in dict(action.get("headers") or {}).items()
            if str(key).casefold() in {"accept", "authorization", "content-type", "x-user-email"}
        }
        request_kwargs: dict[str, Any] = {"headers": headers}
        if "json" in action:
            encoded = json.dumps(action["json"], ensure_ascii=False).encode("utf-8")
            if len(encoded) > MAXIMUM_AGENT_BODY_BYTES:
                raise ValueError("agent request body exceeds the Loop 12 limit")
            request_kwargs["json"] = action["json"]
        elif "form" in action:
            encoded = httpx.QueryParams(action["form"]).__str__().encode("utf-8")
            if len(encoded) > MAXIMUM_AGENT_BODY_BYTES:
                raise ValueError("agent request body exceeds the Loop 12 limit")
            request_kwargs["data"] = action["form"]
        elif "content" in action:
            encoded = str(action["content"]).encode("utf-8")
            if len(encoded) > MAXIMUM_AGENT_BODY_BYTES:
                raise ValueError("agent request body exceeds the Loop 12 limit")
            request_kwargs["content"] = encoded

        cookies_before = sorted(self.client.cookies.keys())
        started = time.perf_counter()
        response = self.client.request(method, path, **request_kwargs)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        body = response.content[:MAXIMUM_EVIDENCE_PREVIEW_BYTES]
        body_text = body.decode(response.encoding or "utf-8", errors="replace")
        body_sha256 = hashlib.sha256(response.content).hexdigest()
        try:
            parsed_json = response.json()
            json_shape = _json_shape(parsed_json)
            semantic_body_sha256 = hashlib.sha256(
                canonical_json(stable_json_projection(parsed_json)).encode("utf-8")
            ).hexdigest()
        except (ValueError, json.JSONDecodeError):
            json_shape = None
            semantic_body_sha256 = body_sha256

        self.step += 1
        observation = agent_observation(
            action={key: value for key, value in action.items() if key != "headers"} | {"method": method, "path": path},
            status_code=response.status_code,
            response_headers={
                key: value
                for key, value in response.headers.items()
                if key.casefold() in {"content-type", "location", "content-length", "www-authenticate"}
            },
            response_summary={
                "body_length": len(response.content),
                "body_sha256": body_sha256,
                "semantic_body_sha256": semantic_body_sha256,
                "body_preview": body_text,
                "json_shape": json_shape,
                "cookie_jar_changed": cookies_before != sorted(self.client.cookies.keys()),
                "elapsed_ms": elapsed_ms,
                "preview_truncated": len(response.content) > len(body),
            },
        )
        if self.ledger is not None:
            self.ledger.append({"step": self.step, **observation})
        return observation


class DockerJuiceShopManager:
    """Evaluator-only exact-target resetter for the disposable local container."""

    def __init__(self, adapter: "JuiceShopAdapter" | None = None) -> None:
        self.adapter = adapter or JuiceShopAdapter()

    @staticmethod
    def target_run_command() -> list[str]:
        return [
            "docker", "run", "-d",
            "--name", TARGET_CONTAINER,
            "--network", TARGET_NETWORK,
            "--add-host", "www.alchemy.com:127.0.0.1",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--memory", "1g",
            "--cpus", "2",
            PINNED_IMAGE,
        ]

    @staticmethod
    def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, check=check, capture_output=True, text=True, timeout=60)

    def reset(self, environment_seed: int, timeout_seconds: float = 120.0) -> dict[str, Any]:
        exact = self._run(
            ["docker", "ps", "-a", "--filter", f"name=^/{TARGET_CONTAINER}$", "--format", "{{.Names}}"]
        ).stdout.strip()
        if exact and exact != TARGET_CONTAINER:
            raise RuntimeError("refusing to reset an ambiguously matched container")
        if exact == TARGET_CONTAINER:
            self._run(["docker", "rm", "-f", TARGET_CONTAINER])
        container_id = self._run(self.target_run_command()).stdout.strip()
        proxy_exact = self._run(
            ["docker", "ps", "-a", "--filter", f"name=^/{PROXY_CONTAINER}$", "--format", "{{.Names}}"]
        ).stdout.strip()
        if proxy_exact != PROXY_CONTAINER:
            raise RuntimeError("fixed local ingress proxy is missing or ambiguous")
        self._run(["docker", "restart", PROXY_CONTAINER])
        deadline = time.monotonic() + timeout_seconds
        last_health: dict[str, Any] = {"reachable": False, "status_code": 0}
        while time.monotonic() < deadline:
            try:
                last_health = self.adapter.health()
                if last_health["reachable"]:
                    break
            except (httpx.HTTPError, OSError):
                pass
            time.sleep(0.5)
        if not last_health["reachable"]:
            raise RuntimeError("fresh local Juice Shop did not become healthy")
        inspect_rows = json.loads(self._run(["docker", "inspect", TARGET_CONTAINER]).stdout)
        inspect_row = inspect_rows[0]
        networks = inspect_row["NetworkSettings"]["Networks"]
        ports = inspect_row["NetworkSettings"]["Ports"]
        image = inspect_row["Image"]
        expected_image = PINNED_IMAGE.rsplit("@", 1)[1]
        if set(networks) != {TARGET_NETWORK} or any(bool(binding) for binding in ports.values()) or image != expected_image:
            raise RuntimeError("fresh target does not satisfy the internal-only network invariant")
        solved = self.adapter.evaluator_solved_state()
        return {
            "environment_seed": int(environment_seed),
            "container_id": container_id,
            "health": last_health,
            "network_invariant": "target internal-only; fixed local proxy is the sole ingress",
            "initial_solved_count": sum(solved.values()),
            "challenge_count": len(solved),
        }


@dataclass(frozen=True)
class JuiceShopAdapter:
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _assert_local_base_url(self.base_url))

    def _get_json(self, path: str) -> dict[str, Any]:
        if not path.startswith("/"):
            raise ValueError("path must be origin-relative")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds, follow_redirects=False) as client:
            response = client.get(path)
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        return body

    def health(self) -> dict[str, Any]:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds) as client:
            response = client.get("/")
        return {"reachable": response.status_code == 200, "status_code": response.status_code}

    def evaluator_catalog(self) -> list[dict[str, Any]]:
        body = self._get_json("/api/Challenges")
        rows = body.get("data")
        if not isinstance(rows, list):
            raise ValueError("challenge catalog is missing data[]")
        return [dict(row) for row in rows if isinstance(row, dict)]

    def evaluator_solved_state(self) -> dict[str, bool]:
        return {str(row["key"]): bool(row.get("solved")) for row in self.evaluator_catalog()}

    def safe_split_manifest(self) -> dict[str, Any]:
        catalog = self.evaluator_catalog()
        selected = select_safe_catalog(catalog)
        splits = split_families(selected)
        return {
            "catalog_count": len(catalog),
            "catalog_sha256": catalog_sha256(catalog),
            "selection_count": len(selected),
            "splits": splits,
            "challenges": attach_splits(selected, splits),
        }
