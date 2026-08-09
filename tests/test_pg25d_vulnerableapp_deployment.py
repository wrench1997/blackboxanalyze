import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict:
    return json.loads(
        (ROOT / "research" / "pg_pk_25d_vulnerableapp_deployment_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_pg25d_is_pinned_loopback_only_and_evaluation_only():
    manifest = _manifest()
    source = manifest["source"]
    deployment = manifest["deployment"]
    safety = manifest["safety"]
    training = manifest["training"]

    assert source["image_digest"].startswith("sha256:")
    assert len(source["image_digest"]) == len("sha256:") + 64
    assert source["image_ref"].endswith(":2.1.44")
    assert deployment["host_binding"] == "127.0.0.1:19090"
    assert deployment["external_network"] is False
    assert deployment["network_options"]["com.docker.network.bridge.enable_ip_masquerade"] is False
    assert deployment["read_only_rootfs"] is True
    assert deployment["cap_drop"] == ["ALL"]
    assert deployment["no_new_privileges"] is True
    assert training["training_eligible"] is False
    assert training["evaluation_only"] is True
    assert safety["local_only"] is True
    assert safety["loopback_host_binding"] is True
    assert safety["destructive_probes_run"] is False


def test_pg25d_reset_adapter_hash_matches_manifest():
    manifest = _manifest()
    reset_script = ROOT / manifest["reset"]["reset_script"]
    actual = _sha256(reset_script)
    assert actual == manifest["source"]["reset_adapter_sha256"]
    assert actual == manifest["reset"]["reset_adapter_sha256"]
