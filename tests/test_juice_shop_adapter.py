import pytest

from app.juice_shop_adapter import (
    DockerJuiceShopManager,
    EvidenceLedger,
    JuiceShopAdapter,
    _validate_agent_action,
    agent_observation,
    attach_splits,
    is_safe_challenge,
    select_safe_catalog,
    split_families,
    stable_json_projection,
)


def row(key: str, name: str, category: str, *, disabled=None, difficulty: int = 1, ident: int = 1):
    return {
        "id": ident,
        "key": key,
        "name": name,
        "category": category,
        "difficulty": difficulty,
        "disabledEnv": disabled,
        "description": "",
    }


def test_adapter_rejects_nonlocal_or_wrong_port_targets():
    with pytest.raises(ValueError):
        JuiceShopAdapter("https://demo.owasp-juice.shop")
    with pytest.raises(ValueError):
        JuiceShopAdapter("http://127.0.0.1:3000")
    assert JuiceShopAdapter("http://localhost:3100").base_url == "http://localhost:3100"


def test_safety_filter_excludes_disabled_external_and_destructive_tasks():
    assert is_safe_challenge(row("adminSectionChallenge", "Admin Section", "Broken Access Control"))
    assert not is_safe_challenge(row("ssrfChallenge", "SSRF", "Broken Access Control"))
    assert not is_safe_challenge(row("rceChallenge", "Blocked RCE DoS", "Insecure Deserialization"))
    assert not is_safe_challenge(row("localXssChallenge", "DOM XSS", "XSS", disabled="Docker"))
    assert not is_safe_challenge(row("nftMintChallenge", "Mint the Honey Pot", "Improper Input Validation"))
    external = row("otherwiseSafe", "Ordinary Task", "Sensitive Data Exposure")
    external["description"] = "Dumpster dive the Internet for leaked data"
    assert not is_safe_challenge(external)
    assert is_safe_challenge(row("redirectChallenge", "Enforce a redirect", "Unvalidated Redirects"))


def test_safe_selection_is_stable_and_capped_per_family():
    rows = [
        row("c", "C", "Broken Access Control", ident=3),
        row("a", "A", "Broken Access Control", ident=1),
        row("d", "D", "Broken Access Control", ident=4),
        row("b", "B", "Broken Access Control", ident=2),
        row("auth", "Auth", "Broken Authentication", ident=5),
        row("input", "Input", "Improper Input Validation", ident=6),
        row("inject", "Inject", "Injection", ident=7),
    ]
    selected = select_safe_catalog(rows)
    access_keys = [item["key"] for item in selected if item["family"] == "access_control"]
    assert access_keys == ["a", "b", "c"]
    assert all("rule_ir_template" in item for item in selected)


def test_family_split_is_complete_and_deterministic():
    selected = [
        {"family": family, "key": f"{family}-{index}"}
        for family in ("access_control", "authentication", "input_validation", "injection")
        for index in range(2)
    ]
    first = split_families(selected)
    second = split_families(list(reversed(selected)))
    assert first == second
    attached = attach_splits(selected, first)
    assignments = {}
    for item in attached:
        assignments.setdefault(item["family"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in assignments.values())
    assert len(first["hidden_test"]) == 2


def test_agent_observation_does_not_expose_evaluator_metadata():
    observation = agent_observation(
        action={"kind": "http", "method": "GET", "path": "/rest/products/search?q=x"},
        status_code=200,
        response_headers={"content-type": "application/json"},
        response_summary={"json_shape": {"data": "list"}, "body_length": 42},
    )
    serialized = str(observation).casefold()
    for forbidden in ("challenge", "difficulty", "hint", "solution", "cwe", "solved"):
        assert forbidden not in serialized


def test_agent_action_boundary_hides_evaluator_and_unsafe_methods():
    assert _validate_agent_action({"method": "GET", "path": "/rest/products/search?q=apple"}) == (
        "GET", "/rest/products/search?q=apple"
    )
    for action in (
        {"method": "GET", "path": "https://example.com"},
        {"method": "GET", "path": "//example.com/path"},
        {"method": "GET", "path": "/api/Challenges"},
        {"method": "GET", "path": "/snippets/localXssChallenge"},
        {"method": "DELETE", "path": "/api/Users/1"},
    ):
        with pytest.raises(ValueError):
            _validate_agent_action(action)


def test_evidence_ledger_is_hash_chained_and_workspace_scoped(tmp_path):
    ledger = EvidenceLedger(tmp_path / "artifacts" / "episode.jsonl", tmp_path)
    first = ledger.append({"step": 1, "value": "a"})
    second = ledger.append({"step": 2, "value": "b"})
    assert first["previous_hash"] == "0" * 64
    assert second["previous_hash"] == first["record_hash"]
    with pytest.raises(ValueError):
        EvidenceLedger(tmp_path.parent / "outside.jsonl", tmp_path)


def test_reset_command_is_exact_pinned_and_has_no_published_port():
    command = DockerJuiceShopManager.target_run_command()
    assert command[-1].startswith("bkimminich/juice-shop@sha256:")
    assert command[command.index("--name") + 1] == "sift-loop12-juice-v20"
    assert command[command.index("--network") + 1] == "sift-loop12-internal"
    assert command[command.index("--add-host") + 1] == "www.alchemy.com:127.0.0.1"
    assert "--publish" not in command
    assert "--cap-drop" in command


def test_stable_json_projection_removes_run_metadata_not_behavior():
    left = {"data": [{"id": 1, "name": "Apple", "createdAt": "t1", "updated_at": "u1"}]}
    right = {"data": [{"id": 1, "name": "Apple", "createdAt": "t2", "updated_at": "u2"}]}
    changed = {"data": [{"id": 1, "name": "Orange", "createdAt": "t2", "updated_at": "u2"}]}
    assert stable_json_projection(left) == stable_json_projection(right)
    assert stable_json_projection(left) != stable_json_projection(changed)
