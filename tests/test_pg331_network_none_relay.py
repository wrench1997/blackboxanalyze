from __future__ import annotations

import pytest

from app.pg331_network_none_relay import CONTAINER_NAME, IMAGE, TARGET_URL, attest_container_inspection, build_container_command, build_relay_contract, response_projection, role_container_name


def _inspection(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {"Name": "/" + CONTAINER_NAME, "Config": {"Image": IMAGE}, "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True}, "NetworkSettings": {"Ports": {}}, "Mounts": []}
    value.update(overrides)
    return value


def test_command_and_contract_are_network_none_without_published_ports_or_mounts() -> None:
    command = build_container_command()
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--publish" not in command and "--volume" not in command
    contract = build_relay_contract()
    assert contract["forwarding"]["target_url"] == TARGET_URL
    assert contract["host_relay"]["bind_host"] == "127.0.0.1"
    assert contract["legacy_bridge_lifecycle_compatible"] is False
    assert contract["execution"]["docker_started"] is False


def test_inspection_attestation_rejects_bridge_ports_mounts_and_wrong_image() -> None:
    assert attest_container_inspection(_inspection())["valid"] is True
    for bad in (
        _inspection(HostConfig={"NetworkMode": "pg25-vulnerableapp-hostonly", "ReadonlyRootfs": True}),
        _inspection(NetworkSettings={"Ports": {"9090/tcp": [{"HostPort": "19090"}]}}),
        _inspection(Mounts=[{"Type": "bind"}]),
        _inspection(Config={"Image": "other"}),
    ):
        assert attest_container_inspection(bad)["valid"] is False


def test_fixed_target_and_response_projection_are_bounded_and_no_body() -> None:
    with pytest.raises(ValueError): build_relay_contract(target_url="http://elsewhere/")
    result = response_projection(method="POST", status=302, headers={"Content-Type": "text/html", "Location": "/done"}, body_length=12)
    assert result["status_class"] == "3xx" and result["redirect_hop_count"] == 1
    assert result["raw_response_body_stored"] is False
    assert "Location" not in str(result) and "done" not in str(result)


def test_role_container_identity_is_fresh_and_attestation_is_name_bound() -> None:
    case_ref = "a" * 64
    names = {role_container_name(seed=33101, case_ref_sha256=case_ref, role=role) for role in ("candidate", "reference", "negative", "replay")}
    assert len(names) == 4
    candidate = role_container_name(seed=33101, case_ref_sha256=case_ref, role="candidate")
    assert candidate in build_container_command(container_name=candidate)
    assert attest_container_inspection(_inspection(Name="/" + candidate), expected_name=candidate)["valid"] is True
    assert attest_container_inspection(_inspection(Name="/" + candidate), expected_name=CONTAINER_NAME)["valid"] is False
    assert build_relay_contract(container_name=candidate)["container_name"] == candidate
    for unsafe in ("", "../../container", "pg331-vapp-nn-33101-not-a-digest-candidate", "pg331-vapp-nn-33101-aaaaaaaaaaaaaaaa-other"):
        with pytest.raises(ValueError):
            build_container_command(container_name=unsafe)
        with pytest.raises(ValueError):
            build_relay_contract(container_name=unsafe)
