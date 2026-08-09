from __future__ import annotations

import pytest

from app.pg333_webgoat_docker_relay import IMAGE, build_container_command, container_name


def test_webgoat_container_identity_and_network_none_contract():
    name = container_name(seed=33340, route_ref_sha256="a" * 64, role="candidate")
    command = build_container_command(name=name, seed=33340, role="candidate")
    assert name == "pg333-webgoat-nn-33340-aaaaaaaaaaaaaaaa-candidate"
    assert "--network" in command and command[command.index("--network") + 1] == "none"
    assert "--publish" not in command and "-p" not in command
    assert "--volume" not in command and "-v" not in command
    assert command[-1] == IMAGE


def test_webgoat_container_names_are_role_bound():
    with pytest.raises(ValueError):
        build_container_command(name="pg333-webgoat-nn-33340-aaaaaaaaaaaaaaaa-candidate", seed=33340, role="reference")
    with pytest.raises(ValueError):
        build_container_command(name="pg333-webgoat-nn-33340-aaaaaaaaaaaaaaaa-candidate", seed=33340, role="candidate;rm")
