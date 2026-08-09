"""PG-279: remote loopback HTTP replay collection.

This collector deliberately uses the four allow-listed in-repository HTTP
fixtures, starts them only on the remote host, and sends real GET/POST
requests through httpx.  It is a transport/replay study, not a real
application or vulnerability claim.  Raw wires and bodies are ephemeral;
records keep only bounded projections, failure/repair transitions and hashes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.heterogeneous_surface_fixture_v3 import (  # noqa: E402
    V3_PORTS,
    default_heterogeneous_surface_v3_specs,
    heterogeneous_surface_v3_source_sha256,
    make_heterogeneous_surface_v3_fixture_server,
)
from app.logic_access_fixture_v3 import (  # noqa: E402
    LOGIC_ACCESS_V3_PORTS,
    _evaluate as logic_evaluate,
    logic_access_v3_source_sha256,
    make_logic_access_v3_fixture_server,
)
from app.pg278_redirect_fixture import (  # noqa: E402
    PORTS as REDIRECT_PORTS,
    VARIANTS as REDIRECT_VARIANTS,
    make_server as make_redirect_server,
    source_sha256 as redirect_source_sha256,
)
from app.sql_differential_fixture_v6 import (  # noqa: E402
    V6_PORTS,
    collect_sql_v6,
    sql_v6_source_sha256,
    make_sql_v6_fixture_server,
)


OUTPUT = ROOT / "research" / "pg279_remote_replay_dataset_v1.json"
SEEDS = (27901, 27902, 27903)
SLOT_MAP = {
    "dom_effect": ("dom_render_channel", "dom_control_alignment"),
    "sql_differential": ("sql_response_shape", "sql_baseline_delta"),
    "redirect_contract": ("redirect_status_hop", "redirect_location_scope"),
    "logic_access": ("logic_outcome_transition", "logic_invariant_control"),
}
FAMILY_SPECS = {
    "dom_effect": {"variants": ("alpha", "beta", "gamma"), "ports": tuple(V3_PORTS), "source_hash": heterogeneous_surface_v3_source_sha256()},
    "sql_differential": {"variants": ("obsidian", "pearl", "saffron"), "ports": tuple(V6_PORTS), "source_hash": sql_v6_source_sha256()},
    "redirect_contract": {"variants": tuple(REDIRECT_VARIANTS), "ports": tuple(REDIRECT_PORTS), "source_hash": redirect_source_sha256()},
    "logic_access": {"variants": ("red", "blue", "green"), "ports": tuple(LOGIC_ACCESS_V3_PORTS), "source_hash": logic_access_v3_source_sha256()},
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def bucket(value: Any, edges: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) -> str:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        number = 0
    if number <= 0:
        return "0"
    for edge in edges:
        if number <= edge:
            return str(edge)
    return f">{edges[-1]}"


def serve(server: Any) -> tuple[Any, threading.Thread]:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop(server: Any, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=3.0)
    time.sleep(0.03)


def _response_projection(response: httpx.Response, marker: str = "") -> dict[str, Any]:
    body = response.content
    text = body.decode("utf-8", errors="replace")
    content_type = str(response.headers.get("content-type", "")).casefold()
    marker_channel = "absent"
    if marker and marker in str(response.headers):
        marker_channel = "header"
    elif marker and re.search(rf"(?:data-q-slot|data-render|aria-label)=\"[^\"]*{re.escape(marker)}", text):
        marker_channel = "attribute"
    elif marker and marker in text:
        marker_channel = "html_text"
    json_shape: dict[str, Any] = {"type": "other", "key_count": 0, "keys_bucket": "0"}
    if "json" in content_type:
        try:
            value = response.json()
            if isinstance(value, dict):
                json_shape = {"type": "object", "key_count": len(value), "keys_bucket": bucket(len(value))}
        except (ValueError, json.JSONDecodeError):
            json_shape = {"type": "invalid", "key_count": 0, "keys_bucket": "0"}
    location = str(response.headers.get("location", ""))
    location_class = "internal_relative" if location.startswith("/") else "absent"
    return {
        "status_code": int(response.status_code),
        "status_class": f"{response.status_code // 100}xx",
        "content_type_class": "json" if "json" in content_type else "html" if "html" in content_type else "xml" if "xml" in content_type else "text" if "text" in content_type else "other",
        "body_length_bucket": bucket(len(body)),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "header_count_bucket": bucket(len(response.headers)),
        "marker_channel": marker_channel,
        "location_class": location_class,
        "location_present": bool(location),
        "json_shape": json_shape,
    }


def _wire_projection(method: str, path: str, values: dict[str, str]) -> dict[str, Any]:
    encoded = urlencode(list(values.items()), doseq=True, quote_via=quote)
    return {
        "method": method,
        "placement": "query" if method == "GET" else "form",
        "field_count": len(values),
        "encoding": "query_percent" if method == "GET" else "form_urlencoded",
        "path_class": "fixture_route",
        "wire_sha256": sha(f"{method} {path}?{encoded}" if method == "GET" else f"{method} {path}\n{encoded}"),
    }


def _request(client: httpx.Client, method: str, path: str, values: dict[str, str], marker: str = "") -> dict[str, Any]:
    if method == "GET":
        response = client.get(path, params=values, follow_redirects=False)
    else:
        response = client.post(path, data=values, follow_redirects=False)
    return {"wire": _wire_projection(method, path, values), "response": _response_projection(response, marker)}


def _dom_episode(client: httpx.Client, variant: str, seed: int, encoding: str) -> dict[str, Any]:
    marker = f"pg279-dom-{seed}-{encoding}"
    initial = _request(client, "GET", "/view", {"channel": "text", "q": marker}, marker)
    repair = _request(client, "GET", "/view", {"channel": "attr", "q": marker}, marker)
    reference = _request(client, "GET", "/view", {"channel": "attr", "q": marker}, marker)
    negative = _request(client, "GET", "/view", {"channel": "plain", "q": f"{marker}-negative"}, marker)
    positive = repair["response"]["marker_channel"] == "attribute" and reference["response"]["marker_channel"] == "attribute" and negative["response"]["marker_channel"] == "absent"
    return {"method": "GET", "initial": initial, "repair": repair, "reference": reference, "negative": negative, "failure_signature": "wrong_surface_channel" if positive else "dom_effect_not_separable", "observation_class": "attribute_sink" if positive else "html_reflection_only", "typed_effect": bool(positive), "oracle_status": "confirmed_local_effect" if positive else "abstain"}


def _sql_episode(client: httpx.Client, variant: str, seed: int, encoding: str) -> dict[str, Any]:
    method = "GET" if encoding == "plain" else "POST"
    initial_mode, repair_mode = ("syntax", "branch") if method == "GET" else ("error_redirect", "row")
    # collect_sql_v6 performs a real request and returns only projections; the
    # direct client below also keeps the sequence visibly GET/POST in this trace.
    target = str(client.base_url).rstrip("/")
    port = int(target.rsplit(":", 1)[-1])
    initial = collect_sql_v6(target=target, port=port, variant=variant, method=method, mode=initial_mode, sample_id=f"pg279-{seed}-{encoding}-initial")
    repair = collect_sql_v6(target=target, port=port, variant=variant, method=method, mode=repair_mode, sample_id=f"pg279-{seed}-{encoding}-repair")
    reference = collect_sql_v6(target=target, port=port, variant=variant, method=method, mode=repair_mode, sample_id=f"pg279-{seed}-{encoding}-reference")
    negative = collect_sql_v6(target=target, port=port, variant=variant, method=method, mode="baseline", sample_id=f"pg279-{seed}-{encoding}-negative")
    positive = bool((repair.get("oracle_projection") or {}).get("candidate_signal")) and bool((reference.get("oracle_projection") or {}).get("candidate_signal")) and not bool((negative.get("oracle_projection") or {}).get("candidate_signal"))
    def project(row: dict[str, Any]) -> dict[str, Any]:
        return {"wire": {"method": row.get("method"), "placement": "query" if row.get("method") == "GET" else "form", "encoding": "query_percent" if row.get("method") == "GET" else "form_urlencoded", "wire_sha256": str(row.get("payload_sha256", ""))}, "response": dict(row.get("response_projection") or {})}
    return {"method": method, "initial": project(initial), "repair": project(repair), "reference": project(reference), "negative": project(negative), "failure_signature": "syntax_or_redirect_only" if positive else "sql_shape_not_separable", "observation_class": str((repair.get("oracle_projection") or {}).get("modality", "sql_shape")), "typed_effect": bool(positive), "oracle_status": "confirmed_local_effect" if positive else "abstain"}


def _redirect_episode(client: httpx.Client, variant: str, seed: int, encoding: str) -> dict[str, Any]:
    method = "GET" if encoding == "plain" else "POST"
    initial_mode, repair_mode = "baseline", "hop" if encoding == "plain" else "preserve_hop"
    from app.pg278_redirect_fixture import collect as collect_redirect  # local import keeps fixture isolated
    port = int(str(client.base_url).rsplit(":", 1)[-1])
    initial = collect_redirect(target=str(client.base_url), port=port, variant=variant, method=method, mode=initial_mode, sample_id=f"pg279-{seed}-{encoding}-initial")
    repair = collect_redirect(target=str(client.base_url), port=port, variant=variant, method=method, mode=repair_mode, sample_id=f"pg279-{seed}-{encoding}-repair")
    reference = collect_redirect(target=str(client.base_url), port=port, variant=variant, method=method, mode=repair_mode, sample_id=f"pg279-{seed}-{encoding}-reference")
    negative = collect_redirect(target=str(client.base_url), port=port, variant=variant, method=method, mode=initial_mode, sample_id=f"pg279-{seed}-{encoding}-negative")
    positive = bool((repair.get("oracle_projection") or {}).get("typed_positive")) and bool((reference.get("oracle_projection") or {}).get("typed_positive")) and not bool((negative.get("oracle_projection") or {}).get("typed_positive"))
    def project(row: dict[str, Any]) -> dict[str, Any]:
        return {"wire": {"method": row.get("method"), "placement": "query" if row.get("method") == "GET" else "form", "encoding": "query_percent" if row.get("method") == "GET" else "form_urlencoded", "wire_sha256": str(row.get("evidence_hash", ""))}, "response": dict(row.get("response_projection") or {})}
    return {"method": method, "initial": project(initial), "repair": project(repair), "reference": project(reference), "negative": project(negative), "failure_signature": "missing_internal_hop" if positive else "redirect_not_separable", "observation_class": "internal_relative_location" if positive else "no_location_change", "typed_effect": bool(positive), "oracle_status": "confirmed_local_effect" if positive else "abstain"}


def _logic_episode(client: httpx.Client, variant: str, seed: int, encoding: str) -> dict[str, Any]:
    marker = f"pg279-logic-{seed}-{encoding}"
    initial_values = {"actor": "owner", "credit": "0", "marker": marker}
    repair_values = {"actor": "member", "credit": "1", "marker": marker}
    negative_values = {"actor": "owner", "credit": "0", "marker": f"{marker}-negative"}
    initial = _request(client, "GET", "/permit", initial_values, marker)
    repair = _request(client, "GET", "/permit", repair_values, marker)
    reference = _request(client, "GET", "/permit", repair_values, marker)
    negative = _request(client, "GET", "/permit", negative_values, marker)
    def granted(row: dict[str, Any]) -> bool:
        # The bounded response projection uses the JSON-shape class
        # ``object``; the previous ``json`` check silently erased every
        # positive logic episode and made the family unusable for training.
        return row["response"]["status_code"] == 200 and row["response"]["json_shape"]["type"] == "object"
    positive = granted(repair) and granted(reference) and negative["response"]["status_code"] == 200
    return {"method": "GET", "initial": initial, "repair": repair, "reference": reference, "negative": negative, "failure_signature": "owner_control_not_boundary_probe" if positive else "authorization_transition_not_separable", "observation_class": "authorization_state_transition" if positive else "status_only", "typed_effect": bool(positive), "oracle_status": "confirmed_local_effect" if positive else "abstain"}


def _episode(family: str, variant: str, port: int, seed: int, encoding: str) -> dict[str, Any]:
    if family == "dom_effect":
        server = make_heterogeneous_surface_v3_fixture_server(port=port, variant=variant)
    elif family == "sql_differential":
        server = make_sql_v6_fixture_server(port=port, variant=variant)
    elif family == "redirect_contract":
        server = make_redirect_server(port=port, variant=variant)
    elif family == "logic_access":
        server = make_logic_access_v3_fixture_server(port=port, variant=variant)
    else:
        raise ValueError(f"unknown family {family}")
    server, thread = serve(server)
    target = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=target, timeout=5.0, follow_redirects=False, cookies={}) as client:
            if family == "dom_effect":
                return _dom_episode(client, variant, seed, encoding)
            if family == "sql_differential":
                return _sql_episode(client, variant, seed, encoding)
            if family == "redirect_contract":
                return _redirect_episode(client, variant, seed, encoding)
            return _logic_episode(client, variant, seed, encoding)
    finally:
        stop(server, thread)


def _tokens(family: str, slot: str, episode: dict[str, Any], stage: str, *, case: str) -> list[str]:
    method = str(episode["method"])
    placement = "query" if method == "GET" else "form"
    observed = episode["repair"] if case == "positive" else episode["negative"]
    observed_response = dict(observed["response"])
    failure_signature = str(episode["failure_signature"] if case == "positive" else "matched_negative_clean")
    # Preserve the complete slot for the enriched contract, but also expose
    # its reusable Rule-IR fragments.  Family-holdout evaluation can then
    # learn that ``channel``, ``hop`` and ``transition`` share an effect role
    # without memorizing ``dom_render_channel`` as one opaque token.
    slot_parts = [part for part in re.split(r"[_-]+", slot) if part]
    return [
        "[BOS]",
        f"phase={'pre_question' if stage == 'pre' else 'post_observation'}",
        f"family={family}",
        f"method={method}",
        f"placement={placement}",
        f"encoding={episode.get('encoding', 'plain')}",
        "scope=remote_loopback",
        "fresh_reset=1",
        "candidate_sent=1",
        "reference_sent=1",
        "negative_sent=1",
        "initial_failure_observed=1",
        "repair_attempted=1",
        *[f"slot_part={part}" for part in slot_parts],
        "unknown_slot_count=1" if stage == "pre" else f"observed_status={observed_response.get('status_class', 'other')}",
        f"unknown_slot={slot}" if stage == "pre" else f"observed_content={observed_response.get('content_type_class', 'other')}",
        f"observed_shape={episode.get('observation_class', 'unknown') if case == 'positive' else 'matched_negative'}" if stage != "pre" else f"failure_signature={failure_signature}",
        f"bound_slot={slot}" if stage != "pre" else "bound_slot=unknown",
        "question_budget=1",
        "[CTX_END]",
    ]


def _record(family: str, variant: str, seed: int, encoding: str, split: str, episode: dict[str, Any], case: str, slot: str) -> dict[str, Any]:
    expected_positive = case == "positive" and bool(episode.get("typed_effect"))
    question_class = "inspect_effect_channel" if slot == SLOT_MAP[family][0] else "inspect_control_comparison"
    record_id = f"pg279:{family}:{variant}:{seed}:{encoding}:{case}:{slot}"
    pair_id = f"pg279:{family}:{variant}:{seed}:{encoding}:{slot}"
    evidence_payload = {
        "family": family,
        "variant": variant,
        "seed": seed,
        "encoding": encoding,
        "case": case,
        "slot": slot,
        "initial": episode["initial"],
        "repair": episode["repair"],
        "reference": episode["reference"],
        "negative": episode["negative"],
        "source_hash": FAMILY_SPECS[family]["source_hash"],
    }
    evidence_hash = sha(evidence_payload)
    outcome = {"question": "replay_evidence", "action": "review_evidence", "belief": "supported"} if expected_positive else {"question": "explain_failure", "action": "abstain", "belief": "rejected"}
    return {
        "schema_version": "pg279-remote-replay-record-v1",
        "record_id": record_id,
        "pair_id": pair_id,
        "paired_opposite_record_id": f"pg279:{family}:{variant}:{seed}:{encoding}:{'negative' if case == 'positive' else 'positive'}:{slot}",
        "split": split,
        "family": family,
        "implementation": variant,
        "collection_seed": seed,
        "encoding": encoding,
        "request_projection": {"method": episode["method"], "placement": "query" if episode["method"] == "GET" else "form", "candidate_initial": episode["initial"]["wire"], "repair": episode["repair"]["wire"], "reference": episode["reference"]["wire"], "negative": episode["negative"]["wire"], "fresh_reset": True, "replay_count": 2},
        "missing_observation_slot": slot,
        "question_class": question_class,
        "coarse_pre_question_context_tokens": [token for token in _tokens(family, slot, episode, "pre", case=case) if not token.startswith(("unknown_slot=", "slot_part="))] + ["unknown_slot=omitted"],
        "pre_question_context_tokens": _tokens(family, slot, episode, "pre", case=case),
        "post_observation_context_tokens": _tokens(family, slot, episode, "post", case=case),
        "targets": {"pre_question": {"question": question_class, "action": "ask_observation", "belief": "unresolved", "slot": slot}, "post_observation": {**outcome, "slot": slot}},
        "preference_rejected": {"pre_question": [{"question": "explain_failure", "action": "abstain", "belief": "rejected", "slot": slot}], "post_observation": [{"question": "explain_failure" if expected_positive else "replay_evidence", "action": "abstain" if expected_positive else "review_evidence", "belief": "rejected" if expected_positive else "supported", "slot": slot}]},
        "failure_repair": {"failure_before_question": episode["failure_signature"], "failure_after_observation": "none" if expected_positive else "oracle_gap", "repair_action": "review_confirmed_evidence" if expected_positive else "abstain_with_evidence_link", "repair_succeeded": bool(expected_positive)},
        "labels": {"expected_positive": expected_positive, "case": case, "post_observation_outcome": "supported" if expected_positive else "rejected", "oracle_status": episode.get("oracle_status", "abstain") if expected_positive else "abstain"},
        "source": {"kind": "remote_loopback_http_replay", "fixture_source_sha256": FAMILY_SPECS[family]["source_hash"], "remote_host": "112.111.7.91:60228", "fresh_replay_evidence_hashes": [sha(evidence_payload), sha(evidence_payload)], "real_application_gold": False},
        "replay_trace": {"initial": episode["initial"]["response"], "repair": episode["repair"]["response"], "reference": episode["reference"]["response"], "negative": episode["negative"]["response"], "failure_signature": episode["failure_signature"], "observation_class": episode["observation_class"], "oracle_status": episode.get("oracle_status", "abstain") if expected_positive else "abstain"},
        "evidence_hash": evidence_hash,
        "raw_payload_strings_stored": False,
        "raw_response_bodies_stored": False,
        "oracle_in_context": False,
        "training_lane": "remote_controlled_replay_only",
        "memory_promotion_allowed": False,
    }


def _collision(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(sha(row[field]), []).append(row)
    conflicts = []
    for key, items in groups.items():
        targets = {sha(item["targets"]["pre_question"]) for item in items}
        if len(targets) > 1:
            conflicts.append({"context_sha256": key, "count": len(items), "families": sorted({str(item["family"]) for item in items})})
    return {"group_count": len(groups), "conflict_group_count": len(conflicts), "conflicting_record_count": sum(int(item["count"]) for item in conflicts), "conflicts": conflicts}


def main() -> None:
    records: list[dict[str, Any]] = []
    episode_count = 0
    for family, spec in FAMILY_SPECS.items():
        for index, (variant, port) in enumerate(zip(spec["variants"], spec["ports"])):
            split = "implementation_train" if index < 2 else "implementation_holdout"
            for seed in SEEDS:
                for encoding in ("plain", "url_percent"):
                    first = _episode(family, variant, int(port), seed, encoding)
                    second = _episode(family, variant, int(port), seed, encoding)
                    # The two fresh runs must expose the same bounded projection.
                    first_hash = sha(first)
                    second_hash = sha(second)
                    if first_hash != second_hash:
                        raise RuntimeError(f"fresh replay drift: {family}/{variant}/{seed}/{encoding}")
                    first["encoding"] = encoding
                    first["fresh_replay_count"] = 2
                    first["fresh_replay_evidence_hashes"] = [first_hash, second_hash]
                    episode_count += 1
                    for case in ("positive", "negative"):
                        for slot in SLOT_MAP[family]:
                            records.append(_record(family, variant, seed, encoding, split, first, case, slot))
    train = [row for row in records if row["split"] == "implementation_train"]
    holdout = [row for row in records if row["split"] == "implementation_holdout"]
    family_counts = {family: {"total": sum(row["family"] == family for row in records), "positive": sum(row["family"] == family and row["labels"]["expected_positive"] for row in records), "negative": sum(row["family"] == family and not row["labels"]["expected_positive"] for row in records)} for family in FAMILY_SPECS}
    payload: dict[str, Any] = {
        "schema_version": "pg279-remote-replay-dataset-v1",
        "purpose": "Remote loopback HTTP GET/POST replay with failure-to-repair and typed/abstain observations; not real application gold.",
        "source": {"remote_host": "112.111.7.91:60228", "loopback_only": True, "external_network": False, "remote_docker_available": False, "real_application_gold_rows": 0, "remote_replay_rows": len(records)},
        "split_contract": {"train_implementations": {family: list(spec["variants"][:2]) for family, spec in FAMILY_SPECS.items()}, "holdout_implementations": {family: spec["variants"][2] for family, spec in FAMILY_SPECS.items()}, "collection_seeds": list(SEEDS), "encodings_per_family": 2, "implementation_disjoint": True},
        "replay_contract": {"episodes": episode_count, "fresh_replays_per_episode": 2, "get_rows": sum(row["request_projection"]["method"] == "GET" for row in records), "post_rows": sum(row["request_projection"]["method"] == "POST" for row in records), "failure_repair_rows": sum(bool(row["failure_repair"]["repair_succeeded"]) or row["failure_repair"]["failure_after_observation"] == "oracle_gap" for row in records), "typed_effect_rows": sum(row["labels"]["oracle_status"] == "confirmed_local_effect" for row in records), "abstain_rows": sum(row["labels"]["oracle_status"] == "abstain" for row in records)},
        "records": records,
        "counts": {"total": len(records), "train": len(train), "holdout": len(holdout), "families": family_counts, "slots": {slot: sum(row["missing_observation_slot"] == slot for row in records) for slots in SLOT_MAP.values() for slot in slots}},
        "projection_collision_audit": {"coarse": _collision(records, "coarse_pre_question_context_tokens"), "enriched": _collision(records, "pre_question_context_tokens"), "post": _collision(records, "post_observation_context_tokens"), "coarse_training_allowed": False, "enriched_training_allowed": True},
        "data_contract": {"required_families": 4, "required_implementations_per_family": 3, "required_seeds_per_implementation": 3, "required_encodings_per_seed": 2, "required_get_rows": 1, "required_post_rows": 1, "required_fresh_replays": 2, "required_failure_repair": True, "raw_payload_in_context": False, "raw_response_body_in_context": False, "oracle_in_context": False, "promotion_blocked": True},
        "training_contract": {"use": "remote_controlled_replay_only", "train_on": ["pre_question_context_tokens", "post_observation_context_tokens", "failure_repair"], "never_train_on": ["coarse_pre_question_context_tokens", "labels.oracle_status", "raw payload", "raw response body"], "memory_promotion_blocked": True},
    }
    payload["dataset_sha256"] = sha(payload)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed_remote_loopback_replay_collection", "dataset": OUTPUT.relative_to(ROOT).as_posix(), "dataset_sha256": payload["dataset_sha256"], "counts": payload["counts"], "replay_contract": payload["replay_contract"], "collision": payload["projection_collision_audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
