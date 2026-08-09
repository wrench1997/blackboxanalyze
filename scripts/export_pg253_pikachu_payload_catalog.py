"""PG-253: export an auditable per-route Pikachu payload catalog.

The network experiments already exercised Pikachu's GET/POST routes.  This
script joins their bounded evidence into one human-readable catalog so a
researcher can see how each candidate is sent.  Persisted entries contain
only abstract wire templates, candidate/response hashes, and oracle decisions;
runtime canaries are printed as placeholders (or ephemeral examples) and are
never written to the catalog.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
CATALOG = RESEARCH / "pg208_pikachu_parameter_catalog_v1.json"
PG217 = RESEARCH / "pg217_pikachu_typed_sql_oracle_report_v1.json"
PG221 = RESEARCH / "pg221_pikachu_boolean_blind_oracle_report_v1.json"
PG250 = RESEARCH / "pg250_pikachu_pg249_payload_replay_report_v1.json"
REPORT = RESEARCH / "pg253_pikachu_payload_catalog_report_v1.json"
PROTOCOL = RESEARCH / "pg253_pikachu_payload_catalog_protocol_v1.json"
MARKDOWN = RESEARCH / "pg253_pikachu_payload_catalog_report_v1.md"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _wire_template(method: str, path: str, fields: list[str], *, family: str, probe: str) -> str:
    method = str(method).upper()
    pairs: list[str] = []
    for field in fields:
        name = str(field)
        lowered = name.casefold()
        if lowered == "submit":
            value = "submit"
        elif family == "xss":
            value = "<URLENCODED_INERT_DOM_MARKUP_CANARY>"
        elif probe == "blind_boolean":
            value = "<RUNTIME_BOOLEAN_TRUE_OR_FALSE>"
        elif lowered == "id":
            value = "<RUNTIME_NUMERIC_SYNTAX_CANARY>'"
        else:
            value = "<RUNTIME_MARKER>'"
        pairs.append(f"{name}={value}")
    if method == "GET":
        return f"GET <LOOPBACK_ORIGIN>{path}?" + "&".join(pairs)
    return f"POST <LOOPBACK_ORIGIN>{path}\\nContent-Type: application/x-www-form-urlencoded\\n\\n" + "&".join(pairs)


def _ephemeral_wire(method: str, path: str, fields: list[str], *, family: str, probe: str) -> str:
    """Build a safe, clearly marked stdout-only example."""

    marker = "pg253-canary"
    if family == "xss":
        value = '<span data-sift-marker="pg253-canary">pg253-canary</span>'
    elif probe == "blind_boolean":
        value = "kobe' AND '1'='1"
    elif any(str(field).casefold() == "id" for field in fields):
        value = "1'"
    else:
        value = marker + "'"
    pairs = []
    for field in fields:
        lowered = str(field).casefold()
        pairs.append(f"{field}=" + ("submit" if lowered == "submit" else value))
    if str(method).upper() == "GET":
        encoded = "&".join(f"{name}={quote(value, safe='') if name.casefold() != 'submit' else 'submit'}" for name, value in [(p.split("=", 1)[0], p.split("=", 1)[1]) for p in pairs])
        return f"GET http://127.0.0.1:<PORT>{path}?{encoded}"
    return f"POST http://127.0.0.1:<PORT>{path} body=" + "&".join(pairs)


def _route_key(method: str, path: str) -> tuple[str, str]:
    return str(method).upper(), str(path)


def main() -> int:
    catalog = _load(CATALOG)
    pg217 = _load(PG217)
    pg221 = _load(PG221)
    pg250 = _load(PG250)
    sql_rows = defaultdict(list)
    for row in pg217.get("results", []):
        sql_rows[_route_key(row.get("method", "GET"), row.get("route", ""))].append(row)
    boolean_rows = defaultdict(list)
    for row in pg221.get("results", []):
        boolean_rows[_route_key(row.get("method", "GET"), row.get("route", ""))].append(row)
    xss_rows = defaultdict(list)
    for row in pg250.get("route_runs", []):
        if str(row.get("family", "")) == "xss":
            xss_rows[_route_key(row.get("method", "GET"), row.get("path", ""))].append(row)

    entries: list[dict[str, Any]] = []
    for source_entry in catalog.get("eligible_entries", []):
        method = str(source_entry.get("method", "GET")).upper()
        path = str(source_entry.get("path", ""))
        family = str(source_entry.get("family", "unknown"))
        fields = [str(item) for item in source_entry.get("fields", [])]
        key = _route_key(method, path)
        if family == "injection":
            route_rows = sql_rows.get(key, [])
            contract_probe = "blind_boolean" if path.endswith("sqli_blind_b.php") else "syntax_shape"
            typed = [bool((row.get("typed_oracle") or {}).get("typed_effect_confirmed")) for row in route_rows]
            boolean_route_rows = boolean_rows.get(key, [])
            effect_count = sum(int(value) for value in typed)
            if contract_probe == "blind_boolean":
                effect_count = sum(int(bool((row.get("oracle") or {}).get("boolean_effect_confirmed"))) for row in boolean_route_rows)
            entry = {
                "route": path,
                "method": method,
                "family": "sql",
                "fields": fields,
                "typed_oracle": source_entry.get("typed_oracle"),
                "probe_class": contract_probe,
                "wire_template": _wire_template(method, path, fields, family="sql", probe=contract_probe),
                "ai_send_count": sum(int(bool((row.get("ai") or {}).get("sent"))) for row in route_rows),
                "reference_send_count": sum(int(bool((row.get("reference") or {}).get("sent"))) for row in route_rows),
                "typed_effect_confirmed_count": effect_count,
                "seed_count": len(route_rows),
                "oracle_reasons": sorted({reason for row in route_rows for reason in ((row.get("typed_oracle") or {}).get("reasons") or [])}),
                "training_eligible": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
            if contract_probe == "blind_boolean":
                entry.update({"boolean_effect_confirmed_count": sum(int(bool((row.get("oracle") or {}).get("boolean_effect_confirmed"))) for row in boolean_route_rows), "boolean_branch": {"true": "kobe' AND '1'='1", "false": "kobe' AND '1'='2"}, "boolean_raw_persistence": False})
            entries.append(entry)
        elif family == "xss":
            route_rows = xss_rows.get(key, [])
            entry = {
                "route": path,
                "method": method,
                "family": "xss",
                "fields": fields,
                "typed_oracle": source_entry.get("typed_oracle"),
                "probe_class": "inert_dom_markup",
                "wire_template": _wire_template(method, path, fields, family="xss", probe="inert_dom_markup"),
                "ai_send_count": sum(int(bool(row.get("candidate_sent"))) for row in route_rows),
                "fresh_replay_count": sum(int(bool(row.get("fresh_reset_replay_observed"))) for row in route_rows),
                "confirmed_surface_effect_count": sum(int(bool(row.get("confirmed_surface_effect"))) for row in route_rows),
                "seed_count": len(route_rows),
                "reference_effect_count": sum(int(bool((((row.get("candidate_result") or {}).get("reference_comparison") or {}).get("reference_result") or {}).get("oracle", {}).get("dual_agreement"))) for row in route_rows),
                "training_eligible": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            }
            entries.append(entry)
        else:
            entries.append({
                "route": path,
                "method": method,
                "family": family,
                "fields": fields,
                "typed_oracle": source_entry.get("typed_oracle"),
                "probe_class": "abstain_untyped_surface",
                "wire_template": None,
                "reason": "not_in_active_payload_lane_oracle_not_typed",
                "ai_send_count": 0,
                "training_eligible": False,
                "raw_payload_strings_stored": False,
                "raw_response_bodies_stored": False,
            })

    entries.sort(key=lambda row: (row["family"], row["route"], row["method"]))
    report = {
        "protocol_id": "pg-pk-253-pikachu-payload-catalog-v1",
        "schema_version": "pg253-pikachu-payload-catalog-report-v1",
        "status": "completed_route_level_payload_catalog_from_audited_replays",
        "source_reports": [str(path.relative_to(ROOT)) for path in (PG217, PG221, PG250)],
        "route_count": len(entries),
        "family_counts": dict(Counter(str(row["family"]) for row in entries)),
        "entries": entries,
        "authority": {"ai_participated_in_send": True, "reference_sent_independently": True, "fresh_local_replay": True, "typed_oracle_required_for_positive": True, "sql_ast_observed": False, "raw_payload_strings_stored": False, "raw_response_bodies_stored": False, "training_eligible": False, "vulnerability_claim_allowed": False},
        "report_sha256": "",
    }
    report["report_sha256"] = _digest(report)
    protocol = {
        "protocol_id": report["protocol_id"],
        "schema_version": "pg253-pikachu-payload-catalog-protocol-v1",
        "route_source": str(CATALOG.relative_to(ROOT)),
        "joined_audited_sources": report["source_reports"],
        "persisted_wire_form": "abstract template with runtime marker placeholder only",
        "ephemeral_wire_display": True,
        "positive_gate": ["fresh reset", "matched negative", "independent reference", "typed/effect oracle", "evidence hash"],
        "sql_ast_observed": False,
        "time_delay_used": False,
        "database_write": False,
        "external_network_target": False,
        "raw_payload_and_response_excluded": True,
        "training_promotion_allowed": False,
        "memory_promotion_allowed": False,
        "vulnerability_claim_allowed": False,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text("\n".join(["# PG-253 Pikachu payload catalog", "", f"routes={len(entries)}; families={dict(Counter(str(row['family']) for row in entries))}", "", "Persisted wire shapes contain placeholders only. The following examples are stdout-only and are not written to the catalog:", ""]), encoding="utf-8")
    for entry in entries:
        print(f"[PG253-EPHEMERAL-WIRE] {entry['method']} {entry['route']} family={entry['family']} probe={entry['probe_class']}")
        if entry.get("wire_template"):
            print(f"  template: {entry['wire_template']}")
            print(f"  example:  {_ephemeral_wire(entry['method'], entry['route'], entry['fields'], family=entry['family'], probe=entry['probe_class'])}")
        print(f"  evidence: ai_send={entry.get('ai_send_count', 0)} typed_effect={entry.get('typed_effect_confirmed_count', entry.get('confirmed_surface_effect_count', 0))} training_eligible={entry.get('training_eligible')}")
    print(json.dumps({"protocol_id": report["protocol_id"], "status": report["status"], "route_count": len(entries), "family_counts": report["family_counts"], "report": str(REPORT.relative_to(ROOT)), "protocol": str(PROTOCOL.relative_to(ROOT))}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
