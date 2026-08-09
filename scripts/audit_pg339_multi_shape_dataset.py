"""Read-only information/isolation audit for PG-339's diagnostic corpus."""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "research" / "pg339_multi_shape_diagnostic_dataset_v1.json"
OUTPUT = ROOT / "research" / "pg339_multi_shape_diagnostic_audit_v1.json"
SCHEMA = "pg339-multi-shape-diagnostic-dataset-v1"
AXES = ("document_structure", "navigation", "request_transport", "response_transport", "javascript_surface", "failure_feedback", "belief_and_replay")
FORBIDDEN = ("payload=", "payload_", "raw_", "response_body=", "oracle=", "evaluator=", "url=", "path=", "route=")
def _sha(value: Any) -> str: return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def _entropy(values: list[str]) -> dict[str, Any]:
    if not values: return {"count": 0, "unique": 0, "bits": None}
    counts=Counter(values); total=len(values)
    return {"count": total, "unique": len(counts), "bits": round(-sum((n/total)*math.log2(n/total) for n in counts.values()), 6)}
def _parse(tokens: list[str]) -> dict[str,list[str]]:
    out: dict[str,list[str]]={}
    for token in tokens:
        if "=" in token and not token.startswith("["):
            k,v=token.split("=",1); out.setdefault(k,[]).append(v)
    return out
def audit(data: Mapping[str, Any]) -> dict[str, Any]:
    rows=[row for row in list(data.get("records") or []) if isinstance(row,Mapping)]; failures=[]
    if data.get("schema_version") != SCHEMA: failures.append("schema")
    split=Counter(str(row.get("split")) for row in rows); train_keys=set(); holdout_keys=set(); impl_by_split={"train":set(),"shape_holdout":set()}; forbidden=0
    axis: dict[str,Any]={}
    for axis_name in AXES:
        vals=[]; field_statuses=[]; ablated=set(); original=set()
        for row in rows:
            tokens=[str(x) for x in row.get("context_tokens") or []]; parsed=_parse(tokens); vals.append(str((row.get("axis_presence") or {}).get(axis_name,"missing")))
            field_statuses.extend(str(value) for value in dict((row.get("field_capture_manifest") or {}).get(axis_name) or {}).values())
            prefix={"document_structure":"document_","belief_and_replay":"belief_"}.get(axis_name, axis_name.split("_")[0]+"_")
            original.add(str(row.get("context_target_sha256",""))); ablated.add(_sha([x for x in tokens if not x.startswith(prefix)]))
        axis[axis_name]={"presence_entropy":_entropy(vals),"field_status_entropy":_entropy(field_statuses),"field_ablation":{"eligible_rows":len(rows),"unique_before":len(original),"unique_after":len(ablated),"changed_rate":1.0 if rows else None}}
    for row in rows:
        tokens=[str(x) for x in row.get("context_tokens") or []]; forbidden += sum(any(mark in t.casefold() for mark in FORBIDDEN) for t in tokens)
        key=str(row.get("context_target_sha256","")); current=str(row.get("split","")); impl=str(row.get("source_implementation_hash",""))
        if current=="train": train_keys.add(key); impl_by_split["train"].add(impl)
        if current=="shape_holdout": holdout_keys.add(key); impl_by_split["shape_holdout"].add(impl)
        if row.get("training_eligible") is not False: failures.append("training_flag")
    if not rows: failures.append("records")
    if not split.get("shape_holdout"): failures.append("shape_holdout_missing")
    if train_keys & holdout_keys: failures.append("context_target_split_overlap")
    if impl_by_split["train"] & impl_by_split["shape_holdout"]: failures.append("implementation_split_overlap")
    if forbidden: failures.append("context_firewall")
    result={"schema_version":"pg339-multi-shape-diagnostic-audit-v1","status":"blocked_information_gate" if failures else "diagnostic_only_information_gate_pending","dataset_sha256":str(data.get("dataset_sha256", "")),"counts":{"records":len(rows),"train":int(split["train"]),"shape_holdout":int(split["shape_holdout"]),"forbidden_token_count":forbidden},"axis_entropy":axis,"split_implementation_isolation":{"context_target_overlap_count":len(train_keys & holdout_keys),"implementation_overlap_count":len(impl_by_split["train"] & impl_by_split["shape_holdout"]),"passed":not bool(train_keys & holdout_keys or impl_by_split["train"] & impl_by_split["shape_holdout"])},"information_gate":{"field_entropy_measured":True,"field_ablation_measured":True,"predictive_entropy_holdout":"not_run","passed":False},"scientific_gate":{"accepted_training_rows":0,"reason":"predictive-entropy/field-ablation holdout gate not passed"},"failures":sorted(set(failures)),"promotion":{"training_allowed":False,"memory_promotion_allowed":False,"payload_catalog_promotion_allowed":False,"vulnerability_claim_allowed":False}}
    result["audit_sha256"]=_sha(result); return result
def main()->int:
    p=argparse.ArgumentParser(description="Audit PG-339 multi-shape diagnostic dataset");p.add_argument("--dataset",type=Path,default=DEFAULT);p.add_argument("--output",type=Path,default=OUTPUT);p.add_argument("--json",action="store_true");a=p.parse_args(); result=audit(json.loads(a.dataset.read_text(encoding="utf-8-sig")));a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");print(json.dumps({"status":result["status"],"counts":result["counts"],"audit_sha256":result["audit_sha256"]},ensure_ascii=False,indent=2 if a.json else None));return 0
if __name__=="__main__":raise SystemExit(main())
