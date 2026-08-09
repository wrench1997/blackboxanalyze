#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from app.research_events import emit_event  # noqa: E402
from app.synthetic_curriculum import generate_curriculum  # noqa: E402
from evaluate_url_root_cause import ADAPTED, load_model  # noqa: E402
from train_rule_memory_pilot import PromptDataset, collate, evaluate, records_to_examples  # noqa: E402


DATA_SEEDS = [20261123, 20261211]
MODEL_SEEDS = [20261001, 20261019, 20261107]


def examples_for(records: list[dict[str, Any]], seed: int):
    return records_to_examples(
        records,
        random.Random(seed),
        4,
        8,
        routed_semantic_features=True,
        episode_rule_features=True,
    )


def make_loader(examples) -> DataLoader:
    return DataLoader(PromptDataset(examples, 640), batch_size=64, shuffle=False, collate_fn=collate)


def rename_path(examples, source: str, target: str):
    return [replace(example, prompt=example.prompt.replace(source, target)) for example in examples]


def trace_only(examples, invert: bool):
    transformed = []
    for example in examples:
        mode_and_trace, rest = example.prompt.split("<RULEMEM>", 1)
        query = rest.split("<QUERY>", 1)[1]
        if invert:
            prefix, trace = mode_and_trace.split("<TRACE>", 1)
            trace = re.sub(r":([01])(?=\|input\.|$)", lambda match: f":{1 - int(match.group(1))}", trace)
            mode_and_trace = f"{prefix}<TRACE>{trace}"
        prompt = f"{mode_and_trace}<RULEMEM><QUERY>{query}"
        transformed.append(replace(example, prompt=prompt, label=1 - example.label if invert else example.label))
    return transformed


@torch.inference_mode()
def scheme_breakdown(model, examples, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, list[int]] = {}
    for batch in make_loader(examples):
        predictions = model(batch["tokens"].to(device), batch["lengths"].to(device)).argmax(dim=-1).cpu()
        for example, prediction in zip(batch["examples"], predictions):
            query = example.prompt.split("<QUERY>", 1)[1].split("<ANSWER>", 1)[0]
            match = re.search(r"=u:([^|]+)\|", query)
            scheme = match.group(1) if match else "invalid"
            stats = totals.setdefault(scheme, [0, 0])
            stats[0] += int(int(prediction) == example.label)
            stats[1] += 1
    return {scheme: round(good / total, 6) for scheme, (good, total) in sorted(totals.items())}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {seed: load_model(ADAPTED[seed], device) for seed in MODEL_SEEDS}
    rows = []
    for data_seed in DATA_SEEDS:
        records = generate_curriculum(2700, 20, data_seed)
        target_records = [record for record in records if record["family"] == "url_scheme_downgrade"]
        primitive_records = [record for record in records if record["family"] == "url_hostname_primitive"]
        target = examples_for(target_records, data_seed)
        target_as_endpoint = rename_path(target, "input.origin", "input.endpoint")
        primitive = examples_for(primitive_records, data_seed)
        primitive_as_origin = rename_path(primitive, "input.endpoint", "input.origin")
        target_trace_only = trace_only(target, invert=False)
        target_trace_inverted = trace_only(target, invert=True)
        for model_seed, model in models.items():
            conditions = {
                "target_origin": evaluate(model, make_loader(target), device, max_length=640, rule_memory_gate=False)["accuracy"],
                "target_rewritten_as_endpoint": evaluate(model, make_loader(target_as_endpoint), device, max_length=640, rule_memory_gate=False)["accuracy"],
                "primitive_endpoint": evaluate(model, make_loader(primitive), device, max_length=640, rule_memory_gate=False)["accuracy"],
                "primitive_rewritten_as_origin": evaluate(model, make_loader(primitive_as_origin), device, max_length=640, rule_memory_gate=False)["accuracy"],
                "target_trace_only": evaluate(model, make_loader(target_trace_only), device, max_length=640, rule_memory_gate=False)["accuracy"],
                "target_trace_labels_inverted": evaluate(model, make_loader(target_trace_inverted), device, max_length=640, rule_memory_gate=False)["accuracy"],
            }
            rows.append({
                "data_seed": data_seed,
                "model_seed": model_seed,
                "conditions": conditions,
                "field_rewrite_delta": round(conditions["target_rewritten_as_endpoint"] - conditions["target_origin"], 6),
                "reverse_field_rewrite_delta": round(conditions["primitive_rewritten_as_origin"] - conditions["primitive_endpoint"], 6),
                "scheme_breakdown": scheme_breakdown(model, target, device),
            })

    def mean(key: str) -> float:
        return round(sum(row["conditions"][key] for row in rows) / len(rows), 6)

    field_deltas = [row["field_rewrite_delta"] for row in rows]
    inverted = [row["conditions"]["target_trace_labels_inverted"] for row in rows]
    scheme_means: dict[str, float] = {}
    schemes = sorted({scheme for row in rows for scheme in row["scheme_breakdown"]})
    for scheme in schemes:
        scheme_means[scheme] = round(sum(row["scheme_breakdown"].get(scheme, 0.0) for row in rows) / len(rows), 6)
    report = {
        "schema_version": "sift-neural-url-diagnostic-v1",
        "experiment": "neural-url-generalization-loop-11-diagnostics",
        "status": "completed",
        "scope": "frozen-checkpoint counterfactuals; no training",
        "aggregate": {
            "condition_means": {key: mean(key) for key in rows[0]["conditions"]},
            "field_rewrite_delta_mean": round(sum(field_deltas) / len(field_deltas), 6),
            "field_rewrite_delta_min": min(field_deltas),
            "scheme_accuracy_means": scheme_means,
        },
        "hypothesis_results": {
            "H1_field_path_shortcut": {
                "passes": min(field_deltas) >= 0.10,
                "finding": "field path is a stable shortcut" if min(field_deltas) >= 0.10 else "field rewrite is not a stable primary cause",
            },
            "H2_protocol_novelty": {
                "passes": bool(scheme_means.get("ws", 1.0) + 0.15 <= max(scheme_means.get("http", 0.0), scheme_means.get("ftp", 0.0), scheme_means.get("https", 0.0))),
                "finding": "accuracy stratified by serialized query scheme; explicit port is currently absent from the routed token",
            },
            "H3_episode_binding_failure": {
                "passes": max(inverted) < 0.65,
                "finding": "inverted episode labels are not followed reliably" if max(inverted) < 0.65 else "at least one checkpoint follows inverted episode labels",
            },
        },
        "per_cross_product": rows,
        "decision": "Use supported diagnostics to choose the minimal fixed-size pilot; do not add target-family examples.",
        "lineage": {"preregistration": "research/neural_url_loop_11_preregistration.json"},
    }
    output = ROOT / "research/neural_url_loop_11_diagnostics.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    emit_event(
        actor="neural-url-diagnoser",
        tool="generalization.neural_url.diagnose",
        phase="counterfactual",
        status="complete",
        message=f"Field rewrite delta {report['aggregate']['field_rewrite_delta_mean']:+.2%}; inverted trace mean {mean('target_trace_labels_inverted'):.2%}",
        payload={"aggregate": report["aggregate"], "hypotheses": report["hypothesis_results"]},
        artifact=str(output.relative_to(ROOT)),
    )
    print(json.dumps({"aggregate": report["aggregate"], "hypotheses": report["hypothesis_results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
