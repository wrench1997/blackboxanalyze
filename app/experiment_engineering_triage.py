"""Independent experiment-vs-engineering failure triage.

The triage layer never changes a model or silently relaxes a gate.  It takes
already materialized metrics and infrastructure observations and returns two
separate acceptance decisions, so an engineering failure cannot be mistaken
for a scientific improvement (or vice versa).
"""

from __future__ import annotations

from typing import Any, Iterable


TRIAGE_SCHEMA = "sift-experiment-engineering-triage-v1"
EXPERIMENT_SIGNALS = (
    "failure_reproduces_at_small_scale",
    "seed_or_split_sensitive",
    "family_holdout_regression",
    "metric_or_oracle_definition_changed",
)
ENGINEERING_SIGNALS = (
    "single_node_passes_but_distributed_fails",
    "data_hash_or_lineage_mismatch",
    "oom_timeout_or_checkpoint_failure",
    "throughput_or_io_regression",
    "nondeterministic_pipeline",
)


def triage_failure(
    *,
    experiment_signals: Iterable[str] = (),
    engineering_signals: Iterable[str] = (),
    experiment_gate_passed: bool | None = None,
    engineering_gate_passed: bool | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify a run while preserving independent scientific/engineering gates."""

    # The caller has already selected the scientific or engineering path.  Keep
    # every non-empty signal on that path: silently filtering a domain-specific
    # name would turn a real failure into a false ``inconclusive`` result.
    exp = sorted({str(signal).strip() for signal in experiment_signals if str(signal).strip()})
    eng = sorted({str(signal).strip() for signal in engineering_signals if str(signal).strip()})
    exp_unregistered = sorted(set(exp).difference(EXPERIMENT_SIGNALS))
    eng_unregistered = sorted(set(eng).difference(ENGINEERING_SIGNALS))
    if exp and eng:
        classification = "mixed"
    elif exp:
        classification = "experiment_problem"
    elif eng:
        classification = "engineering_capability_problem"
    else:
        classification = "inconclusive"
    return {
        "schema_version": TRIAGE_SCHEMA,
        "classification": classification,
        "experiment_path": {
            "signals": exp,
            "unregistered_signals": exp_unregistered,
            "gate_passed": experiment_gate_passed,
            "action": (
                "register_or_map_signal_before_scientific_repair"
                if exp_unregistered
                else "repair_data_oracle_objective_or_model_only"
                if exp
                else "no_scientific_failure_signal"
            ),
        },
        "engineering_path": {
            "signals": eng,
            "unregistered_signals": eng_unregistered,
            "gate_passed": engineering_gate_passed,
            "action": (
                "register_or_map_signal_before_engineering_repair"
                if eng_unregistered
                else "repair_lineage_runtime_resources_or_reliability"
                if eng
                else "no_engineering_failure_signal"
            ),
        },
        "evidence": dict(evidence or {}),
        "model_change_authorized": classification == "experiment_problem" and not exp_unregistered,
        "infrastructure_scale_authorized": classification == "engineering_capability_problem" and not eng_unregistered,
    }


__all__ = ["ENGINEERING_SIGNALS", "EXPERIMENT_SIGNALS", "TRIAGE_SCHEMA", "triage_failure"]
