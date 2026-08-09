from __future__ import annotations

from typing import Any, Dict, List


MATURITY_REQUIREMENTS = {
    "minimum_independent_seeds": 3,
    "minimum_family_holdout_runs": 1,
}


def evaluate_research_maturity(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the research-to-engineering gate without subjective overrides."""
    checks = {
        "reproducible": bool(evidence.get("reproducible", False)),
        "independent_seeds": int(evidence.get("independent_seeds", 0))
        >= MATURITY_REQUIREMENTS["minimum_independent_seeds"],
        "family_holdout_runs": int(evidence.get("family_holdout_runs", 0))
        >= MATURITY_REQUIREMENTS["minimum_family_holdout_runs"],
        "ablation_supports_mechanism": bool(evidence.get("ablation_supports_mechanism", False)),
        "preregistered_target_met": bool(evidence.get("preregistered_target_met", False)),
        "guardrails_passed": bool(evidence.get("guardrails_passed", False)),
        "data_lineage_complete": bool(evidence.get("data_lineage_complete", False)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    ready = not failed
    return {
        "state": "ready_for_engineering_scale" if ready else "research_only",
        "ready": ready,
        "checks": checks,
        "failed_checks": failed,
        "next_action": (
            "Freeze the research claim and begin a versioned engineering-scale run."
            if ready
            else "Keep the run small and resolve every failed gate before scaling data or compute."
        ),
    }


def triage_scale_failure(signals: Dict[str, bool]) -> Dict[str, Any]:
    """Separate scientific failure signals from scale/infrastructure failures."""
    experiment_weights = {
        "failure_reproduces_at_small_scale": 3,
        "seed_or_split_sensitive": 1,
        "family_holdout_regression": 2,
        "metric_or_oracle_definition_changed": 2,
    }
    engineering_weights = {
        "single_node_passes_but_distributed_fails": 3,
        "data_hash_or_lineage_mismatch": 3,
        "oom_timeout_or_checkpoint_failure": 2,
        "throughput_or_io_regression": 1,
        "nondeterministic_pipeline": 2,
    }

    experiment_evidence = [name for name in experiment_weights if bool(signals.get(name, False))]
    engineering_evidence = [name for name in engineering_weights if bool(signals.get(name, False))]
    experiment_score = sum(experiment_weights[name] for name in experiment_evidence)
    engineering_score = sum(engineering_weights[name] for name in engineering_evidence)

    if experiment_score == 0 and engineering_score == 0:
        classification = "inconclusive"
    elif experiment_score > 0 and engineering_score > 0:
        classification = "mixed"
    elif experiment_score > 0:
        classification = "experiment_problem"
    else:
        classification = "engineering_capability_problem"

    next_checks: List[str]
    if classification == "experiment_problem":
        next_checks = [
            "Reproduce with the frozen single-node pipeline and multiple seeds.",
            "Repeat the preregistered ablation and family-holdout evaluation.",
        ]
    elif classification == "engineering_capability_problem":
        next_checks = [
            "Compare sample hashes and outputs between single-node and scaled runs.",
            "Profile data loading, distributed synchronization, checkpoints, memory, and I/O.",
        ]
    elif classification == "mixed":
        next_checks = [
            "Open separate experiment and engineering incidents with independent owners.",
            "Repair and accept each path independently before combining changes.",
        ]
    else:
        next_checks = [
            "Create a minimal failing run and a single-node versus distributed control.",
            "Capture data hashes, environment fingerprints, seeds, logs, and metric definitions.",
        ]

    return {
        "classification": classification,
        "experiment_score": experiment_score,
        "engineering_score": engineering_score,
        "experiment_evidence": experiment_evidence,
        "engineering_evidence": engineering_evidence,
        "next_checks": next_checks,
        "policy": (
            "Do not scale infrastructure to hide a failed experiment; do not change the scientific "
            "hypothesis to hide an engineering failure."
        ),
    }
