from app.promotion_runner import provenance_summary, run_promotion_audit


def _row(dataset, target, seed, rule="xss::surface", accepted=True, oracle=True):
    return {
        "dataset_id": dataset,
        "target_instance_id": target,
        "sampling_seed": seed,
        "rule_key": rule,
        "accepted": accepted,
        "oracle_revalidated": oracle,
        "false_positive": False,
        "evidence_hash": f"{dataset}{target}{seed}".encode().hex().ljust(64, "0")[:64],
        "source_hash": (dataset * 64)[:64],
        "local_only": True,
    }


def test_promotion_runner_reports_provenance_and_replay_queue():
    rows = [_row(f"d{i}", f"t{i}", seed) for i in range(3) for seed in (1, 2)]
    result = run_promotion_audit(rows, rule_keys=["xss::surface"])
    assert result["all_promoted"] is True
    assert result["provenance"]["evidence_hash_count"] == 6
    assert result["provenance"]["source_hashes"] == [(f"d{i}" * 64)[:64] for i in range(3)]
    bad = run_promotion_audit(rows[:1], rule_keys=["xss::surface"])
    assert bad["all_promoted"] is False
    assert bad["replay_queue"]
    assert bad["memory_write_allowed"] is False
