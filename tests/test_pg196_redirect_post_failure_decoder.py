import json
from pathlib import Path

from app.pg196_failure_action_decoder import guarded_action, guarded_metrics, enumerate_rows


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "research" / "pg196_redirect_post_failure_decoder_report_v1.json"
TRACE = ROOT / "research" / "pg196_redirect_post_failure_decoder_trace_v1.json"
PROTOCOL = ROOT / "research" / "pg196_redirect_post_failure_decoder_protocol_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_pg196_guard_is_fail_closed_for_unknown_post_and_allows_typed_candidate() -> None:
    assert guarded_action(method="POST", redirect_hops=1, status_class="4xx", candidate_signal=1, typed_available=0, negative_control=1, budget_remaining=1, failure_kind="post_validation") == "abstain"
    assert guarded_action(method="GET", redirect_hops=1, status_class="3xx", candidate_signal=1, typed_available=1, negative_control=1, budget_remaining=1, failure_kind="redirect_chain") == "safe_candidate"
    assert guarded_action(method="GET", redirect_hops=0, status_class="2xx", candidate_signal=0, typed_available=1, negative_control=0, budget_remaining=3, failure_kind="no_effect") == "baseline"


def test_pg196_decoder_holdout_keeps_raw_and_guarded_metrics_separate() -> None:
    report = _load(REPORT)
    protocol = _load(PROTOCOL)
    assert report["status"] == "completed_redirect_chain_and_post_failure_ood_replay"
    assert report["model"]["variant"] == "xxl"
    assert report["model"]["base_parameter_count"] > 100_000_000
    assert report["decoder_training"]["train_rows"] == 1080
    assert report["decoder_training"]["balanced_train_rows"] == 2048
    raw = report["decoder_training"]["holdout"]
    guarded = report["decoder_training"]["guarded_holdout"]
    assert raw["unsafe_allow_count"] > 0
    assert guarded["accuracy"] == 1.0
    assert guarded["safe_candidate_recall"] == 1.0
    assert guarded["abstain_recall"] == 1.0
    assert guarded["unsafe_allow_count"] == 0
    assert protocol["raw_decoder_action_is_non_authoritative"] is True
    assert protocol["guarded_action_required_before_send"] is True


def test_pg196_redirect_chain_is_typed_and_only_loopback_controlled() -> None:
    report = _load(REPORT)
    redirect_runs = [row for row in report["runs"] if row["family"] == "url_redirect"]
    assert len(redirect_runs) == 3
    assert all(row["fresh_container"] for row in redirect_runs)
    assert all(row["typed_oracle_available"] for row in redirect_runs)
    assert all(row["confirmed_positive"] is True for row in redirect_runs)
    assert all(row["vulnerability_claim_allowed"] is True for row in redirect_runs)
    for row in redirect_runs:
        candidate = next(step for step in row["steps"] if step["role"] == "candidate")
        assert candidate["typed_oracle"]["oracle_id"] == "pg52-url-redirect-loopback-v1"
        assert candidate["typed_oracle"]["positive"] is True
        assert candidate["response_projection"]["status_chain"] == [302, 204]
        assert candidate["response_projection"]["redirect_hop_count"] == 1
        assert candidate["controller_decision"] == "send_typed_redirect_candidate"
        assert candidate["guarded_action"] == "safe_candidate"
    assert report["counts"]["redirect_positive_count"] == 3
    assert report["counts"]["redirect_chain_complete_count"] == 6


def test_pg196_post_failure_trace_abstains_without_target_typed_oracle() -> None:
    report = _load(REPORT)
    post_runs = [row for row in report["runs"] if row["method"] == "POST"]
    assert len(post_runs) == 3
    assert all(row["fresh_container"] for row in post_runs)
    assert all(row["typed_oracle_available"] is False for row in post_runs)
    assert all(row["post_failure_signature_observed"] is True for row in post_runs)
    assert all(row["observed_field_names"] == ["password", "submit", "username"] for row in post_runs)
    assert all(row["replay_field_names"] == ["submit", "username"] for row in post_runs)
    for row in post_runs:
        final = row["steps"][-1]
        assert final["controller_decision"] == "abstain_unknown_oracle"
        assert final["abstain_reason"] == "post_typed_oracle_unavailable"
        assert final["guarded_action"] == "abstain"
        assert final["typed_oracle"]["positive"] is False
    assert report["counts"]["post_failure_signature_count"] == 3
    assert report["counts"]["post_unknown_abstain_count"] == 3


def test_pg196_promotion_and_raw_material_remain_quarantined() -> None:
    report = _load(REPORT)
    trace = _load(TRACE)
    protocol = _load(PROTOCOL)
    rules = _load(ROOT / "research" / "improvement_rules.json")
    assert report["counts"]["guarded_unsafe_allow_count"] == 0
    assert report["counts"]["false_positive_count"] == 0
    assert report["promotion"]["training_eligible"] is False
    assert report["promotion"]["memory_promotion_allowed"] is False
    assert report["promotion"]["vulnerability_claim_allowed"] is False
    assert report["promotion"]["raw_payload_strings_stored"] is False
    assert report["promotion"]["raw_response_bodies_stored"] is False
    assert trace["training_eligible"] is False
    assert trace["memory_promotion_allowed"] is False
    assert trace["raw_payload_strings_stored"] is False
    assert trace["raw_response_bodies_stored"] is False
    assert protocol["raw_payload_and_response_excluded"] is True
    rule = rules["pg196_redirect_post_failure_decoder"]
    assert rule["raw_decoder_is_non_authoritative"] is True
    assert rule["guard_required_before_send"] is True
    assert rule["controlled_redirect_positive_count"] == 3
    assert rule["post_unknown_abstain_count"] == 3
    assert rule["guarded_holdout"]["unsafe_allow_count"] == 0
    assert rule["training_promotion_allowed"] is False
    assert rule["memory_promotion_allowed"] is False
    assert rule["vulnerability_claim_allowed"] is False


def test_pg196_guarded_contract_is_complete_on_its_abstract_holdout() -> None:
    _train, holdout = enumerate_rows()
    result = guarded_metrics(holdout)
    assert result["accuracy"] == 1.0
    assert result["safe_candidate_recall"] == 1.0
    assert result["abstain_recall"] == 1.0
    assert result["unsafe_allow_count"] == 0
