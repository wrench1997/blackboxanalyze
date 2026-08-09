from app.experiment_engineering_triage import triage_failure


def test_triage_keeps_experiment_and_engineering_paths_separate():
    result = triage_failure(
        experiment_signals=["family_holdout_regression"],
        engineering_signals=["data_hash_or_lineage_mismatch"],
        experiment_gate_passed=False,
        engineering_gate_passed=False,
    )
    assert result["classification"] == "mixed"
    assert result["experiment_path"]["unregistered_signals"] == []
    assert result["engineering_path"]["unregistered_signals"] == []
    assert result["model_change_authorized"] is False
    assert result["infrastructure_scale_authorized"] is False


def test_triage_marks_clean_run_inconclusive_without_failure_signal():
    result = triage_failure(experiment_gate_passed=True, engineering_gate_passed=True)
    assert result["classification"] == "inconclusive"
    assert result["experiment_path"]["signals"] == []
    assert result["engineering_path"]["signals"] == []


def test_triage_preserves_unregistered_experiment_signal_without_authorizing_model_change():
    result = triage_failure(
        experiment_signals=["sql_cross_source_promotion_regression"],
        experiment_gate_passed=False,
        engineering_gate_passed=True,
    )
    assert result["classification"] == "experiment_problem"
    assert result["experiment_path"]["signals"] == ["sql_cross_source_promotion_regression"]
    assert result["experiment_path"]["unregistered_signals"] == ["sql_cross_source_promotion_regression"]
    assert result["model_change_authorized"] is False


def test_triage_preserves_unregistered_engineering_signal_without_authorizing_scale():
    result = triage_failure(
        engineering_signals=["missing_joint_regression_artifact"],
        experiment_gate_passed=True,
        engineering_gate_passed=False,
    )
    assert result["classification"] == "engineering_capability_problem"
    assert result["engineering_path"]["signals"] == ["missing_joint_regression_artifact"]
    assert result["engineering_path"]["unregistered_signals"] == ["missing_joint_regression_artifact"]
    assert result["infrastructure_scale_authorized"] is False
