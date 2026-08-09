from __future__ import annotations

from scripts.run_pg367_a800_process_candidate import _load_rows, _vocabulary
from scripts.build_pg367_waf_staircase_dataset import build


def test_process_candidate_reads_only_abstract_rows() -> None:
    document = build()
    train, failures = _load_rows(document, "train")
    holdout, holdout_failures = _load_rows(document, "implementation_holdout")
    assert train and holdout
    assert failures == [] and holdout_failures == []
    vocabulary = _vocabulary(train)
    assert vocabulary
    assert all("payload=" not in token for token in vocabulary)


def test_process_candidate_is_not_promotable() -> None:
    document = build()
    assert document["promotion"]["training_allowed"] is False
