import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_rule_memory_pilot.py"
SPEC = importlib.util.spec_from_file_location("train_rule_memory_pilot", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_rule_memory_prediction = MODULE._rule_memory_prediction
make_prompt = MODULE.make_prompt
compact_input = MODULE.compact_input


def _row(output: bool, **values):
    return {"input": values, "context": {}, "state": {}, "output": output}


def test_executable_rule_memory_induces_numeric_cut():
    context = [_row(value > 0, score=value) for value in (-2, -1, 0, 1, 2, 3)]
    prompt = make_prompt(context, _row(False, score=4), episode_rule_features=True)
    assert "generic=input.score#gt=0" in prompt
    assert _rule_memory_prediction(prompt) == 1


def test_executable_rule_memory_binds_categorical_guard_and_nonzero_flag():
    context = [
        _row(False, role="guest", flags=0),
        _row(False, role="member", flags=0),
        _row(True, role="admin", flags=0),
        _row(True, role="guest", flags=1),
        _row(True, role="member", flags=-1),
        _row(True, role="admin", flags=2),
        _row(True, role="guest", flags=-2),
        _row(True, role="member", flags=3),
    ]
    admin_prompt = make_prompt(context, _row(False, role="admin", flags=0), episode_rule_features=True)
    guest_prompt = make_prompt(context, _row(False, role="guest", flags=0), episode_rule_features=True)
    assert "generic=" in admin_prompt
    assert _rule_memory_prediction(admin_prompt) == 1
    assert _rule_memory_prediction(guest_prompt) == 0


def test_executable_rule_memory_recovers_previous_step_relation():
    context = []
    for current, previous in (("a", "a"), ("a", "b"), ("b", "b"), ("c", "a"), ("d", "d"), ("b", "c"), ("c", "c"), ("d", "a")):
        row = _row(current == previous, token=current)
        row["history"] = [{"input": {"token": previous}, "context": {}, "state": {}}]
        context.append(row)
    query = _row(False, token="z")
    query["history"] = [{"input": {"token": "z"}, "context": {}, "state": {}}]
    prompt = make_prompt(context, query, episode_rule_features=True, routed_semantic_features=True)
    assert "#eq_prev1" in prompt
    assert "prev1.input.token" in prompt
    assert _rule_memory_prediction(prompt) == 1


def test_routed_markup_projection_records_decode_depth():
    trace = _row(False, encoded="&amp;lt;b&amp;gt;x&amp;lt;/b&amp;gt;")
    projected = compact_input(trace, routed_semantic_features=True)
    assert "d01" in projected


def test_canonical_url_slot_removes_source_field_name_and_keeps_effective_port():
    origin = compact_input(_row(False, origin="https://trusted.com:443/home"), routed_semantic_features=True, canonical_url_slots=True)
    endpoint = compact_input(_row(False, endpoint="https://trusted.com/home"), routed_semantic_features=True, canonical_url_slots=True)
    assert origin == endpoint == "url.value=u:https|trusted.com|p443|/home"
    assert "input.origin" not in origin
    assert "input.endpoint" not in endpoint


def test_iid_split_is_stratified_within_every_family():
    records = [
        {"family": family, "record_id": f"{family}-{index}"}
        for index in range(10)
        for family in ("alpha", "beta", "gamma")
    ]
    train, validation = MODULE.stratified_iid_split(records)
    assert {row["family"] for row in train} == {"alpha", "beta", "gamma"}
    assert {row["family"] for row in validation} == {"alpha", "beta", "gamma"}
    assert len(train) == 24
    assert len(validation) == 6


def test_structured_url_rule_induces_hostname_equality_without_family_metadata():
    context = [
        _row(True, origin="https://trusted.com"),
        _row(True, origin="http://trusted.com"),
        _row(True, origin="ftp://trusted.com"),
        _row(True, origin="ws://trusted.com"),
        _row(False, origin="https://sub.trusted.com"),
        _row(False, origin="https://trusted.com.evil.test"),
        _row(False, origin="null"),
    ]
    prompt = make_prompt(
        context,
        _row(False, origin="https://trusted.com"),
        episode_rule_features=True,
        structured_url_rule=True,
    )
    assert "url.hostname(input.origin)==trusted.com" in prompt
    assert "|kind=url_hostname" in prompt


def test_structured_url_rule_rejects_homogeneous_unidentifiable_episode():
    context = [
        _row(True, url="https://trusted.com/home"),
        _row(True, url="https://trusted.com.evil.test/phish"),
        _row(True, url="https://evil.test/trusted.com"),
        _row(True, url="https://evil.test/?next=trusted.com"),
    ]
    prompt = make_prompt(
        context,
        _row(False, url="https://example.test/home"),
        episode_rule_features=True,
        structured_url_rule=True,
    )
    assert "|kind=url_" not in prompt


def test_confidence_arbitration_prefers_exact_structured_url_rule_over_suffix():
    prompt = (
        "<RSEM><TRACE><RULEMEM>"
        "url.hostname(input.origin)==trusted.com|q=1|fit=7/7|kind=url_hostname;"
        "input.origin#suffix=trusted.com|q=0|fit=6/7"
        "<QUERY>input.origin=u:https|trusted.com|/<ANSWER>"
    )
    assert _rule_memory_prediction(prompt) == 0
    assert _rule_memory_prediction(prompt, confidence_arbitration=True) == 1
    assert _rule_memory_prediction(prompt, confidence_arbitration=True, abstain_on_rule_conflict=True) == 1


def test_identifiability_arbitration_abstains_on_equal_fit_conflict():
    prompt = (
        "<RSEM><TRACE><RULEMEM>"
        "url.hostname(input.endpoint)==trusted.com|q=0|fit=8/8|kind=url_hostname;"
        "input.endpoint#suffix=trusted.com|q=1|fit=8/8"
        "<QUERY>input.endpoint=u:https|other.test|/trusted.com<ANSWER>"
    )
    assert _rule_memory_prediction(prompt, confidence_arbitration=True, abstain_on_rule_conflict=True) is None


def test_identifiability_arbitration_uses_better_fitting_rule_before_priority():
    prompt = (
        "<RSEM><TRACE><RULEMEM>"
        "url.hostname(input.origin)==trusted.com|q=1|fit=7/7|kind=url_hostname;"
        "input.origin#suffix=trusted.com|q=0|fit=6/7"
        "<QUERY>input.origin=u:https|trusted.com|/<ANSWER>"
    )
    assert _rule_memory_prediction(prompt, confidence_arbitration=True, abstain_on_rule_conflict=True) == 1


def test_url_suffix_suppression_does_not_remove_plain_string_suffix_rule():
    url_context = [
        _row(True, origin="https://trusted.com"),
        _row(True, origin="http://trusted.com"),
        _row(False, origin="https://evil.test"),
        _row(False, origin="null"),
    ]
    url_prompt = make_prompt(url_context, _row(False, origin="ftp://trusted.com"), episode_rule_features=True, suppress_url_suffix=True)
    assert "#suffix=" not in url_prompt
    text_context = [_row(value.endswith(".good"), token=value) for value in ("a.good", "b.good", "c.bad", "d.bad")]
    text_prompt = make_prompt(text_context, _row(False, token="z.good"), episode_rule_features=True, suppress_url_suffix=True)
    assert "#suffix=.good" in text_prompt
