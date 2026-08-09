from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from app.juice_shop_adapter import RULE_FAMILY_TEMPLATES, canonical_json


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_neural_url_set_head import TinyRuleSetGPT, url_set_features  # noqa: E402
from train_rule_memory_pilot import encode, make_prompt  # noqa: E402


LOOP11_CHECKPOINT = ROOT / "artifacts/neural-url-loop-11-url-meta-v2-20261529/tiny_rule_set_gpt.pt"
EXPECTED_CHECKPOINT_SHA256 = "37a627806ac5f796e2d90408ad04dcd73f9642178e1b68db9e32da20e4f1de32"
NEGATIVE_CONTROL_PATHS = {"/.well-known/security.txt", "/sitemap.xml", "/does-not-exist-sift-control"}
CONFIRMED_DEVIATION_PATHS = {"/metrics", "/ftp/"}


@dataclass(frozen=True)
class RankedAction:
    action: dict[str, str]
    score: float
    rationale: str
    inferred_family: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_action_bank(protocol_path: Path) -> list[dict[str, str]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    actions = protocol["candidate_generation"]["actions"]
    if len(actions) != len({canonical_json(action) for action in actions}):
        raise ValueError("baseline action bank contains duplicates")
    return [dict(action) for action in actions]


def random_ranking(actions: list[dict[str, str]], seed: int) -> list[RankedAction]:
    shuffled = [dict(action) for action in actions]
    random.Random(seed).shuffle(shuffled)
    return [
        RankedAction(action=action, score=float(len(shuffled) - index), rationale="seeded_random_order")
        for index, action in enumerate(shuffled)
    ]


def _query_row(path: str) -> dict[str, Any]:
    return {
        "episode_id": "juice-shop-unseen",
        "step": 0,
        "input": {"endpoint": f"http://127.0.0.1:3100{path}"},
        "context": {},
        "state": {},
        "history": [],
        "output": False,
    }


def _native_prompt(path: str, synthetic_context: list[dict[str, Any]] | None) -> str:
    return make_prompt(
        synthetic_context or [],
        _query_row(path),
        include_memory=synthetic_context is not None,
        routed_semantic_features=True,
        canonical_url_slots=True,
    )


def fixed_synthetic_url_context() -> list[dict[str, Any]]:
    """Pre-Juice-Shop URL examples; no target observations or metadata."""
    return [
        _synthetic_trace("https://safe.local/account", True, 0),
        _synthetic_trace("https://sub.safe.local/help", False, 1),
        _synthetic_trace("http://safe.local/status", True, 2),
        _synthetic_trace("https://other.local/", False, 3),
        _synthetic_trace("https://safe.local/docs", True, 4),
        _synthetic_trace("ftp://safe.local/archive", False, 5),
        _synthetic_trace("http://sub.safe.local/", False, 6),
        _synthetic_trace("https://safe.local/", True, 7),
    ]


def _synthetic_trace(endpoint: str, output: bool, step: int) -> dict[str, Any]:
    return {
        "episode_id": "pre-js-synthetic-url-memory",
        "step": step,
        "input": {"endpoint": endpoint},
        "context": {},
        "state": {},
        "history": [],
        "output": output,
    }


class FrozenLoop11Ranker:
    def __init__(self, checkpoint_path: Path = LOOP11_CHECKPOINT) -> None:
        if file_sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("Loop 11 checkpoint hash changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = TinyRuleSetGPT()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def score(self, path: str, *, use_synthetic_memory: bool) -> tuple[float, str]:
        context = fixed_synthetic_url_context() if use_synthetic_memory else None
        prompt = _native_prompt(path, context)
        token_values = encode(prompt, 639)
        tokens = torch.tensor([token_values], dtype=torch.long)
        lengths = torch.tensor([len(token_values)], dtype=torch.long)
        features = url_set_features(prompt).unsqueeze(0)
        probability = self.model(tokens, lengths, features).softmax(dim=-1)[0, 1].item()
        mode = "pre_js_synthetic_url_memory" if use_synthetic_memory else "empty_memory"
        return float(probability), mode

    def rank(self, actions: list[dict[str, str]], *, use_synthetic_memory: bool) -> list[RankedAction]:
        rows = []
        for action in actions:
            score, mode = self.score(action["path"], use_synthetic_memory=use_synthetic_memory)
            rows.append(RankedAction(dict(action), score, mode, inferred_family=None))
        return sorted(rows, key=lambda row: (-row.score, canonical_json(row.action)))


def c5_surface_ranking(actions: list[dict[str, str]]) -> list[RankedAction]:
    """Zero-parameter, executable generic-surface rules frozen before execution."""
    weights: list[tuple[str, float, str]] = [
        ("metrics", 1.00, "observability"),
        ("logs", 0.95, "observability"),
        ("debug", 0.90, "misconfiguration"),
        ("backup", 0.85, "information_exposure"),
        ("ftp", 0.80, "information_exposure"),
        ("swagger", 0.75, "misconfiguration"),
        ("api-docs", 0.70, "misconfiguration"),
        ("admin", 0.65, "access_control"),
        ("actuator", 0.60, "observability"),
    ]
    ranked = []
    for action in actions:
        path = action["path"].casefold()
        match = next(((score, family, token) for token, score, family in weights if token in path), None)
        if match is None:
            score, family, token = 0.0, None, "no_executable_surface_predicate"
        else:
            score, family, token = match
        ranked.append(
            RankedAction(
                dict(action),
                score,
                f"generic_path_token:{token}",
                inferred_family=family,
            )
        )
    return sorted(ranked, key=lambda row: (-row.score, canonical_json(row.action)))


def ranking_summary(rows: list[RankedAction], budget: int) -> dict[str, Any]:
    selected = rows[:budget]
    selected_controls = [row for row in selected if row.action["path"] in NEGATIVE_CONTROL_PATHS]
    top = selected[0] if selected else None
    return {
        "budget": budget,
        "top1": top.action if top else None,
        "counterexample_top1": bool(top and top.action["path"] in CONFIRMED_DEVIATION_PATHS),
        "selected_negative_controls": len(selected_controls),
        "negative_control_false_positive_rate": round(len(selected_controls) / len(NEGATIVE_CONTROL_PATHS), 6),
        "rule_abstraction_coverage": round(
            sum(row.inferred_family is not None for row in selected) / len(selected), 6
        ) if selected else 0.0,
        "ranked": [
            {
                "rank": index + 1,
                "action": row.action,
                "score": round(row.score, 8),
                "rationale": row.rationale,
                "inferred_family": row.inferred_family,
                "rule_ir": RULE_FAMILY_TEMPLATES.get(row.inferred_family) if row.inferred_family else None,
            }
            for index, row in enumerate(rows)
        ],
    }


def policy_builders(seed: int) -> dict[str, Callable[[list[dict[str, str]]], list[RankedAction]]]:
    neural = FrozenLoop11Ranker()
    return {
        "random": lambda actions: random_ranking(actions, seed),
        "frozen_neural_no_memory": lambda actions: neural.rank(actions, use_synthetic_memory=False),
        "frozen_neural_synthetic_memory": lambda actions: neural.rank(actions, use_synthetic_memory=True),
        "C5_executable_rule": c5_surface_ranking,
    }
