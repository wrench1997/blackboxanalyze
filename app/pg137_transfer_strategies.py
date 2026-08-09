"""PG-137 transfer-strategy definitions for causal-token pretraining.

This module keeps the comparison explicit: the same fresh replay and the
same causal vocabulary are used for every strategy, while only the action
adaptation rule changes.  It is an experiment registry, not a vulnerability
scanner and not a promotion path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SCHEMA_VERSION = "pg137-causal-transfer-strategies-v1"
StrategyName = Literal["scratch", "frozen_body", "low_lr_full", "joint_lm_action"]
STRATEGIES: tuple[StrategyName, ...] = ("scratch", "frozen_body", "low_lr_full", "joint_lm_action")


@dataclass(frozen=True)
class TransferConfig:
    name: StrategyName
    pretrained: bool
    freeze_causal_body: bool
    body_learning_rate: float
    action_learning_rate: float
    lm_loss_weight: float


CONFIGS: dict[StrategyName, TransferConfig] = {
    "scratch": TransferConfig("scratch", False, False, 2e-3, 2e-3, 0.0),
    "frozen_body": TransferConfig("frozen_body", True, True, 0.0, 2e-3, 0.0),
    "low_lr_full": TransferConfig("low_lr_full", True, False, 2e-4, 2e-3, 0.0),
    "joint_lm_action": TransferConfig("joint_lm_action", True, False, 2e-4, 2e-3, 0.25),
}


def config_for(name: str) -> TransferConfig:
    try:
        return CONFIGS[name]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"unknown PG-137 transfer strategy: {name}") from exc


def strategy_manifest() -> list[dict[str, object]]:
    return [
        {
            "name": config.name,
            "pretrained": config.pretrained,
            "freeze_causal_body": config.freeze_causal_body,
            "body_learning_rate": config.body_learning_rate,
            "action_learning_rate": config.action_learning_rate,
            "lm_loss_weight": config.lm_loss_weight,
        }
        for config in (CONFIGS[name] for name in STRATEGIES)
    ]


__all__ = ["CONFIGS", "SCHEMA_VERSION", "STRATEGIES", "TransferConfig", "config_for", "strategy_manifest"]
