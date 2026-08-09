from __future__ import annotations

from typing import Any
import sys
from pathlib import Path

import torch

from app.synthetic_http_curriculum import HttpSurface, inference_context, render_prompt
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from train_rule_memory_pilot import encode


URL_SLOTS = 120
RESPONSE_SLOTS = 8


def projection_surface(projection: Any) -> HttpSurface:
    status = str(projection.status_class).replace("xx", "xx")
    if projection.prometheus_media_type:
        media, body = "text_plain_prometheus", "metric_samples"
    elif projection.content_type_class == "html":
        media, body = "text_html", "directory_links" if projection.generic_listing_surface else "application_shell"
    elif projection.content_type_class == "json":
        media, body = "application_json", "ordinary_records"
    elif projection.content_type_class == "text":
        media, body = "text_plain", "policy_contacts"
    else:
        media, body = "other", "ordinary_records"
    return HttpSurface(
        "observed_http_surface",
        0,
        (("status", status), ("media", media), ("length", projection.content_length_bucket), ("body", body)),
    )


def response_prompt(projection: Any) -> str:
    return render_prompt(inference_context(), projection_surface(projection))


def response_feature_tensor(projection: Any, *, enabled: bool = True) -> torch.Tensor:
    features = torch.zeros(128, dtype=torch.float32)
    if enabled:
        features[URL_SLOTS:] = torch.tensor(projection.feature_vector(), dtype=torch.float32)
    return features


def score_observation(model: torch.nn.Module, projection: Any, *, device: torch.device, enabled: bool = True) -> float:
    prompt = response_prompt(projection)
    token_values = encode(prompt, 639)
    tokens = torch.tensor([token_values], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_values)], dtype=torch.long, device=device)
    features = response_feature_tensor(projection, enabled=enabled).unsqueeze(0).to(device)
    with torch.inference_mode():
        return float(model(tokens, lengths, features).softmax(dim=-1)[0, 1])
