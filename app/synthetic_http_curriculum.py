from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpSurface:
    name: str
    label: int
    features: tuple[tuple[str, str], ...]

    def render(self) -> str:
        return "|".join(f"http.{key}={value}" for key, value in self.features)


POSITIVE_SURFACES = (
    HttpSurface("prometheus_text_surface", 1, (("status", "2xx"), ("media", "text_plain_prometheus"), ("length", "large"), ("body", "metric_samples"))),
    HttpSurface("html_directory_listing", 1, (("status", "2xx"), ("media", "text_html"), ("length", "large"), ("body", "directory_links"))),
    HttpSurface("json_debug_surface", 1, (("status", "2xx"), ("media", "application_json"), ("length", "medium"), ("body", "diagnostic_fields"))),
    HttpSurface("traceback_surface", 1, (("status", "5xx"), ("media", "text_html"), ("length", "medium"), ("body", "stack_trace"))),
    HttpSurface("source_map_surface", 1, (("status", "2xx"), ("media", "application_json"), ("length", "large"), ("body", "source_mapping"))),
)
NEGATIVE_SURFACES = (
    HttpSurface("spa_shell", 0, (("status", "2xx"), ("media", "text_html"), ("length", "medium"), ("body", "application_shell"))),
    HttpSurface("robots_text", 0, (("status", "2xx"), ("media", "text_plain"), ("length", "tiny"), ("body", "robots_directives"))),
    HttpSurface("security_policy_text", 0, (("status", "2xx"), ("media", "text_plain"), ("length", "small"), ("body", "policy_contacts"))),
    HttpSurface("ordinary_json", 0, (("status", "2xx"), ("media", "application_json"), ("length", "small"), ("body", "ordinary_records"))),
    HttpSurface("not_found_json", 0, (("status", "4xx"), ("media", "application_json"), ("length", "tiny"), ("body", "not_found"))),
)


def feature_vector(surface: HttpSurface) -> list[float]:
    values = dict(surface.features)
    status = values.get("status", "")
    media = values.get("media", "")
    body = values.get("body", "")
    length = values.get("length", "")
    return [
        float(body == "metric_samples"),
        float(body == "directory_links"),
        float(body == "diagnostic_fields"),
        float(body == "stack_trace"),
        float(body == "source_mapping"),
        float(body == "application_shell"),
        float(body in {"robots_directives", "policy_contacts"}),
        float(body in {"ordinary_records", "not_found"}),
    ]


def render_prompt(context: list[HttpSurface], query: HttpSurface) -> str:
    trace = "|".join(f"{surface.render()}:{surface.label}" for surface in context)
    return f"<RSEM><TRACE>{trace}<RULEMEM><QUERY>{query.render()}<ANSWER>"


def generate_examples(count: int, seed: int, *, validation: bool = False) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    surfaces = POSITIVE_SURFACES + NEGATIVE_SURFACES
    examples = []
    for index in range(count):
        query = rng.choice(surfaces)
        context = [rng.choice(surfaces) for _ in range(8)]
        examples.append({
            "prompt": render_prompt(context, query),
            "label": query.label,
            "family": "synthetic_http_response_validation" if validation else "synthetic_http_response_train",
            "record_id": f"http-{seed}-{index}",
            "intended_label": query.label,
        })
    return examples


def inference_context() -> list[HttpSurface]:
    return [
        POSITIVE_SURFACES[0],
        NEGATIVE_SURFACES[0],
        POSITIVE_SURFACES[1],
        NEGATIVE_SURFACES[1],
        POSITIVE_SURFACES[2],
        NEGATIVE_SURFACES[2],
        POSITIVE_SURFACES[3],
        NEGATIVE_SURFACES[3],
    ]
