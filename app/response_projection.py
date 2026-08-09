from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _length_bucket(value: int | None) -> str:
    if value is None:
        return "missing"
    if value == 0:
        return "zero"
    if value < 128:
        return "tiny"
    if value < 4096:
        return "small"
    if value < 11000:
        return "medium"
    return "large"


@dataclass(frozen=True)
class ResponseProjection:
    status_class: str
    content_type_class: str
    content_length_bucket: str
    has_content_length: bool
    body_length: int
    prometheus_media_type: bool
    generic_text_surface: bool
    generic_listing_surface: bool
    body_shape: str

    @classmethod
    def from_observation(cls, observation: dict[str, Any]) -> "ResponseProjection":
        outer = observation.get("observation", observation)
        status = int(outer.get("status_code", 0))
        headers = {str(k).casefold(): str(v) for k, v in dict(outer.get("headers") or {}).items()}
        summary = dict(outer.get("summary") or {})
        content_type = headers.get("content-type", "").casefold()
        if "json" in content_type:
            type_class = "json"
        elif "html" in content_type:
            type_class = "html"
        elif "text" in content_type:
            type_class = "text"
        elif content_type:
            type_class = "other"
        else:
            type_class = "missing"
        declared_length = None
        if headers.get("content-length", "").isdigit():
            declared_length = int(headers["content-length"])
        body_length = int(summary.get("body_length", 0))
        body_preview = str(summary.get("body_preview", ""))
        body_shape = str(summary.get("body_shape") or _infer_body_shape(content_type, body_preview, body_length, declared_length))
        effective_length = declared_length if declared_length is not None else body_length
        prometheus_media_type = "text/plain; version=0.0.4" in content_type
        return cls(
            status_class=f"{status // 100}xx",
            content_type_class=type_class,
            content_length_bucket=_length_bucket(effective_length),
            has_content_length=declared_length is not None,
            body_length=body_length,
            prometheus_media_type=prometheus_media_type,
            generic_text_surface=status // 100 == 2 and prometheus_media_type,
            generic_listing_surface=status // 100 == 2 and type_class == "html" and effective_length >= 11000,
            body_shape=body_shape,
        )

    def score(self) -> float:
        if self.generic_text_surface:
            return 1.0
        if self.generic_listing_surface:
            return 0.8
        if self.status_class == "2xx" and self.content_type_class in {"json", "other"}:
            return 0.4
        return 0.0

    def inferred_family(self) -> str | None:
        if self.generic_text_surface:
            return "observability"
        if self.generic_listing_surface:
            return "information_exposure"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_class": self.status_class,
            "content_type_class": self.content_type_class,
            "content_length_bucket": self.content_length_bucket,
            "has_content_length": self.has_content_length,
            "body_length": self.body_length,
            "prometheus_media_type": self.prometheus_media_type,
            "generic_text_surface": self.generic_text_surface,
            "generic_listing_surface": self.generic_listing_surface,
            "score": self.score(),
            "inferred_family": self.inferred_family(),
            "body_shape": self.body_shape,
        }

    def feature_vector(self) -> list[float]:
        shape = self.body_shape
        return [
            float(shape == "prometheus"),
            float(shape == "directory_listing"),
            float(shape == "diagnostic"),
            float(shape == "traceback"),
            float(shape == "source_map"),
            float(shape == "spa_shell"),
            float(shape in {"robots_text", "security_policy_text"}),
            float(shape in {"ordinary_json", "not_found"}),
        ]


def _infer_body_shape(content_type: str, body_preview: str, body_length: int, declared_length: int | None) -> str:
    lowered = body_preview.casefold()
    effective_length = declared_length if declared_length is not None else body_length
    if "text/plain; version=0.0.4" in content_type or "# help " in lowered or "# type " in lowered:
        return "prometheus"
    if "listing directory" in lowered or ("<pre" in lowered and "href=" in lowered and effective_length >= 11000):
        return "directory_listing"
    if "sourcemappingurl" in lowered or '"sources"' in lowered or '"mappings"' in lowered:
        return "source_map"
    if "stack trace" in lowered or "error:    at " in lowered or "traceback" in lowered:
        return "traceback"
    if any(token in lowered for token in ("debug", "diagnostic", '"stack"', '"error"')) and "json" in content_type:
        return "diagnostic"
    if "user-agent:" in lowered or "disallow:" in lowered:
        return "robots_text"
    if "security.txt" in lowered or "contact:" in lowered or "policy" in lowered:
        return "security_policy_text"
    if "html" in content_type and ("<app-root" in lowered or "owasp juice shop" in lowered or effective_length in range(9800, 10050)):
        return "spa_shell"
    if "json" in content_type:
        return "ordinary_json" if effective_length < 4096 else "diagnostic"
    return "ordinary_json" if "json" in content_type else "unknown"
