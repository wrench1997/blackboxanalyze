"""PG-348 static page fixture group A."""

from .generate_pages_a import (
    FIXTURE_ID,
    SCHEMA_VERSION,
    TEMPLATES,
    VARIANTS,
    build_manifest,
    generate,
)

__all__ = ["FIXTURE_ID", "SCHEMA_VERSION", "TEMPLATES", "VARIANTS", "build_manifest", "generate"]

