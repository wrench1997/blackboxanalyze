"""Generate per-episode inert markers for local reflection/DOM observations."""

from __future__ import annotations

import hashlib
import re
import secrets


_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]{1,24}$")
_MARKER_RE = re.compile(r"^[A-Za-z0-9_.-]{3,96}$")


def fresh_marker(prefix: str = "PG25XSS", *, entropy_bytes: int = 10) -> str:
    if not _PREFIX_RE.fullmatch(prefix):
        raise ValueError("marker prefix must be bounded and inert")
    if not 4 <= int(entropy_bytes) <= 32:
        raise ValueError("marker entropy must stay within the bounded range")
    marker = f"{prefix}_{secrets.token_hex(int(entropy_bytes)).upper()}"
    if not _MARKER_RE.fullmatch(marker):
        raise AssertionError("generated marker failed its inert grammar")
    return marker


def marker_sha256(marker: str) -> str:
    if not _MARKER_RE.fullmatch(marker):
        raise ValueError("marker is not in the inert grammar")
    return hashlib.sha256(marker.encode("utf-8")).hexdigest()


__all__ = ["fresh_marker", "marker_sha256"]
