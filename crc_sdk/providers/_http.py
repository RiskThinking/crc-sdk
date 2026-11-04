"""Shared request defaults for providers reading public data services."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.request import Request

#: Identifies SDK traffic to public data services (same as the workflows').
USER_AGENT = "crc-sdk"
#: Seconds a provider waits on a connect or a stalled read before failing.
HTTP_TIMEOUT = 60


def http_request(
    url: str,
    *,
    method: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Request:
    """A `Request` carrying the SDK `User-Agent` plus any extra headers."""
    return Request(
        url,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method=method,
    )
