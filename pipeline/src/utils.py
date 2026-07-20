"""Shared helpers: JSON parsing and the retry-once-on-malformed-output pattern
used by both extract_entities.py and map_codes.py.
"""
import json
import time
from typing import Callable

from groq import RateLimitError


def parse_json_or_none(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def call_with_backoff(
    call_fn: Callable[[], str],
    max_retries: int = 8,
    backoff_seconds: float = 8.0,
) -> str:
    """Call `call_fn()`, transparently retrying on a per-minute (TPM) rate
    limit by waiting `backoff_seconds`. A per-day (TPD) limit means the
    day's quota is genuinely exhausted, so that's re-raised immediately
    instead of retried.
    """
    for attempt in range(max_retries):
        try:
            return call_fn()
        except RateLimitError as e:
            if "tokens per day" in str(e).lower():
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff_seconds)
    raise AssertionError("unreachable")


def call_with_retry(
    primary_call: Callable[[], str],
    fallback_call: Callable[[], str],
    validate_fn: Callable[[dict], bool],
) -> tuple[dict | None, str]:
    """Call `primary_call()`; if the result isn't valid JSON matching
    `validate_fn`, retry once with `fallback_call()`. Returns
    (parsed_or_None, last_raw_response) so the raw text can be logged
    for debugging even on failure.
    """
    raw = primary_call()
    parsed = parse_json_or_none(raw)
    if parsed is not None and validate_fn(parsed):
        return parsed, raw

    raw = fallback_call()
    parsed = parse_json_or_none(raw)
    if parsed is not None and validate_fn(parsed):
        return parsed, raw

    return None, raw
