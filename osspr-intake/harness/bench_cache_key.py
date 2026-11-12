"""Baseline output size of `default_cache_key` on a cached-node state."""

from __future__ import annotations

import json
import statistics

from langgraph._internal._cache import default_cache_key


def _state() -> dict:
    """Node input shaped like a cached chat turn: unique message bodies.

    Pregel calls `cache_policy.key_func(val)` with this kind of state, then
    immediately passes the returned bytes to `xxh3_128_hexdigest`.
    """
    messages = []
    for i in range(800):
        message = {
            "role": "assistant" if i % 2 else "user",
            "content": f"msg-{i:04d}:" + ("x" * 1200),
            "id": f"msg-{i:05d}",
            "additional_kwargs": {},
            "response_metadata": {"model": "benchmark", "idx": i},
        }
        if i % 2:
            message["tool_calls"] = [
                {
                    "id": f"call-{i:05d}",
                    "name": "search",
                    "args": {"q": f"q{i:04d}-" + ("y" * 240)},
                }
            ]
        messages.append(message)
    return {"messages": messages, "user": "benchmark-user", "step": 3}


def _nbytes(state: dict) -> int:
    blob = default_cache_key(state)
    if not isinstance(blob, (bytes, bytearray)):
        raise SystemExit(f"expected bytes, got {type(blob).__name__}")
    return len(blob)


def main() -> None:
    state = _state()
    _nbytes(state)  # warmup
    samples = [_nbytes(state) for _ in range(5)]
    median = float(statistics.median(samples))
    noise = (max(samples) - min(samples)) / median * 100.0 if median else 0.0
    print(
        json.dumps(
            {
                "samples": samples,
                "median": median,
                "noise_pct": noise,
            }
        )
    )


if __name__ == "__main__":
    main()
