"""Baseline peak Python allocations inside `Topic.update`."""

from __future__ import annotations

import json
import statistics
import tracemalloc

from langgraph.channels.topic import Topic

# One superstep of fan-in: 400 batches of 2_000 ids. ``Topic.update`` flattens
# them through ``tuple(_flatten(...))`` and then copies those pointers again
# into `self.values`.
N_BATCHES = 400
BATCH = 2000


def _batches() -> list[list[int]]:
    return [list(range(BATCH)) for _ in range(N_BATCHES)]


def _peak(batches: list[list[int]]) -> int:
    topic = Topic(int, accumulate=False)
    tracemalloc.start()
    tracemalloc.reset_peak()
    updated = topic.update(batches)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if not updated or len(topic.values) != N_BATCHES * BATCH:
        raise SystemExit(
            f"unexpected topic state updated={updated} n={len(topic.values)}"
        )
    return peak


def main() -> None:
    batches = _batches()
    _peak(batches)  # warmup
    samples = [_peak(batches) for _ in range(5)]
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
