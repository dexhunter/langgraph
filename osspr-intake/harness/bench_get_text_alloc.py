"""Baseline allocation count of `get_text_at_path` on a store index path."""

from __future__ import annotations

import json
import os
import statistics
import tempfile

import memray
from langgraph.store.base.embed import get_text_at_path

# Store put indexing `records[*].{title,body}` over a few thousand documents.
# Each record re-tokenizes the same multi-field path and allocates a fresh
# one-element list per leaf.
N_RECORDS = 4000


def _document() -> dict:
    records = [
        {
            "title": f"title-{i:05d}",
            "body": f"body-{i:05d}-" + ("c" * 64),
            "tags": ["alpha", "beta"],
        }
        for i in range(N_RECORDS)
    ]
    return {"records": records}


def _allocations(document: dict) -> int:
    fd, name = tempfile.mkstemp(prefix="lg-gettext-", suffix=".bin")
    os.close(fd)
    os.remove(name)
    try:
        with memray.Tracker(
            name,
            native_traces=False,
            trace_python_allocators=True,
        ):
            texts = get_text_at_path(document, "records[*].{title,body}")
        if len(texts) != N_RECORDS * 2:
            raise SystemExit(f"unexpected text count {len(texts)}")
        reader = memray.FileReader(name)
        try:
            total = sum(
                record.n_allocations for record in reader.get_allocation_records()
            )
        finally:
            reader.close()
        return int(total)
    finally:
        if os.path.exists(name):
            os.remove(name)


def main() -> None:
    document = _document()
    _allocations(document)  # warmup
    samples = [_allocations(document) for _ in range(5)]
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
