"""Fixed parameters for the issue 54195 reproduction.

The defaults intentionally match the issue report:

* 200,000 vectors with 128 dimensions, 100 tenants, correlation 0.85
* HNSW M=16 / efConstruction=128, search ef=64, top_k=10
* 128 repetitions of the always-true ``age`` predicate
* Strong consistency on every request

Every value can be overridden from the command line. Keep the defaults when
the goal is to compare numbers with the issue report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any


@dataclass(frozen=True)
class Config:
    uri: str = "http://127.0.0.1:19530"
    token: str | None = None

    ann_collection: str = "issue54195_ann"
    flat_collection: str = "issue54195_flat"

    num_rows: int = 200_000
    dim: int = 128
    num_tenants: int = 100
    correlation: float = 0.85

    data_seed: int = 20261002
    query_seed: int = 20261003
    order_seed: int = 20261004

    top_k: int = 10
    ef: int = 64
    hnsw_m: int = 16
    hnsw_ef_construction: int = 128

    true_predicate_repeats: int = 128
    insert_batch_size: int = 20_000
    query_count: int = 4

    warmup_pairs: int = 4
    measured_pairs: int = 10
    flat_warmup_pairs: int = 2
    # The issue report used two balanced-order pairs on the FLAT collection.
    flat_pairs: int = 2

    # Default server-side window for a single query. The issue's eligible set is
    # 1,985 rows, so the fast path stays below this limit.
    query_window: int = 16_384
    timeout: float = 1_800.0

    def __post_init__(self) -> None:
        if self.num_rows <= 0:
            raise ValueError("num_rows must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_tenants <= 0:
            raise ValueError("num_tenants must be positive")
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError("correlation must be in [0, 1)")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.ef <= 0:
            raise ValueError("ef must be positive")
        if self.true_predicate_repeats < 1:
            raise ValueError("true_predicate_repeats must be at least 1")
        if self.hnsw_m <= 0 or self.hnsw_ef_construction <= 0:
            raise ValueError("HNSW M and efConstruction must be positive")
        if self.insert_batch_size <= 0:
            raise ValueError("insert_batch_size must be positive")
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if self.warmup_pairs < 0:
            raise ValueError("warmup_pairs must be non-negative")
        if self.flat_warmup_pairs < 0:
            raise ValueError("flat_warmup_pairs must be non-negative")
        if self.measured_pairs <= 0 or self.flat_pairs <= 0:
            raise ValueError("measured_pairs and flat_pairs must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.query_window <= 0:
            raise ValueError("query_window must be positive")
        if self.ann_collection == self.flat_collection:
            raise ValueError("ANN and FLAT collections must have different names")

    @classmethod
    def from_env(cls) -> "Config":
        """Build a config from MILVUS_URI / MILVUS_TOKEN when present."""

        return cls(
            uri=os.environ.get("MILVUS_URI", "http://127.0.0.1:19530"),
            token=os.environ.get("MILVUS_TOKEN"),
        )

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with non-None overrides applied."""

        known = {field.name for field in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise TypeError(f"unknown config fields: {', '.join(sorted(unknown))}")
        applied = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **applied)
