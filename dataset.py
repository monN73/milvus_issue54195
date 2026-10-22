"""Deterministic data and query generation for issue 54195.

The dataset follows the issue description:

* one normalized random center per tenant;
* each vector is ``c * center[tenant] + sqrt(1-c^2) * normalized_noise``,
  normalized again;
* ``category == tenant + 1`` and ``parity`` is derived from tenant parity.

Metadata uses an independent RNG stream, so changing ``insert_batch_size`` does
not change any generated value. Vector generation keeps the draw order from the
issue report: centers, tenants, noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .config import Config


@dataclass(frozen=True)
class Dataset:
    ids: np.ndarray
    vectors: np.ndarray
    tenant: np.ndarray
    level: np.ndarray
    category: np.ndarray
    category5: np.ndarray
    age: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.ids.shape[0])


def generate_centers(cfg: Config) -> np.ndarray:
    """Generate the tenant centers exactly as the dataset generator does."""

    rng = np.random.default_rng(cfg.data_seed)
    centers = rng.normal(size=(cfg.num_tenants, cfg.dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    return centers


def generate_dataset(cfg: Config) -> Dataset:
    rng = np.random.default_rng(cfg.data_seed)

    centers = rng.normal(size=(cfg.num_tenants, cfg.dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    tenant = rng.integers(0, cfg.num_tenants, size=cfg.num_rows).astype(np.int64)

    noise = rng.normal(size=(cfg.num_rows, cfg.dim)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)

    mix = np.float32(cfg.correlation)
    residual = np.float32(np.sqrt(1.0 - cfg.correlation**2))
    vectors = mix * centers[tenant] + residual * noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors.astype(np.float32, copy=False)
    del noise

    # Independent stream: metadata never depends on the vector batch size.
    meta_rng = np.random.default_rng(cfg.data_seed + 1)
    level = meta_rng.integers(0, 32, size=cfg.num_rows, dtype=np.int64)
    age = meta_rng.integers(1, 1_001, size=cfg.num_rows, dtype=np.int64)

    ids = np.arange(cfg.num_rows, dtype=np.int64)
    return Dataset(
        ids=ids,
        vectors=vectors,
        tenant=tenant,
        level=level,
        category=tenant + 1,
        category5=tenant % 5 + 1,
        age=age,
    )


def iter_row_batches(
    dataset: Dataset, batch_size: int
) -> Iterator[list[dict[str, object]]]:
    """Yield MilvusClient row-based insert batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, dataset.num_rows, batch_size):
        stop = min(start + batch_size, dataset.num_rows)
        rows: list[dict[str, object]] = []
        for index in range(start, stop):
            tenant = int(dataset.tenant[index])
            rows.append(
                {
                    "id": int(dataset.ids[index]),
                    "vector": dataset.vectors[index].tolist(),
                    "tenant": tenant,
                    "level": int(dataset.level[index]),
                    "category": int(dataset.category[index]),
                    "category5": int(dataset.category5[index]),
                    "age": int(dataset.age[index]),
                    "parity": "even" if tenant % 2 == 0 else "odd",
                }
            )
        yield rows


def generate_queries(cfg: Config, centers: np.ndarray) -> list[np.ndarray]:
    """Generate deterministic query vectors near the eligible tenant center.

    ``category == 1`` is equivalent to ``tenant == 0``. The issue selects an
    eligible tenant center, adds N(0, 0.03) noise, then normalizes the result.
    """

    rng = np.random.default_rng(cfg.query_seed)
    eligible_tenants = [0]
    queries: list[np.ndarray] = []
    for _ in range(cfg.query_count):
        tenant = int(rng.choice(eligible_tenants))
        query = centers[tenant] + rng.normal(scale=0.03, size=cfg.dim)
        query = (query / np.linalg.norm(query)).astype(np.float32)
        queries.append(query)
    return queries
