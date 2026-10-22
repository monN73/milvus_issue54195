"""Balanced-order latency measurement for issue 54195."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from pymilvus import MilvusClient

from .config import Config
from .dataset import generate_centers, generate_queries
from .filters import ORIGINAL_FILTER, equivalent_filter

Log = Callable[[str], None]


@dataclass(frozen=True)
class Sample:
    latency_ms: float
    ids: tuple[int, ...]


@dataclass(frozen=True)
class PairMeasurement:
    original: tuple[Sample, ...]
    equivalent: tuple[Sample, ...]
    mismatched_pairs: int

    @property
    def topk_equal(self) -> bool:
        return self.mismatched_pairs == 0


def _hit_id(hit: object) -> int:
    if isinstance(hit, dict):
        return int(hit["id"])
    return int(getattr(hit, "id"))  # pragma: no cover - ORM-style fallback


def search_once(
    client: MilvusClient,
    collection_name: str,
    query: np.ndarray,
    filter_expr: str,
    cfg: Config,
    *,
    hnsw: bool,
) -> Sample:
    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": cfg.ef} if hnsw else {},
    }
    started = time.perf_counter()
    result = client.search(
        collection_name=collection_name,
        data=[query],
        anns_field="vector",
        search_params=search_params,
        limit=cfg.top_k,
        filter=filter_expr,
        output_fields=["id"],
        consistency_level="Strong",
        timeout=cfg.timeout,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    ids = tuple(_hit_id(hit) for hit in result[0])
    return Sample(latency_ms=latency_ms, ids=ids)


def measure_query(
    client: MilvusClient,
    collection_name: str,
    query: np.ndarray,
    cfg: Config,
    *,
    pairs: int,
    warmup_pairs: int,
    hnsw: bool,
    order_seed: int,
) -> PairMeasurement:
    original_filter = ORIGINAL_FILTER
    equivalent_expr = equivalent_filter(cfg.true_predicate_repeats)

    for round_index in range(warmup_pairs):
        first, second = (
            (original_filter, equivalent_expr)
            if round_index % 2 == 0
            else (equivalent_expr, original_filter)
        )
        search_once(client, collection_name, query, first, cfg, hnsw=hnsw)
        search_once(client, collection_name, query, second, cfg, hnsw=hnsw)

    directions = (
        ["original_first"] * (pairs // 2 + pairs % 2)
        + ["equivalent_first"] * (pairs // 2)
    )
    random.Random(order_seed).shuffle(directions)

    original_samples: list[Sample] = []
    equivalent_samples: list[Sample] = []
    mismatched_pairs = 0
    for direction in directions:
        if direction == "original_first":
            original_sample = search_once(
                client, collection_name, query, original_filter, cfg, hnsw=hnsw
            )
            equivalent_sample = search_once(
                client, collection_name, query, equivalent_expr, cfg, hnsw=hnsw
            )
        else:
            equivalent_sample = search_once(
                client, collection_name, query, equivalent_expr, cfg, hnsw=hnsw
            )
            original_sample = search_once(
                client, collection_name, query, original_filter, cfg, hnsw=hnsw
            )
        original_samples.append(original_sample)
        equivalent_samples.append(equivalent_sample)
        if original_sample.ids != equivalent_sample.ids:
            mismatched_pairs += 1

    return PairMeasurement(
        original=tuple(original_samples),
        equivalent=tuple(equivalent_samples),
        mismatched_pairs=mismatched_pairs,
    )


def summarize(samples: Sequence[Sample]) -> dict[str, float | int]:
    if not samples:
        return {"count": 0}
    values = np.asarray([sample.latency_ms for sample in samples], dtype=np.float64)
    return {
        "count": int(values.size),
        "median_ms": float(np.median(values)),
        "mean_ms": float(values.mean()),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _ratio(equivalent: dict[str, float | int], original: dict[str, float | int]) -> float | None:
    original_median = original.get("median_ms")
    equivalent_median = equivalent.get("median_ms")
    if not original_median or equivalent_median is None:
        return None
    return float(equivalent_median) / float(original_median)


def _recall_at_k(ids: Sequence[int], ground_truth: Sequence[int]) -> float:
    if not ground_truth:
        return 0.0
    return len(set(ids) & set(ground_truth)) / len(ground_truth)


def _query_record(measurement: PairMeasurement, ground_truth: Sequence[int] | None) -> dict:
    original_summary = summarize(measurement.original)
    equivalent_summary = summarize(measurement.equivalent)
    record: dict[str, object] = {
        "original": original_summary,
        "equivalent": equivalent_summary,
        "ratio_median": _ratio(equivalent_summary, original_summary),
        "topk_equal": measurement.topk_equal,
        "mismatched_pairs": measurement.mismatched_pairs,
        "original_latencies_ms": [
            sample.latency_ms for sample in measurement.original
        ],
        "equivalent_latencies_ms": [
            sample.latency_ms for sample in measurement.equivalent
        ],
        "original_topk_ids": list(measurement.original[0].ids),
        "equivalent_topk_ids": list(measurement.equivalent[0].ids),
    }
    if ground_truth is not None:
        record["recall_at_k"] = {
            "original": float(
                np.median(
                    [_recall_at_k(sample.ids, ground_truth) for sample in measurement.original]
                )
            ),
            "equivalent": float(
                np.median(
                    [
                        _recall_at_k(sample.ids, ground_truth)
                        for sample in measurement.equivalent
                    ]
                )
            ),
        }
    return record


def run_benchmark(
    client: MilvusClient,
    cfg: Config,
    *,
    skip_flat: bool = False,
    log: Log = print,
) -> dict[str, object]:
    centers = generate_centers(cfg)
    queries = generate_queries(cfg, centers)

    ann_measurements: list[PairMeasurement] = []
    ann_queries: list[dict[str, object]] = []
    flat_queries: list[dict[str, object]] = []
    flat_measurements: list[PairMeasurement] = []

    for query_index, query in enumerate(queries):
        log(
            f"query {query_index + 1}/{len(queries)}: "
            f"ANN warmup={cfg.warmup_pairs} pairs, measured={cfg.measured_pairs} pairs"
        )
        ann_measurement = measure_query(
            client,
            cfg.ann_collection,
            query,
            cfg,
            pairs=cfg.measured_pairs,
            warmup_pairs=cfg.warmup_pairs,
            hnsw=True,
            order_seed=cfg.order_seed + query_index,
        )

        ground_truth: tuple[int, ...] | None = None
        if not skip_flat:
            log(
                f"query {query_index + 1}/{len(queries)}: "
                f"FLAT warmup={cfg.flat_warmup_pairs} pairs, measured={cfg.flat_pairs} pairs"
            )
            flat_measurement = measure_query(
                client,
                cfg.flat_collection,
                query,
                cfg,
                pairs=cfg.flat_pairs,
                warmup_pairs=cfg.flat_warmup_pairs,
                hnsw=False,
                order_seed=cfg.order_seed + 100 + query_index,
            )
            flat_measurements.append(flat_measurement)
            ground_truth = flat_measurement.original[0].ids
            flat_queries.append(_query_record(flat_measurement, ground_truth=None))

        ann_measurements.append(ann_measurement)
        ann_queries.append(_query_record(ann_measurement, ground_truth))

    ann_original = summarize(
        [sample for measurement in ann_measurements for sample in measurement.original]
    )
    ann_equivalent = summarize(
        [
            sample
            for measurement in ann_measurements
            for sample in measurement.equivalent
        ]
    )
    flat_original = summarize(
        [sample for measurement in flat_measurements for sample in measurement.original]
    )
    flat_equivalent = summarize(
        [sample for measurement in flat_measurements for sample in measurement.equivalent]
    )

    original_filter = ORIGINAL_FILTER
    equivalent_expr = equivalent_filter(cfg.true_predicate_repeats)
    result: dict[str, object] = {
        "filters": {
            "original": original_filter,
            "equivalent": equivalent_expr,
            "original_length": len(original_filter),
            "equivalent_length": len(equivalent_expr),
            "true_predicate_repeats": cfg.true_predicate_repeats,
        },
        "ann": {
            "collection": cfg.ann_collection,
            "original": ann_original,
            "equivalent": ann_equivalent,
            "ratio_median": _ratio(ann_equivalent, ann_original),
            "topk_equal": all(record["topk_equal"] for record in ann_queries),
            "queries": ann_queries,
        },
        "flat": None,
    }
    if not skip_flat:
        result["flat"] = {
            "collection": cfg.flat_collection,
            "original": flat_original,
            "equivalent": flat_equivalent,
            "ratio_median": _ratio(flat_equivalent, flat_original),
            "topk_equal": all(record["topk_equal"] for record in flat_queries),
            "queries": flat_queries,
        }
    return result
