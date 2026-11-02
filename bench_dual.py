"""Dual benchmark suite for both Issue #54195 and Issue #54043."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from pymilvus import MilvusClient

from .config import Config
from .dataset import generate_centers, generate_queries

Log = print


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
    return int(getattr(hit, "id"))


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


def measure_filter_pair(
    client: MilvusClient,
    collection_name: str,
    query: np.ndarray,
    original_filter: str,
    equivalent_filter: str,
    cfg: Config,
    *,
    pairs: int,
    warmup_pairs: int,
    hnsw: bool,
    order_seed: int,
) -> PairMeasurement:
    for round_index in range(warmup_pairs):
        first, second = (
            (original_filter, equivalent_filter)
            if round_index % 2 == 0
            else (equivalent_filter, original_filter)
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
                client, collection_name, query, equivalent_filter, cfg, hnsw=hnsw
            )
        else:
            equivalent_sample = search_once(
                client, collection_name, query, equivalent_filter, cfg, hnsw=hnsw
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


def run_benchmark_for_issue(
    client: MilvusClient,
    issue_name: str,
    original_filter: str,
    equivalent_filter: str,
    queries: np.ndarray,
    cfg: Config,
) -> dict:
    print(f"\n--- Running Benchmark for {issue_name} ---")
    print(f"Original Filter:   {original_filter}")
    print(f"Equivalent Filter: {equivalent_filter[:60]}... ({len(equivalent_filter)} bytes)")

    # ANN
    ann_orig_samples: list[Sample] = []
    ann_equiv_samples: list[Sample] = []
    ann_mismatches = 0

    # FLAT
    flat_orig_samples: list[Sample] = []
    flat_equiv_samples: list[Sample] = []
    flat_mismatches = 0

    for idx, q in enumerate(queries):
        # ANN
        meas_ann = measure_filter_pair(
            client,
            cfg.ann_collection,
            q,
            original_filter,
            equivalent_filter,
            cfg,
            pairs=cfg.measured_pairs,
            warmup_pairs=cfg.warmup_pairs,
            hnsw=True,
            order_seed=cfg.order_seed + idx * 10,
        )
        ann_orig_samples.extend(meas_ann.original)
        ann_equiv_samples.extend(meas_ann.equivalent)
        ann_mismatches += meas_ann.mismatched_pairs

        # FLAT
        meas_flat = measure_filter_pair(
            client,
            cfg.flat_collection,
            q,
            original_filter,
            equivalent_filter,
            cfg,
            pairs=cfg.flat_pairs,
            warmup_pairs=cfg.flat_warmup_pairs,
            hnsw=False,
            order_seed=cfg.order_seed + idx * 10 + 100,
        )
        flat_orig_samples.extend(meas_flat.original)
        flat_equiv_samples.extend(meas_flat.equivalent)
        flat_mismatches += meas_flat.mismatched_pairs
        print(f"  Query {idx + 1}/{len(queries)} completed")

    ann_orig_sum = summarize(ann_orig_samples)
    ann_equiv_sum = summarize(ann_equiv_samples)
    ann_ratio = ann_equiv_sum["median_ms"] / ann_orig_sum["median_ms"]

    flat_orig_sum = summarize(flat_orig_samples)
    flat_equiv_sum = summarize(flat_equiv_samples)
    flat_ratio = flat_equiv_sum["median_ms"] / flat_orig_sum["median_ms"]

    print(f"  HNSW p50: original={ann_orig_sum['median_ms']:.4f} ms, equivalent={ann_equiv_sum['median_ms']:.4f} ms, ratio={ann_ratio:.3f}x, topk_equal={ann_mismatches == 0}")
    print(f"  FLAT p50: original={flat_orig_sum['median_ms']:.4f} ms, equivalent={flat_equiv_sum['median_ms']:.4f} ms, ratio={flat_ratio:.3f}x, topk_equal={flat_mismatches == 0}")

    return {
        "issue": issue_name,
        "original_filter": original_filter,
        "equivalent_filter_len": len(equivalent_filter),
        "ann": {
            "original": ann_orig_sum,
            "equivalent": ann_equiv_sum,
            "ratio": ann_ratio,
            "topk_equal": ann_mismatches == 0,
        },
        "flat": {
            "original": flat_orig_sum,
            "equivalent": flat_equiv_sum,
            "ratio": flat_ratio,
            "topk_equal": flat_mismatches == 0,
        },
    }


def main(output_file: str = "dual_bench_result.json"):
    cfg = Config()
    client = MilvusClient(cfg.uri)
    server_version = client.get_server_version()
    print(f"Connected to Milvus: {server_version}")

    centers = generate_centers(cfg)
    queries = generate_queries(cfg, centers)

    # 1. Issue #54195 (AND-TRUE identity)
    f54195_orig = "category == 1"
    f54195_true_pred = "(age >= 1) and (age <= 1000)"
    f54195_equiv = f"({f54195_orig}) and " + " and ".join(f"({f54195_true_pred})" for _ in range(128))

    res_54195 = run_benchmark_for_issue(
        client, "Issue #54195 (AND-TRUE 128x)", f54195_orig, f54195_equiv, queries, cfg
    )

    # 2. Issue #54043 (OR-idempotent)
    f54043_orig = "(parity == 'even') and (age <= 500)"
    f54043_equiv = " or ".join(f"({f54043_orig})" for _ in range(128))

    res_54043 = run_benchmark_for_issue(
        client, "Issue #54043 (OR-idempotent 128x)", f54043_orig, f54043_equiv, queries, cfg
    )

    full_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_version": server_version,
        "benchmarks": {
            "54195": res_54195,
            "54043": res_54043,
        },
    }

    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved benchmark results to {out_p}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "repro/results/dual_bench_current.json"
    main(out)
