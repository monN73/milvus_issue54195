"""Semantic equivalence checks for issue 54195."""

from __future__ import annotations

from typing import Callable

import numpy as np
from pymilvus import MilvusClient

from .benchmark import search_once
from .config import Config
from .dataset import generate_centers, generate_queries
from .filters import ORIGINAL_FILTER, equivalent_filter

Log = Callable[[str], None]


def eligible_ids(
    client: MilvusClient,
    cfg: Config,
    collection_name: str,
    filter_expr: str,
    log: Log = print,
) -> set[int]:
    """Return every primary key selected by ``filter_expr``."""

    rows = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=["id"],
        limit=cfg.query_window,
        consistency_level="Strong",
        timeout=cfg.timeout,
    )
    ids = {int(row["id"]) for row in rows}
    if len(rows) < cfg.query_window:
        return ids

    log(
        f"  {collection_name}: single query hit the {cfg.query_window:,}-row window; "
        "switching to query_iterator"
    )
    ids = set()
    iterator_kwargs = {
        "collection_name": collection_name,
        "batch_size": 1_000,
        "filter": filter_expr,
        "output_fields": ["id"],
    }
    try:
        iterator = client.query_iterator(
            **iterator_kwargs, consistency_level="Strong"
        )
    except TypeError:
        # Some pymilvus 2.6 builds do not forward consistency_level through
        # query_iterator; the collection itself still uses Strong consistency.
        iterator = client.query_iterator(**iterator_kwargs)
    try:
        while True:
            batch = iterator.next()
            if not batch:
                break
            ids.update(int(row["id"]) for row in batch)
    finally:
        iterator.close()
    return ids


def _recall_at_k(ids: tuple[int, ...], ground_truth: tuple[int, ...]) -> float:
    if not ground_truth:
        return 0.0
    return len(set(ids) & set(ground_truth)) / len(ground_truth)


def run_verify(client: MilvusClient, cfg: Config, log: Log = print) -> dict[str, object]:
    original_filter = ORIGINAL_FILTER
    equivalent_expr = equivalent_filter(cfg.true_predicate_repeats)

    log("enumerating eligible primary keys")
    original_ids = eligible_ids(
        client, cfg, cfg.ann_collection, original_filter, log=log
    )
    equivalent_ids = eligible_ids(
        client, cfg, cfg.ann_collection, equivalent_expr, log=log
    )
    eligible_equal = original_ids == equivalent_ids
    if not eligible_equal:
        only_original = sorted(original_ids - equivalent_ids)[:10]
        only_equivalent = sorted(equivalent_ids - original_ids)[:10]
        raise RuntimeError(
            "eligible ID sets differ: "
            f"original={len(original_ids)}, equivalent={len(equivalent_ids)}, "
            f"example only-original={only_original}, "
            f"example only-equivalent={only_equivalent}"
        )

    centers = generate_centers(cfg)
    queries = generate_queries(cfg, centers)

    log("checking Top-K equality and Recall@K")
    flat_original: list[tuple[int, ...]] = []
    flat_equivalent: list[tuple[int, ...]] = []
    ann_original: list[tuple[int, ...]] = []
    ann_equivalent: list[tuple[int, ...]] = []

    for query in queries:
        flat_original.append(
            search_once(
                client,
                cfg.flat_collection,
                query,
                original_filter,
                cfg,
                hnsw=False,
            ).ids
        )
        flat_equivalent.append(
            search_once(
                client,
                cfg.flat_collection,
                query,
                equivalent_expr,
                cfg,
                hnsw=False,
            ).ids
        )
        ann_original.append(
            search_once(
                client,
                cfg.ann_collection,
                query,
                original_filter,
                cfg,
                hnsw=True,
            ).ids
        )
        ann_equivalent.append(
            search_once(
                client,
                cfg.ann_collection,
                query,
                equivalent_expr,
                cfg,
                hnsw=True,
            ).ids
        )

    flat_topk_equal = flat_original == flat_equivalent
    ann_topk_equal = ann_original == ann_equivalent
    ann_recall_original = float(
        np.median(
            [
                _recall_at_k(ids, ground_truth)
                for ids, ground_truth in zip(ann_original, flat_original, strict=True)
            ]
        )
    )
    ann_recall_equivalent = float(
        np.median(
            [
                _recall_at_k(ids, ground_truth)
                for ids, ground_truth in zip(ann_equivalent, flat_original, strict=True)
            ]
        )
    )

    result: dict[str, object] = {
        "eligible": {
            "original_count": len(original_ids),
            "equivalent_count": len(equivalent_ids),
            "equal": eligible_equal,
        },
        "ann_topk_equal": ann_topk_equal,
        "flat_topk_equal": flat_topk_equal,
        "recall_at_k": {
            "k": cfg.top_k,
            "original": ann_recall_original,
            "equivalent": ann_recall_equivalent,
        },
        "query_count": len(queries),
    }
    if not flat_topk_equal or not ann_topk_equal:
        raise RuntimeError(
            "Top-K mismatch between semantically equivalent filters: "
            f"ANN={ann_topk_equal}, FLAT={flat_topk_equal}"
        )
    return result
