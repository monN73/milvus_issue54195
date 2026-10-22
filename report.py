"""Console, JSON and Markdown reporting."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from .config import Config


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _physical_memory_gib() -> float | None:
    try:
        return round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 2
        )
    except (AttributeError, OSError, ValueError):
        return None


def gather_environment(client: MilvusClient, cfg: Config) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "milvus_server_version": client.get_server_version(),
        "pymilvus_version": _package_version("pymilvus"),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "physical_memory_gib": _physical_memory_gib(),
        "uri": cfg.uri,
    }


def config_snapshot(cfg: Config) -> dict[str, object]:
    snapshot = asdict(cfg)
    snapshot.pop("token", None)
    return snapshot


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(results: dict[str, object]) -> str:
    environment = results["environment"]
    ann = results["ann"]
    flat = results.get("flat")
    verify = results.get("verify")

    lines = [
        "# Issue 54195 benchmark result",
        "",
        f"- Generated: `{environment['timestamp_utc']}`",
        f"- Milvus: `{environment['milvus_server_version']}`",
        f"- pymilvus: `{environment['pymilvus_version']}`",
        f"- Python: `{environment['python_version']}`",
        f"- Platform: `{environment['platform']}`",
        f"- CPU count: `{environment['cpu_count']}`",
        f"- Physical memory (GiB): `{environment['physical_memory_gib']}`",
        "",
    ]

    if verify:
        lines.extend(
            [
                "## Semantic checks",
                "",
                "| Check | Value |",
                "|---|---:|",
                f"| Eligible entities (original) | {verify['eligible']['original_count']:,} |",
                f"| Eligible entities (equivalent) | {verify['eligible']['equivalent_count']:,} |",
                f"| Eligible sets equal | {verify['eligible']['equal']} |",
                f"| ANN Top-K equal | {verify['ann_topk_equal']} |",
                f"| FLAT Top-K equal | {verify['flat_topk_equal']} |",
                f"| Recall@{verify['recall_at_k']['k']} (original) | {_fmt(verify['recall_at_k']['original'])} |",
                f"| Recall@{verify['recall_at_k']['k']} (equivalent) | {_fmt(verify['recall_at_k']['equivalent'])} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Latency (client end-to-end, ms)",
            "",
            "| Collection | Original p50 | Equivalent p50 | Ratio | Top-K equal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, payload in (("HNSW", ann), ("FLAT", flat)):
        if payload is None:
            lines.append(f"| {label} | skipped | skipped | - | - |")
            continue
        lines.append(
            f"| {label} "
            f"| {_fmt(payload['original'].get('median_ms'))} "
            f"| {_fmt(payload['equivalent'].get('median_ms'))} "
            f"| {_fmt(payload['ratio_median'], 3)}x "
            f"| {payload['topk_equal']} |"
        )

    lines.extend(
        [
            "",
            "## Per-query p50",
            "",
            "| Query | HNSW original | HNSW equivalent | HNSW ratio | FLAT original | FLAT equivalent | FLAT ratio |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ann_queries = ann["queries"]
    flat_queries = flat["queries"] if flat else []
    for index, ann_query in enumerate(ann_queries):
        flat_query = flat_queries[index] if index < len(flat_queries) else None
        lines.append(
            f"| {index + 1} "
            f"| {_fmt(ann_query['original'].get('median_ms'))} "
            f"| {_fmt(ann_query['equivalent'].get('median_ms'))} "
            f"| {_fmt(ann_query['ratio_median'], 3)}x "
            f"| {_fmt(flat_query['original'].get('median_ms')) if flat_query else '-'} "
            f"| {_fmt(flat_query['equivalent'].get('median_ms')) if flat_query else '-'} "
            f"| {_fmt(flat_query['ratio_median'], 3) if flat_query else '-'} |"
        )

    filters = results["filters"]
    lines.extend(
        [
            "",
            "## Filters",
            "",
            f"- Original length: {filters['original_length']} bytes",
            f"- Equivalent length: {filters['equivalent_length']} bytes",
            f"- True predicate repetitions: {filters['true_predicate_repeats']}",
            "",
        ]
    )
    return "\n".join(lines)


def save_results(results: dict[str, object], output_dir: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = Path(output_dir) / timestamp
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target / "summary.md").write_text(render_markdown(results), encoding="utf-8")
    return target


def print_summary(results: dict[str, object]) -> None:
    environment = results["environment"]
    ann = results["ann"]
    flat = results.get("flat")
    print(
        f"Milvus {environment['milvus_server_version']} | "
        f"pymilvus {environment['pymilvus_version']} | "
        f"{environment['cpu_count']} CPUs | "
        f"{environment['physical_memory_gib']} GiB"
    )
    print(
        "HNSW p50: "
        f"original={_fmt(ann['original'].get('median_ms'))} ms, "
        f"equivalent={_fmt(ann['equivalent'].get('median_ms'))} ms, "
        f"ratio={_fmt(ann['ratio_median'], 3)}x, "
        f"Top-K equal={ann['topk_equal']}"
    )
    if flat:
        print(
            "FLAT p50: "
            f"original={_fmt(flat['original'].get('median_ms'))} ms, "
            f"equivalent={_fmt(flat['equivalent'].get('median_ms'))} ms, "
            f"ratio={_fmt(flat['ratio_median'], 3)}x, "
            f"Top-K equal={flat['topk_equal']}"
        )
