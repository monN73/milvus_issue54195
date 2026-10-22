"""Command line entry point: ``python -m repro <command>``."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .benchmark import run_benchmark
from .config import Config
from .report import (
    config_snapshot,
    gather_environment,
    print_summary,
    save_results,
)
from .store import connect, drop_collections, prepare_collections
from .verify import run_verify


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--uri", help="Milvus endpoint (default: MILVUS_URI or http://127.0.0.1:19530)")
    parser.add_argument("--token", help="Milvus token (default: MILVUS_TOKEN)")
    parser.add_argument("--ann-collection", dest="ann_collection")
    parser.add_argument("--flat-collection", dest="flat_collection")
    parser.add_argument("--num-rows", dest="num_rows", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--repeats", dest="repeats", type=int, help="AND-TRUE repetitions")
    parser.add_argument("--query-count", dest="query_count", type=int)
    parser.add_argument("--warmup-pairs", dest="warmup_pairs", type=int)
    parser.add_argument("--measured-pairs", dest="measured_pairs", type=int)
    parser.add_argument("--flat-warmup-pairs", dest="flat_warmup_pairs", type=int)
    parser.add_argument("--flat-pairs", dest="flat_pairs", type=int)
    parser.add_argument("--insert-batch-size", dest="insert_batch_size", type=int)
    parser.add_argument("--timeout", type=float, help="per-operation timeout in seconds")
    parser.add_argument("--output-dir", default="repro/results")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m repro",
        description="Reproduce milvus-io/milvus issue 54195 (F AND TRUE latency anomaly).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="create collections, indexes and load data")
    _add_common_arguments(setup)
    setup.add_argument(
        "--drop-existing",
        action="store_true",
        help="drop the two reproduction collections before recreating them",
    )

    verify = subparsers.add_parser(
        "verify", help="check eligible ID sets, Top-K equality and Recall"
    )
    _add_common_arguments(verify)

    bench = subparsers.add_parser("bench", help="run the balanced-order latency benchmark")
    _add_common_arguments(bench)
    bench.add_argument(
        "--skip-flat",
        action="store_true",
        help="skip FLAT latency measurements (no exact-control numbers)",
    )

    all_in_one = subparsers.add_parser(
        "all", help="setup, verify and benchmark in one process"
    )
    _add_common_arguments(all_in_one)
    all_in_one.add_argument(
        "--drop-existing",
        action="store_true",
        help="drop the two reproduction collections before recreating them",
    )
    all_in_one.add_argument(
        "--skip-flat",
        action="store_true",
        help="skip FLAT latency measurements (verification still uses FLAT)",
    )

    drop = subparsers.add_parser("drop", help="drop the two reproduction collections")
    _add_common_arguments(drop)
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    return Config.from_env().with_overrides(
        uri=args.uri,
        token=args.token,
        ann_collection=args.ann_collection,
        flat_collection=args.flat_collection,
        num_rows=args.num_rows,
        dim=args.dim,
        true_predicate_repeats=args.repeats,
        query_count=args.query_count,
        warmup_pairs=args.warmup_pairs,
        measured_pairs=args.measured_pairs,
        flat_warmup_pairs=args.flat_warmup_pairs,
        flat_pairs=args.flat_pairs,
        insert_batch_size=args.insert_batch_size,
        timeout=args.timeout,
    )


def _result_payload(
    client,
    cfg: Config,
    bench_result: dict[str, object],
    verify_result: dict[str, object] | None,
) -> dict[str, object]:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": gather_environment(client, cfg),
        "config": config_snapshot(cfg),
        "verify": verify_result,
    }
    payload.update(bench_result)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _config_from_args(args)
    client = connect(cfg)
    try:
        if args.command == "drop":
            drop_collections(client, cfg)
            return 0

        if args.command == "setup":
            prepare_collections(
                client, cfg, drop_existing=args.drop_existing, log=print
            )
            return 0

        if args.command == "verify":
            result = run_verify(client, cfg)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        if args.command == "bench":
            bench_result = run_benchmark(client, cfg, skip_flat=args.skip_flat)
            payload = _result_payload(client, cfg, bench_result, verify_result=None)
            target = save_results(payload, args.output_dir)
            print_summary(payload)
            print(f"saved: {target / 'results.json'}")
            print(f"saved: {target / 'summary.md'}")
            return 0

        if args.command == "all":
            prepare_collections(
                client, cfg, drop_existing=args.drop_existing, log=print
            )
            verify_result = run_verify(client, cfg)
            bench_result = run_benchmark(client, cfg, skip_flat=args.skip_flat)
            payload = _result_payload(client, cfg, bench_result, verify_result)
            target = save_results(payload, args.output_dir)
            print_summary(payload)
            print(f"saved: {target / 'results.json'}")
            print(f"saved: {target / 'summary.md'}")
            return 0

        parser.error(f"unknown command: {args.command}")
        return 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
