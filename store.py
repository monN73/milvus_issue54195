"""Milvus collection setup, insert and load helpers."""

from __future__ import annotations

import time
from typing import Callable

from pymilvus import DataType, MilvusClient

from .config import Config
from .dataset import Dataset, generate_dataset, iter_row_batches

Log = Callable[[str], None]

INT_FIELDS = ("tenant", "level", "category", "category5", "age")


def connect(cfg: Config) -> MilvusClient:
    if cfg.token:
        client = MilvusClient(uri=cfg.uri, token=cfg.token)
    else:
        client = MilvusClient(uri=cfg.uri)
    # Fail fast when the endpoint is unreachable.
    client.get_server_version()
    return client


def create_collection(
    client: MilvusClient,
    collection_name: str,
    index_type: str,
    cfg: Config,
) -> None:
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=cfg.dim)
    for field in INT_FIELDS:
        schema.add_field(field_name=field, datatype=DataType.INT64)
    schema.add_field(field_name="parity", datatype=DataType.VARCHAR, max_length=8)

    index_params = client.prepare_index_params()
    if index_type == "HNSW":
        vector_params: dict[str, int] = {
            "M": cfg.hnsw_m,
            "efConstruction": cfg.hnsw_ef_construction,
        }
    elif index_type == "FLAT":
        vector_params = {}
    else:
        raise ValueError(f"unsupported vector index type: {index_type}")

    index_params.add_index(
        field_name="vector",
        index_type=index_type,
        metric_type="COSINE",
        params=vector_params,
    )
    for field in INT_FIELDS:
        index_params.add_index(field_name=field, index_type="INVERTED")
    index_params.add_index(field_name="parity", index_type="TRIE")

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong",
    )


def insert_dataset(
    client: MilvusClient,
    collection_names: tuple[str, ...],
    dataset: Dataset,
    cfg: Config,
    log: Log = print,
) -> None:
    total_batches = (dataset.num_rows + cfg.insert_batch_size - 1) // cfg.insert_batch_size
    for batch_number, rows in enumerate(
        iter_row_batches(dataset, cfg.insert_batch_size), start=1
    ):
        inserted_rows = min(batch_number * cfg.insert_batch_size, dataset.num_rows)
        for collection_name in collection_names:
            client.insert(
                collection_name=collection_name,
                data=rows,
                timeout=cfg.timeout,
            )
        log(
            f"  inserted batch {batch_number}/{total_batches} "
            f"({inserted_rows:,}/{dataset.num_rows:,} rows) "
            f"into {len(collection_names)} collections"
        )


def flush_collection(client: MilvusClient, collection_name: str, cfg: Config) -> None:
    client.flush(collection_name=collection_name, timeout=cfg.timeout)


def load_and_wait(
    client: MilvusClient, collection_name: str, cfg: Config, log: Log = print
) -> None:
    log(f"  loading {collection_name} (timeout={cfg.timeout:.0f}s)")
    client.load_collection(collection_name=collection_name, timeout=cfg.timeout)

    deadline = time.monotonic() + cfg.timeout
    while time.monotonic() < deadline:
        state = client.get_load_state(collection_name=collection_name)
        state_value = state.get("state") if isinstance(state, dict) else state
        label = getattr(state_value, "name", str(state_value))
        if label == "Loaded" or state_value == 3:
            log(f"  {collection_name} is loaded")
            return
        time.sleep(1.0)
    raise TimeoutError(
        f"{collection_name} did not reach Loaded state within {cfg.timeout:.0f}s"
    )


def prepare_collections(
    client: MilvusClient,
    cfg: Config,
    drop_existing: bool = False,
    log: Log = print,
) -> None:
    names = (cfg.ann_collection, cfg.flat_collection)
    for name in names:
        exists = client.has_collection(collection_name=name)
        if exists and not drop_existing:
            raise RuntimeError(
                f"collection {name!r} already exists; rerun with --drop-existing "
                "or choose a different collection name"
            )
        if exists:
            log(f"dropping existing collection {name}")
            client.drop_collection(collection_name=name)

    log("creating collections and indexes")
    create_collection(client, cfg.ann_collection, "HNSW", cfg)
    create_collection(client, cfg.flat_collection, "FLAT", cfg)

    log(f"generating {cfg.num_rows:,} rows (seed={cfg.data_seed})")
    dataset = generate_dataset(cfg)
    insert_dataset(client, names, dataset, cfg, log=log)

    for name in names:
        log(f"flushing {name}")
        flush_collection(client, name, cfg)
        load_and_wait(client, name, cfg, log=log)


def drop_collections(client: MilvusClient, cfg: Config, log: Log = print) -> None:
    for name in (cfg.ann_collection, cfg.flat_collection):
        if client.has_collection(collection_name=name):
            log(f"dropping {name}")
            client.drop_collection(collection_name=name)
        else:
            log(f"{name} does not exist")
