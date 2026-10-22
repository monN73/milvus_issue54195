# Issue 54195 复现代码

这个目录是 `issue-54195-repro-and-optimization.md` 的配套可执行实现，用来复现：

```text
category == 1
```

与

```text
category == 1 AND (age >= 1 AND age <= 1000) × 128
```

在 Milvus v3.0.0、HNSW/COSINE 上的端到端搜索延迟差异，并用 FLAT/COSINE 作为 exact control。

## 快速开始

```bash
cd /path/to/milvus_issue#54195

python3 -m venv .venv-54195
source .venv-54195/bin/activate
pip install -r repro/requirements.txt

docker compose -f repro/docker-compose.v3.0.0.yml up -d
until curl -sf http://127.0.0.1:9091/healthz >/dev/null; do sleep 2; done

python -m repro setup --drop-existing
python -m repro verify
python -m repro bench
```

也可以一次性执行 setup、verify 和 bench：

```bash
python -m repro all --drop-existing
```

运行结束后，结果会保存在 `repro/results/<timestamp>/`：

- `results.json`：完整配置、环境、eligible 集合、逐查询统计、原始延迟样本和汇总；
- `summary.md`：可直接贴到 issue 或 PR 的结果表。

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | issue 使用的固定参数、随机种子、集合名和超时 |
| `filters.py` | 原始过滤表达式和 AND-TRUE 等价表达式 |
| `dataset.py` | 确定性向量、标量字段和查询向量生成 |
| `store.py` | 建集合、建索引、分批插入、flush、load |
| `verify.py` | eligible ID 集合、ANN/FLAT Top-K 一致性和 Recall |
| `benchmark.py` | warmup、balanced-order 配对搜索、p50/p95 统计 |
| `report.py` | 环境信息、JSON 和 Markdown 报告 |
| `__main__.py` | `setup` / `verify` / `bench` / `all` / `drop` 命令 |
| `docker-compose.v3.0.0.yml` | 固定 `milvusdb/milvus:v3.0.0` 的 standalone 环境 |

## 默认测量口径

- 200,000 条 128 维向量，100 个 tenant，correlation `0.85`；
- HNSW `M=16`、`efConstruction=128`，查询 `ef=64`；
- 4 个确定性查询向量；
- 每个查询先做 4 组 warmup；
- 再做 10 组 measured pair，恰好 5 组 original→equivalent、5 组 equivalent→original；
- FLAT 默认每个查询 2 组 warmup + 2 组 measured pair，与 issue 描述一致；
- 所有查询使用 `top_k=10`、COSINE 和 Strong consistency；
- 延迟使用客户端 `time.perf_counter()` 端到端计时，汇总使用 p50。

## 常用参数

所有参数在 `setup`、`verify`、`bench`、`all` 上都可覆盖：

```bash
python -m repro bench \
  --uri http://127.0.0.1:19530 \
  --repeats 128 \
  --measured-pairs 10 \
  --flat-pairs 2 \
  --output-dir repro/results
```

环境变量也可以提供连接信息：

```bash
export MILVUS_URI=http://127.0.0.1:19530
export MILVUS_TOKEN=root:Milvus   # 仅在开启鉴权时需要
```

清理复现用集合：

```bash
python -m repro drop
```

## 注意事项

- `pymilvus` 固定为 issue 使用的 `2.6.9`。版本不同，客户端序列化和请求路径可能不同。
- `setup --drop-existing` 会删除并重建两个复现集合，不要对生产集合使用。
- 默认数据集约占 100 MiB 原始向量内存；插入时每批还会临时构造 Python 行对象，8 GiB 主机建议保持默认 `--insert-batch-size` 或更小。
- 如果要和 issue 的 2.706x 直接比较，必须保持 8 CPU、32 GiB、v3.0.0 镜像和默认参数。资源不同只能做同机改动前后对比。
- `bench --skip-flat` 只跳过 FLAT 延迟测量；`verify`（或 `all`）仍然需要 FLAT 集合计算 Recall。
