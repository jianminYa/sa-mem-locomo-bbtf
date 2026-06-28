# SA-Mem-Research

Code for VLDB Research Track. This README currently only documents the LoCoMo running scripts.

Run all commands from the project root:

```bash
cd /data/wjl/SA-Mem-Research
```

Set API configuration by environment variables or CLI flags:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"  # optional
```

If memory blocks have not been built yet, run:

```bash
python build_stage_locomo.py \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --limit-conversations -1
```

The following stages assume the build output is under `out/locomo/`.

## Retrieval

LoCoMo retrieval uses:

```bash
python retrieval/retrieve_stage_enhanced_locomo.py
```

### Baseline

Baseline is the plain vector retrieval setting.

```bash
python retrieval/retrieve_stage_enhanced_locomo.py \
  --mode baseline \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --overwrite
```

Output:

```text
out/locomo/retrieval_baseline.jsonl
out/locomo/retrieval_baseline.csv
```

### Time Filtering

Time filtering corresponds to the code's `enhanced` mode. It enables query parsing and temporal/metadata filtering.

```bash
python retrieval/retrieve_stage_enhanced_locomo.py \
  --mode enhanced \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --overwrite
```

Output:

```text
out/locomo/retrieval_enhanced.jsonl
out/locomo/retrieval_enhanced.csv
```

### Time Filtering + Graph Expansion

This uses `enhanced` mode and additionally enables graph expansion.

```bash
python retrieval/retrieve_stage_enhanced_locomo.py \
  --mode enhanced \
  --raw-data-file /path/to/locomo.json \
  --run-id locomo \
  --graph-expand \
  --graph-min-score 0.7 \
  --graph-limit 200 \
  --graph-hops 1 \
  --overwrite \
  --output-suffix graph
```

Output:

```text
out/locomo/retrieval_enhanced_graph.jsonl
out/locomo/retrieval_enhanced_graph.csv
```

Useful retrieval options:

- `--raw-data-file`: LoCoMo data file.
- `--run-id`: output directory name under `out/`.
- `--final-content-file`: manually specify memory blocks if not using `out/<run-id>/final_boxes_content.jsonl`.
- `--output-dir`: manually specify output directory.
- `--output-suffix`: append a suffix to output filenames.
- `--limit-conversations`: run only part of the dataset.
- `--overwrite`: remove existing retrieval output before writing.

## Generation

Generate answers with the retrieval file you want to test.

### From Baseline Retrieval

```bash
python generate_stage_locomo.py \
  --run-id locomo \
  --raw-data-file /path/to/locomo.json \
  --retrieval-file out/locomo/retrieval_baseline.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --output-suffix baseline
```

### From Time Filtering Retrieval

```bash
python generate_stage_locomo.py \
  --run-id locomo \
  --raw-data-file /path/to/locomo.json \
  --retrieval-file out/locomo/retrieval_enhanced.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --output-suffix time_filtering
```

### From Time Filtering + Graph Expansion Retrieval

```bash
python generate_stage_locomo.py \
  --run-id locomo \
  --raw-data-file /path/to/locomo.json \
  --retrieval-file out/locomo/retrieval_enhanced_graph.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --use-graph-context \
  --output-suffix graph
```

Useful generation options:

- `--answer-topn`: number of retrieved memory blocks used for answering.
- `--text-modes`: usually `content`; also supports `event`, `content_trace_event`, and `trace_event`.
- `--provider openai|ollama`: choose generation provider.
- `--llm-model`, `--api-key`, `--base-url`: model/API overrides.
- `--use-graph-context`: inject graph context when using graph-expanded retrieval results.
- `--graph-context-categories`: only inject graph context for selected LoCoMo categories, e.g. `3,4`.

## Evaluation

Evaluate any generated LoCoMo result file:

```bash
python evaluate_locomo.py \
  --run-id locomo \
  --generation-file out/locomo/generation_results_locomo_time_filtering.jsonl
```

For LLM-as-judge evaluation:

```bash
python evaluate_locomo.py \
  --run-id locomo \
  --generation-file out/locomo/generation_results_locomo_time_filtering.jsonl \
  --use-llm-judge
```

Useful evaluation options:

- `--generation-file`: generation result JSONL to evaluate.
- `--sample-size`: evaluate only the first N samples.
- `--eval-output`: custom detailed evaluation output path.
- `--summary-output`: custom summary output path.

## LoCoMo B/B+TF 复现预览（中文）

本小节记录我们基于公开代码对 SA-Mem LoCoMo B/B+TF 路径的复现结果。详细说明见：

- `docs/locomo_b_btf_reproduction.md`
- `docs/locomo_b_btf_metrics_summary.md`

### 核心观察

| 方面 | 结果 | 说明 |
|------|------|------|
| QA（B） | 接近论文 | F1=0.5133，BLEU=0.3892；论文 SA-Mem overall 约 0.5203/0.3908 |
| QA（B+TF） | 未对齐 | F1=0.4879，BLEU=0.3692，低于 B |
| Evidence（B） | 接近论文 | Hit@5=0.8201，Recall@5=0.7580；论文 Table 5 参考值 0.8162/0.7475 |
| Evidence（B+TF） | 未对齐 | Hit@5=0.7708，Recall@5=0.7065，低于 B |
| Latency（B core） | 接近论文 | core search p50=0.102s，p95=0.144s |
| Latency（B+TF 全量） | 效果有限 | core search p50=0.108s；平均候选池 93.36 -> 87.33 |
| Latency（B+TF 触发时间约束子集） | 有局部收益 | category-2 POINT/RANGE 子集 core p50=0.0085s；候选池 96.00 -> 33.65 |

### 实验配置

| 参数 | 值 |
|------|----|
| Dataset | LoCoMo10，10 conversations，1540 个非 category-5 QA |
| LLM | `gpt-4o-mini` |
| Embedding | `text-embedding-3-small` |
| Retrieval top-k | 5 |
| Generation top-n | 5 |
| Text mode | `content` |
| Graph | disabled |
| axis-mode | auto |

### 重要口径说明

B+TF 当前公开代码路径会使用 `QueryParser` 生成 `rewritten_query`，并用 rewritten-query embedding 做语义排序。论文没有给出 exact rewriting prompt，也没有单独拆分 query rewriting 与 temporal filtering 的 ablation。因此，当前 B+TF 结果应理解为“公开 enhanced retrieval 路径 + rewritten-query embedding 启用”的结果。

后续如果要进一步定位差异，可以补一个 `B+TF-original-query` ablation：保留 query parsing 和 temporal-index candidate pruning，但语义排序时使用原始 question embedding。

### 不建议提交到上游的内容

上游 PR 建议只提交 `docs/` 下的精简报告和必要脚本/补丁，不提交完整运行目录：

- 不提交 `.env`
- 不提交 `dataset/locomo10.json`
- 不提交完整 `out/locomo_b_btf_*`
- 不提交 vector cache
- 不提交 raw retrieval/generation/evaluation JSONL
- 不提交 token stream 或 trace log
