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

## LoCoMo B/B+TF Reproduction Notes

This branch documents a lightweight reproduction of the non-graph LoCoMo B/B+TF path. See:

- `docs/locomo_b_btf_reproduction.md`
- `docs/locomo_b_btf_metrics_summary.md`
- `docs/locomo_b_btf_metrics_summary.json`

### Key Results

| Aspect | Result | Details |
|--------|--------|---------|
| QA (B) | Close to paper | F1=0.5133, BLEU=0.3892; paper SA-Mem overall is 0.5203/0.3908 |
| QA (B+TF) | Not reproduced | F1=0.4879, BLEU=0.3692; lower than B |
| Evidence (B) | Close to paper | Hit@5=0.8201, Recall@5=0.7580; paper Table 5 reference is 0.8162/0.7475 |
| Evidence (B+TF) | Not reproduced | Hit@5=0.7708, Recall@5=0.7065; lower than B |
| Latency (B core) | Close to paper | Core search p50=0.102s, p95=0.144s |
| Latency (B+TF all queries) | Limited full-run gain | Core search p50=0.108s; average candidate pool 93.36 -> 87.33 |
| Latency (B+TF triggered temporal subset) | Local gain | Category-2 POINT/RANGE subset core p50=0.0085s; pool 96.00 -> 33.65 |

### Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo10, 10 conversations, 1540 non-category-5 QA pairs |
| LLM | `gpt-4o-mini` |
| Embedding | `text-embedding-3-small` |
| Retrieval top-k | 5 |
| Generation top-n | 5 |
| Text mode | `content` |
| Graph | disabled |
| axis-mode | auto |

### Notes

B+TF in the current public code path uses `QueryParser` to produce `rewritten_query`, and enhanced retrieval uses the rewritten-query embedding for semantic ranking. The paper does not provide the exact rewrite prompt or an ablation separating query rewriting from temporal filtering. A useful follow-up is `B+TF-original-query`: keep query parsing and temporal-index candidate pruning, but rank with the original question embedding.

This branch intentionally includes aggregate metrics and reusable scripts, not full raw run artifacts. Raw local outputs were generated under `out/locomo_b_btf_recheck_20260624/` and can be regenerated with the provided scripts.
