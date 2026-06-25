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

## LoCoMo B/B+TF Reproduction

This section documents our reproduction of the SA-Mem paper's LoCoMo B/B+TF experiments.

### Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | LoCoMo10 (10 conversations, 1540 non-cat5 QA pairs) |
| LLM | gpt-4o-mini |
| Embedding | text-embedding-3-small |
| Retrieval top-k | 5 |
| Generation top-n | 5 |
| Text mode | content |
| Graph | disabled |

### QA Results vs Paper Table 1

| Method | Our F1 | Paper F1 | Our BLEU | Paper BLEU |
|--------|--------|----------|----------|------------|
| B | 0.5140 | ~0.52 | 0.3894 | ~0.39 |
| B+TF | 0.5035 | ~0.51 | 0.3792 | ~0.38 |

**Note**: Our results are close but not identical to the paper due to API randomness and model version differences.

### Retrieval Latency (warm-cache)

| Method | Parse | Filter | Search No Parse | Total With Parse | p50 | p95 |
|--------|-------|--------|-----------------|------------------|-----|-----|
| B | 0 | 0 | 0.100s | 0.100s | 0.101s | 0.142s |
| B+TF | 1.116s | 0.002s | 0.103s | 1.219s | 1.381s | 2.059s |

**Latency breakdown**:
- **With online parse**: includes LLM call for query parsing (~1.1s)
- **No-parse (warm)**: excludes parse; vector cache pre-loaded per user
- **Paper-like search latency**: comparable to "Core No Parse" in paper

### Evidence Metrics (top-5)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B (all) | 0.6592 | 0.5608 | 0.8201 | 0.7580 |
| B+TF (all) | 0.6525 | 0.5531 | 0.8045 | 0.7416 |
| B+TF (temporal) | 0.4944 | 0.4634 | 0.6058 | 0.5853 |

**Paper reference** (Table 5, SA-Mem ~1200 tokens): C-MRR 0.5510, Hit@k 0.8162, Recall@k 0.7475

**Note**: 
- Our top-5 is not equivalent to the paper's controlled token budget.
- Complete-MRR = M/rank_max if all gold memories in top-k, else 0.
- First-RR = reciprocal rank of first relevant item.

### Code Changes

1. **Bi-temporal filtering**: Added `dispatch_temporal_filter` and `axis_mode` parameter
2. **Warm-cache**: EmbeddingStore created once per user, not per QA
3. **Fine-grained latency**: parse/filter/store_init/query_vector/block_vector/cosine/sort/flush
4. **Complete-MRR**: Paper-style evidence metric
5. **Fallback**: Temporal filtering returns 0 → fallback to full pool

### Output Directories

| Directory | Description |
|-----------|-------------|
| `out/locomo_b_btf_full/` | Original run (old retrieval latency) |
| `out/locomo_b_btf_fix_latency_20260623/` | Fixed fallback + old latency |
| `out/locomo_b_btf_warm_latency_20260623/` | Warm-cache + bi-temporal + Complete-MRR |

### Running the Experiments

```bash
# Full run (with build)
RUN_ID=locomo_b_btf_full bash scripts/run_locomo_b_btf.sh

# Skip build (reuse existing memory blocks)
SKIP_BUILD=1 RUN_ID=locomo_b_btf_full bash scripts/run_locomo_b_btf.sh

# Custom run with axis mode
python scripts/retrieve_locomo_b_btf.py --top-k 5 --mode both \
  --run-id locomo_b_btf_full --output-dir out/custom_run \
  --raw-data-file dataset/locomo10.json \
  --final-content-file out/locomo_b_btf_full/final_boxes_content.jsonl \
  --limit-conversations -1 --overwrite --axis-mode auto

# Metrics analysis
python scripts/analyze_repro_metrics.py out/locomo_b_btf_warm_latency_20260623
```
