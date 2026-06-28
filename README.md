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

## LoCoMo B/B+TF Reproduction (Partial)

This section documents a partial reproduction of SA-Mem's LoCoMo B/B+TF experiments.

**Status**: Partial reproduction. B baseline is close to paper numbers, but B+TF QA/evidence gains are NOT reproduced in this run. For latency, B+TF only shows the expected core-search speedup on the subset where temporal constraints are actually triggered.

### Key Findings

| Aspect | Status | Details |
|--------|--------|---------|
| QA (B) | ✅ Close | F1=0.5133, BLEU=0.3892 (paper: ~0.52, ~0.39) |
| QA (B+TF) | ❌ Lower | F1=0.4879 (lower than B) |
| Evidence (B) | ✅ Close | Hit@5=0.8201, Recall@5=0.7580 (paper: 0.8162, 0.7475) |
| Evidence (B+TF) | ❌ Lower | Hit@5=0.7708, Recall@5=0.7065 (lower than B) |
| Latency (B core) | ✅ Close | Core search p50=0.102s, p95=0.144s |
| Latency (B+TF all) | ⚠️ Mixed | Core search p50=0.108s; candidate pool 93.36 -> 87.33 on average |
| Latency (B+TF triggered temporal subset) | ✅ Local gain | Category-2 POINT/RANGE subset core p50=0.0085s; pool 96.00 -> 33.65 |
| Temporal gain | ❌ Not reproduced | B+TF < B on temporal-constrained queries |

See `docs/locomo_b_btf_reproduction.md` for full details.

### Latency Boundary Note

Paper Table 3 reports search-stage latency. The reproduction logs separate:

- **Online no-parse wall**: excludes LLM query parsing, but still includes rewritten-query embedding, vector-cache flush, and semantic ranking.
- **Paper-like core search**: temporal filter + block vector fetch + cosine scoring + sorting.

After the query-cache fix, B+TF online no-parse wall time is dominated by rewritten-query embedding and cache flush. Under the paper-like core-search boundary, B+TF is comparable to B across all queries and faster only on category-2 queries where the parser emits POINT/RANGE constraints.

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
| axis-mode | auto (bi-temporal inference) |

### Evidence Metrics Definition

- **Complete-MRR**: If all gold memories in top-k, score = M / rank_max, else 0 (paper definition)
- **First-RR**: Reciprocal rank of first relevant item (supplementary)
- **Hit@5**: Fraction of queries with ≥1 gold memory in top-5
- **Recall@5**: Fraction of gold memories found in top-5

**Note**: Our top-5 is not equivalent to paper's controlled token budget.

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
python scripts/analyze_repro_metrics.py out/locomo_b_btf_recheck_20260624
```

### Output Directories

| Directory | Description |
|-----------|-------------|
| `out/locomo_b_btf_full/` | Original run (old retrieval latency) |
| `out/locomo_b_btf_fix_latency_20260623/` | Fixed fallback + old latency |
| `out/locomo_b_btf_warm_latency_20260623/` | Warm-cache + bi-temporal |
| `out/locomo_b_btf_recheck_20260624/` | Corrected query key + evidence metrics |

**Note**: These directories contain large files (vector_store, token_stream, etc.) and should not be submitted to upstream PRs.
