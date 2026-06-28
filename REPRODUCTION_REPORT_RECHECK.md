# SA-Mem LoCoMo B/B+TF Reproduction Report (Recheck Version)

**Base commit**: `8d6ed17`
**Report branch**: `repro-latency-report-core-search`
**Date**: 2026-06-24
**New result directory**: `out/locomo_b_btf_recheck_20260624/`

## Executive Summary

This report documents a partial reproduction of the SA-Mem LoCoMo B/B+TF experiments using the public research code and a local LoCoMo10 dataset. The reproduction focuses on:

- **B**: baseline vector retrieval over MemBlocks.
- **B+TF**: baseline vector retrieval plus query parsing; the parsed temporal constraints are then used for Time Filtering, i.e., temporal-index candidate pruning before semantic ranking.

The main finding is:

- **B is close to the paper** on LoCoMo QA, all-query evidence metrics, and warm/core retrieval latency.
- **B+TF does not reproduce the paper's overall temporal-filtering gain** in QA or evidence metrics.
- **B+TF does show the intended core-latency behavior on the subset where temporal constraints are actually triggered**, but this local gain is diluted in the full run because most category-2 questions are parsed as `NONE`, and online rewritten-query embedding/cache flush dominates wall-clock timing.

Therefore, this should be submitted upstream as a **partial reproduction example**, not as a full reproduction claim.

## Reproduction Scope

| Item | Setting |
|------|---------|
| Dataset | LoCoMo10, 10 conversations |
| QA samples | 1540 non-category-5 QA pairs |
| Methods | B and B+TF |
| Graph retrieval | Disabled |
| Retrieval top-k | 5 |
| Generation answer top-n | 5 |
| Generation context | `content` text mode |
| LLM | `gpt-4o-mini` |
| Embedding model | `text-embedding-3-small` |
| Result directory | `out/locomo_b_btf_recheck_20260624/` |

This run reuses the public code path rather than implementing graph-enhanced retrieval. It does not claim reproduction of B+HTM, graph expansion, HaluMem, or controlled token-budget experiments beyond the top-5 evidence comparison reported below.

## Reproduction Pipeline

The reproduction follows the full LoCoMo QA workflow:

1. **Build MemBlocks** with `build_stage_locomo.py`.
   - Graph is disabled.
   - The build stage produces `out/<run_id>/final_boxes_content.jsonl`.
2. **Retrieve evidence** with both B and B+TF.
   - B writes `retrieval_baseline.jsonl`.
   - B+TF writes `retrieval_enhanced.jsonl`.
   - Retrieval uses top-k = 5.
3. **Generate answers** with `generate_stage_locomo.py`.
   - B uses `retrieval_baseline.jsonl`.
   - B+TF uses `retrieval_enhanced.jsonl`.
   - Generation uses `answer_topn=5` and `text_modes=content`.
4. **Evaluate QA** with `evaluate_locomo.py`.
   - Reports F1, precision, recall, accuracy, and BLEU.
5. **Analyze retrieval/evidence/latency** with `scripts/analyze_repro_metrics.py`.
   - Reports Complete-MRR, First-RR, Hit@5, Recall@5, latency breakdowns, and temporal-subset diagnostics.

The helper script `scripts/run_locomo_b_btf.sh` runs the end-to-end flow, while `scripts/retrieve_locomo_b_btf.py` exposes an explicit retrieval top-k before dispatching to the original retrieval entrypoint.

## Changes Made

### 1. Fixed B+TF Query Embedding Cache Key

**Problem**: B baseline and B+TF enhanced both used `key=qa_{user_id}_{q_id}`, `field="question"`. If B ran first, B+TF would reuse B's original question embedding instead of the rewritten query.

**Fix**: B+TF now uses `key=qa_enhanced_{user_id}_{q_id}_{md5(query_text)[:12]}`, `field="question_rewritten"`.

**Impact**: B+TF ranking changed significantly. The online no-parse wall time increased because rewritten-query embeddings and cache flushes are now charged to the retrieval path. The paper-like core search latency should therefore be reported separately from online query embedding and flush overhead. B+TF evidence metrics decreased, suggesting the rewritten query embeddings were not being used correctly before.

### 2. Fixed Temporal-Constrained Subset Evidence Statistics

**Problem**: B retrieval has no `time_constraint_type`, so filtering B by it returned 0/0.

**Fix**: Use B+TF's `time_constraint_type` to define the subset, then compute metrics for both B and B+TF on the same `(user_id, qa_idx)` pairs.

### 3. Added Any-Temporal Subset

Added `any_temporal` subset (POINT + RANGE + BEFORE + AFTER + ANCHOR) in addition to POINT+RANGE.

### 4. Added Warm-Cache and Fine-Grained Latency Instrumentation

The retrieval code now records separate latency fields for parse, temporal filter, query vector, block vector fetch, cosine scoring, sort, flush, rank total, search without parse, and total with parse. This is necessary because the paper Table 3 latency is a search-stage number, while the reproduction pipeline can also include online rewritten-query embedding and vector-cache flushing.

### 5. Added Reproducible Metrics Analysis

`scripts/analyze_repro_metrics.py` now:

- Computes paper-style Complete-MRR.
- Reports First-RR as a supplementary metric.
- Uses B+TF's temporal parser output to define temporal subsets, then evaluates B and B+TF on the same `(user_id, qa_idx)` pairs.
- Reports paper-like core search latency separately from online wall-clock components.
- Reports category-2 and category-2 POINT/RANGE-triggered latency diagnostics.
- Reads and writes UTF-8 files explicitly for Windows reproducibility.

## Sample Completeness

| File | Count |
|------|-------|
| retrieval_baseline.jsonl | 1540 |
| retrieval_enhanced.jsonl | 1540 |
| generation_results_locomo_baseline.jsonl | 1540 |
| generation_results_locomo_time_filtering.jsonl | 1540 |
| evaluation_summary_locomo_baseline.json | 1540 |
| evaluation_summary_locomo_time_filtering.json | 1540 |

## QA vs Paper Table 1

| Method | Our F1 | Our BLEU | Notes |
|--------|--------|----------|-------|
| B | 0.5133 | 0.3892 | Close to paper |
| B+TF | 0.4879 | 0.3692 | Lower than B |

**Note**: Paper Table 1 shows final SA-Mem results, not strict B/B+TF rows. Our B results are close; B+TF is lower, indicating temporal filtering gain is not reproduced.

## Retrieval Latency vs Paper Table 3

Paper Table 3 reports search-stage latency. In this reproduction, the raw retrieval JSONL exposes several nested latency fields, so we report two boundaries:

- **Online no-parse wall**: excludes LLM query parsing, but still includes rewritten-query embedding, vector-cache flush, and semantic ranking.
- **Paper-like core search**: temporal filter + block vector fetch + cosine scoring + sorting. This excludes LLM parsing, rewritten-query embedding API calls, and vector-cache flush.

### Online Components (All 1540 Queries)

| Method | Parse | Filter | Search No Parse | Total With Parse | p50 | p95 |
|--------|-------|--------|-----------------|------------------|-----|-----|
| B | 0 | 0 | 0.105s | 0.105s | 0.104s | 0.147s |
| B+TF | 1.003s | 0.002s | 2.227s | 3.231s | 3.189s | 4.264s |

For B+TF, `Search No Parse` is not a pure search number: it includes rewritten-query embedding (mean 0.780s) and vector-cache flush (mean 1.340s). This explains why the online no-parse wall time is much larger than the paper latency.

### Paper-Like Core Search (All 1540 Queries)

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1024s | 0.1023s | 0.1438s | 93.36 | 93.36 | 93.0 |
| B+TF | 0.1037s | 0.1078s | 0.1522s | 93.36 | 87.33 | 93.0 |

Under this paper-like boundary, B+TF is not 21x slower. Instead, it is roughly comparable to B. However, B+TF also does not become faster overall because temporal filtering only reduces the average candidate pool from 93.36 to 87.33 blocks, and the median filtered pool remains 93.0.

### Category-2 Temporal Questions

Category 2 is the LoCoMo temporal-question category. Looking at all 321 category-2 queries:

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1010s | 0.1010s | 0.1473s | 91.65 | 91.65 | 93.0 |
| B+TF | 0.1001s | 0.1040s | 0.1507s | 91.65 | 83.88 | 86.0 |

The aggregate category-2 effect is small because the B+TF parser only produced explicit temporal constraints for 40 of the 321 category-2 queries:

| B+TF Constraint Type | Count |
|----------------------|-------|
| NONE | 281 |
| POINT | 36 |
| RANGE | 4 |

Restricting the analysis to the 40 category-2 queries where B+TF actually produced POINT/RANGE constraints shows the intended latency pattern:

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1060s | 0.1053s | 0.1449s | 96.00 | 96.00 | 93.0 |
| B+TF | 0.0429s | 0.0085s | 0.1362s | 96.00 | 33.65 | 4.5 |

On this triggered temporal subset, temporal filtering substantially reduces the candidate pool and core semantic-ranking latency. This local effect is hidden in the full run because most category-2 queries are parsed as `NONE`, and because online rewritten-query embedding and cache flush dominate the no-parse wall time.

## Evidence Metrics vs Paper Table 5

### All Queries

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 |
| B+TF | 0.6066 | 0.5114 | 0.7708 | 0.7065 |

**Paper reference** (SA-Mem ~1200 tokens): Complete-MRR 0.5510, Hit@5 0.8162, Recall@5 0.7475

**Note**: Our top-5 is not equivalent to paper's controlled token budget. First-RR is supplementary, not the paper's C-MRR.

### Temporal-Constrained Subset (POINT+RANGE, n=209)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.5459 | 0.5218 | 0.7225 | 0.7083 |
| B+TF | 0.5103 | 0.4821 | 0.6268 | 0.6042 |

**Paper reference** (Table 7, no temporal filter): Recall@5 constrained = 0.6891

**Finding**: B+TF is lower than B on temporal-constrained queries. Temporal filtering gain is NOT reproduced.

### Any-Temporal Subset (n=216)

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.5374 | 0.5140 | 0.7222 | 0.7085 |
| B+TF | 0.5050 | 0.4778 | 0.6250 | 0.6031 |

Same pattern: B outperforms B+TF on temporal queries.

## Evaluation Results

| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |
|--------|-----|-----------|--------|----------|------|---------|
| B | 0.5133 | 0.5440 | 0.5409 | 0.2318 | 0.3892 | 1540 |
| B+TF | 0.4879 | 0.5151 | 0.5148 | 0.2195 | 0.3692 | 1540 |

## By Category

### B

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3636 | 0.3540 | 0.4296 | 0.0461 | 0.2371 |
| Temporal (2) | 321 | 0.5257 | 0.5511 | 0.5367 | 0.1931 | 0.4014 |
| Open reasoning (3) | 96 | 0.2689 | 0.2991 | 0.3290 | 0.1250 | 0.1925 |
| Single-hop (4) | 841 | 0.5867 | 0.6330 | 0.6040 | 0.3210 | 0.4580 |

### B+TF

| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |
|----------|-------|-----|-----------|--------|----------|------|
| Multi-hop (1) | 282 | 0.3572 | 0.3479 | 0.4246 | 0.0426 | 0.2336 |
| Temporal (2) | 321 | 0.5105 | 0.5328 | 0.5211 | 0.1807 | 0.3900 |
| Open reasoning (3) | 96 | 0.2419 | 0.2687 | 0.3034 | 0.0938 | 0.1584 |
| Single-hop (4) | 841 | 0.5512 | 0.5925 | 0.5667 | 0.3080 | 0.4307 |

## Conclusion

### Partially Aligned

1. **QA metrics (B)**: F1/BLEU close to paper Table 1
2. **All-query evidence (B)**: Hit@5, Recall@5, Complete-MRR close to paper Table 5
3. **Warm-cache latency (B)**: ~0.1s search latency, comparable to paper

### NOT Reproduced

1. **B+TF gain over B**: B+TF performs worse than B in this run
2. **Temporal-constrained evidence gain**: B+TF lower than B on POINT+RANGE queries
3. **B+TF end-to-end retrieval latency**: Online no-parse wall time is dominated by rewritten-query embedding and cache flush
4. **Full-run B+TF latency improvement**: Paper-like core search only improves on the subset where temporal constraints are actually triggered; across all queries the candidate-pool reduction is too small

### Cannot Claim

- Full reproduction of paper's temporal filtering ablation (Table 7)
- B+TF improvement over B
- Controlled token budget equivalence

### Can Claim

- Partial reproduction: B baseline is close to paper numbers
- Temporal filtering shows the expected core-latency reduction on the category-2 POINT/RANGE-triggered subset
- Evidence metrics methodology is correct (Complete-MRR, First-RR)
- Code changes are documented and reproducible

## Recommended Upstream Submission

The original target repository is `RichardWang11/SA-Mem-Research`. The upstream submission should be a clean branch, for example `locomo-b-btf-reproduction`, and should not copy this working repository wholesale. The recommended upstream shape is:

### Submit

| File / Change | Purpose |
|---------------|---------|
| `README.md` short section | Briefly explain LoCoMo B/B+TF partial reproduction and link to the detailed report |
| `docs/locomo_b_btf_reproduction.md` | Upstream-facing version of this report |
| `docs/locomo_b_btf_metrics_summary.md` | Small human-readable metrics summary |
| `docs/locomo_b_btf_metrics_summary.json` | Small machine-readable metrics summary |
| `scripts/run_locomo_b_btf.sh` | End-to-end B/B+TF runner |
| `scripts/retrieve_locomo_b_btf.py` | Thin wrapper to expose retrieval top-k |
| `scripts/analyze_repro_metrics.py` | Reproducible evidence, QA, and latency analysis |
| `retrieval/retrieval_impl_locomo.py` patch | Baseline warm-cache and latency fields |
| `retrieval/retrieval_enhanced_locomo.py` patch | B+TF query parsing, temporal-index candidate pruning, warm-cache, query-key fix, and latency fields |
| `retrieval/query_pasing_byllm.py` patch | Time-axis parsing and temporal-filter dispatch helpers |
| `retrieval/retrieve_stage_enhanced_locomo.py` patch | CLI support for axis mode and related options |
| `generate_impl_locomo.py` patch, if still needed | `LIMIT_CONVERSATIONS=-1` handling |

### Do Not Submit

| File / Directory | Reason |
|------------------|--------|
| `.env` | Contains API keys |
| `dataset/locomo10.json` | Dataset/license should remain external unless upstream explicitly wants it |
| Full `out/locomo_b_btf_*` directories | Large run artifacts, not suitable for a clean upstream PR |
| `out/**/vector_store/*.json` | Embedding cache files |
| `out/**/retrieval_*.jsonl` | Large raw retrieval outputs |
| `out/**/generation_*.jsonl` | Large raw generation outputs |
| `out/**/evaluation_*.jsonl` | Large raw evaluation outputs |
| `out/**/token_stream.jsonl` | Debug/token logs |
| `out/**/trace_build_process.jsonl` | Debug trace logs |
| `ANALYSIS_BTF_VS_B.md` as-is | Useful internal diagnosis, but should be condensed into the upstream report |
| `UPSTREAM_REPRO_SUBMISSION_PLAN.md` as-is | Planning document for this work, not an upstream artifact |

### Suggested Upstream Wording

The upstream report should use conservative wording:

> This is a partial LoCoMo B/B+TF reproduction. The B baseline is close to the paper's LoCoMo QA, evidence, and warm/core latency metrics. After isolating rewritten-query embeddings from baseline query embeddings, B+TF does not reproduce the paper's overall QA/evidence temporal-filtering gain in this run. On the category-2 subset where the parser emits POINT/RANGE constraints, temporal filtering does reduce the candidate pool and core semantic-ranking latency, but this local effect is diluted in the full run.

## Output Directory Structure

```
out/locomo_b_btf_recheck_20260624/
├── retrieval_baseline.jsonl          (1540 lines)
├── retrieval_enhanced.jsonl          (1540 lines)
├── generation_results_locomo_baseline.jsonl    (1540 lines)
├── generation_results_locomo_time_filtering.jsonl  (1540 lines)
├── evaluation_summary_locomo_baseline.json
├── evaluation_summary_locomo_time_filtering.json
├── metrics_summary.json
└── metrics_summary.md
```

---

*Report generated: 2026-06-24*
*Recheck version with corrected query embedding cache key*
