# LoCoMo B/B+TF Reproduction Notes

Base commit in this reproduction repository: `8d6ed17`
Dry-run branch: `upstream-repro-submission-final`
Run date: 2026-06-24
Raw local output directory: `out/locomo_b_btf_recheck_20260624/`

## Summary

This branch is a dry run of the submission shape intended for the original `RichardWang11/SA-Mem-Research` repository. It keeps the submission lightweight: only the reproduction notes, aggregate metrics, and reusable scripts are intended for review. Full raw `out/` artifacts are not included in the docs submission.

The reproduction focuses on the non-graph LoCoMo B/B+TF path:

- **B**: baseline vector retrieval over MemBlocks.
- **B+TF**: baseline retrieval plus query parsing; parsed temporal constraints are used for Time Filtering, i.e. temporal-index candidate pruning before semantic ranking.

Main observations:

- **B is close to the paper** on LoCoMo QA, all-query evidence metrics, and warm/core retrieval latency.
- **B+TF QA/evidence gains are not reproduced** under the current public enhanced retrieval path.
- **B+TF shows the intended core-latency behavior on the subset where temporal constraints are actually triggered**, but this local gain is diluted in the full run because most category-2 questions are parsed as `NONE`, and online rewritten-query embedding/cache flush dominates wall-clock timing.

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

This run covers only the non-graph B/B+TF path. B+HTM, graph expansion, HaluMem, and controlled token-budget experiments are outside this run.

## Query Rewriting Note

The public repository code already contains a `rewritten_query` field in `QueryParser`, and enhanced retrieval uses `directive.rewritten_query` as the query text for vector ranking. Query rewriting was therefore not introduced from scratch in this reproduction.

However, the paper does not provide the exact query-rewriting prompt or an isolated ablation separating query rewriting from temporal filtering. The README also describes enhanced mode mainly as Time Filtering. In the recheck run, fixing the query embedding cache key makes B+TF use rewritten-query embeddings instead of reusing B's original-question embeddings. After that fix, B+TF QA/evidence metrics decrease.

Therefore, the current B+TF result should be read as the behavior of the public enhanced retrieval path with rewritten-query embeddings enabled. A useful follow-up ablation is:

- **B+TF-original-query**: keep query parsing and temporal-index candidate pruning, but use the original question embedding for semantic ranking.

This would isolate whether the B+TF discrepancy mainly comes from temporal pruning, query rewriting quality, or their interaction.

## Qualitative Diagnostics

The recheck analysis includes examples that help interpret the B+TF gap. They are diagnostic examples, not a replacement for a dedicated no-rewrite ablation.

### Rewritten-Query Behavior

| Original query | Rewritten query | Removed time expression |
|----------------|-----------------|-------------------------|
| `What did Mel and her kids paint in their latest project in July 2023?` | `What did Mel and her kids paint in their latest project` | `in July 2023` |
| `What painting did Melanie show to Caroline on October 13, 2023?` | `What painting did Melanie show to Caroline` | `on October 13, 2023` |
| `Where did Caroline move from 4 years ago?` | `Where did Caroline move from` | `4 years ago` |

This is consistent with the public code design: time information is moved into `time_constraint`, while the rewritten query is used for vector ranking. The potential risk is that some time expressions also help disambiguate the semantic target.

### Cases Where B Is Better

| Query | Temporal filter effect | B top-5 | B+TF top-5 | Observation |
|-------|------------------------|---------|------------|-------------|
| `What did Mel and her kids paint in their latest project in July 2023?` | POINT filter, pool `67 -> 3` | Hits target `25` | Misses target | The correct block is excluded from the filtered pool. |
| `Why did Maria sit with the little girl at the shelter event in February 2023?` | POINT filter, pool `86 -> 7` | Hits target `127` | Misses target | The filtered pool is small but loses the relevant block. |

### Cases Where B+TF Is Better

| Query | Temporal filter effect | B top-5 | B+TF top-5 | Observation |
|-------|------------------------|---------|------------|-------------|
| `What painting did Melanie show to Caroline on October 13, 2023?` | POINT filter, pool `67 -> 4` | Misses target | Hits target `58` | Temporal pruning isolates the relevant time-local blocks. |

These examples suggest that the current discrepancy is not simply that Time Filtering has no effect. The mechanism helps on some cases, while over-pruning and rewritten-query mismatch hurt other cases.

## Reproduction Pipeline

1. **Build MemBlocks** with `build_stage_locomo.py`.
   - Graph disabled.
   - Output: `out/<run_id>/final_boxes_content.jsonl`.
2. **Retrieve evidence** with B and B+TF.
   - B output: `retrieval_baseline.jsonl`.
   - B+TF output: `retrieval_enhanced.jsonl`.
   - Retrieval top-k = 5.
3. **Generate answers** with `generate_stage_locomo.py`.
   - B uses `retrieval_baseline.jsonl`.
   - B+TF uses `retrieval_enhanced.jsonl`.
   - `answer_topn=5`, `text_modes=content`.
4. **Evaluate QA** with `evaluate_locomo.py`.
   - Reports F1, precision, recall, accuracy, BLEU.
5. **Analyze metrics** with `scripts/analyze_repro_metrics.py`.
   - Reports Complete-MRR, First-RR, Hit@5, Recall@5, latency breakdowns, and temporal-subset diagnostics.

Reusable scripts:

- `scripts/run_locomo_b_btf.sh`
- `scripts/retrieve_locomo_b_btf.py`
- `scripts/analyze_repro_metrics.py`

## Results

### QA vs Paper Table 1

| Method | F1 | BLEU | Note |
|--------|----|------|------|
| B | 0.5133 | 0.3892 | Close to paper SA-Mem overall |
| B+TF | 0.4879 | 0.3692 | Lower than B |

Paper reference: SA-Mem overall F1 = 0.5203, BLEU = 0.3908.

### Evidence vs Paper Table 5

| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|----------|--------------|-------|----------|
| B | 0.6592 | 0.5608 | 0.8201 | 0.7580 |
| B+TF | 0.6066 | 0.5114 | 0.7708 | 0.7065 |

Paper reference for SA-Mem at about 1200 tokens: Complete-MRR = 0.5510, Hit@k = 0.8162, Recall@k = 0.7475.

This top-5 setup is not equivalent to the paper's controlled token-budget setup. First-RR is supplementary and is not the paper's C-MRR.

### Temporal-Constrained Evidence

| Subset | Method | First-RR | Complete-MRR | Hit@5 | Recall@5 |
|--------|--------|----------|--------------|-------|----------|
| POINT+RANGE (n=209) | B | 0.5459 | 0.5218 | 0.7225 | 0.7083 |
| POINT+RANGE (n=209) | B+TF | 0.5103 | 0.4821 | 0.6268 | 0.6042 |
| any temporal (n=216) | B | 0.5374 | 0.5140 | 0.7222 | 0.7085 |
| any temporal (n=216) | B+TF | 0.5050 | 0.4778 | 0.6250 | 0.6031 |

Paper Table 7 reference: no temporal filter constrained Recall@5 = 0.6891; Union Session & Event constrained Recall@5 = 0.7596.

### Retrieval Latency

Paper Table 3 reports search-stage latency. This reproduction separates:

- **Online no-parse wall**: excludes LLM query parsing, but includes rewritten-query embedding, vector-cache flush, and semantic ranking.
- **Paper-like core search**: temporal filter + block vector fetch + cosine scoring + sorting.

| Method | Online no-parse mean | Total-with-parse p50 | Total-with-parse p95 | Core p50 | Core p95 |
|--------|----------------------|----------------------|----------------------|----------|----------|
| B | 0.105s | 0.104s | 0.147s | 0.102s | 0.144s |
| B+TF | 2.227s | 3.189s | 4.264s | 0.108s | 0.152s |

B+TF online no-parse time includes rewritten-query embedding (mean 0.780s) and vector-cache flush (mean 1.340s). Under the paper-like core-search boundary, B+TF is roughly comparable to B on all queries.

On category-2 POINT/RANGE-triggered queries (n=40), B+TF shows the expected local latency pattern:

| Method | Core Mean | Core p50 | Core p95 | Initial Pool Mean | Filtered Pool Mean | Filtered Pool p50 |
|--------|-----------|----------|----------|-------------------|--------------------|-------------------|
| B | 0.1060s | 0.1053s | 0.1449s | 96.00 | 96.00 | 93.0 |
| B+TF | 0.0429s | 0.0085s | 0.1362s | 96.00 | 33.65 | 4.5 |

## What Aligns

- B QA F1/BLEU are close to paper Table 1 overall SA-Mem values.
- B all-query evidence metrics are close to paper Table 5 at about 1200 tokens.
- B warm/core retrieval latency is close to paper Table 3.
- B+TF shows local core-latency improvement when POINT/RANGE temporal constraints are actually triggered.

## Remaining Differences

- B+TF QA/evidence gains over B are not reproduced in the current run.
- B+TF temporal-constrained evidence metrics are lower than B on this run.
- Full-run B+TF core latency does not improve much because average candidate-pool reduction is small.
- Online B+TF latency is dominated by rewritten-query embedding and cache flush.
- Query rewriting and temporal filtering are not isolated; a B+TF-original-query ablation would help.
- Top-5 evidence is not identical to the paper's controlled token-budget evaluation.

## Submitted Files in This Dry Run

| File | Purpose |
|------|---------|
| `README.md` | Short reproduction summary and links |
| `docs/locomo_b_btf_reproduction.md` | Full reproduction notes |
| `docs/locomo_b_btf_metrics_summary.md` | Human-readable aggregate metrics |
| `docs/locomo_b_btf_metrics_summary.json` | Machine-readable aggregate metrics |
| `scripts/run_locomo_b_btf.sh` | End-to-end runner |
| `scripts/retrieve_locomo_b_btf.py` | Retrieval wrapper exposing top-k |
| `scripts/analyze_repro_metrics.py` | Metrics and latency analyzer |

Some scripts may already exist in this reproduction repository's `main` branch, so they may not appear as new files in this dry-run PR. They are still part of the intended upstream submission package.

## Files Intentionally Not Submitted

| File / Directory | Reason |
|------------------|--------|
| `.env` | Contains API keys |
| `dataset/locomo10.json` | Dataset/license should remain external |
| Full `out/locomo_b_btf_*` directories | Large run artifacts |
| `out/**/vector_store/*.json` | Embedding cache files |
| `out/**/retrieval_*.jsonl` | Raw retrieval outputs |
| `out/**/generation_*.jsonl` | Raw generation outputs |
| `out/**/evaluation_*.jsonl` | Raw evaluation outputs |
| `out/**/token_stream.jsonl` | Debug/token logs |
| `out/**/trace_build_process.jsonl` | Build trace logs |

Raw artifacts were generated locally under `out/locomo_b_btf_recheck_20260624/`. This branch includes aggregate summaries under `docs/` so the submission remains reviewable. The raw JSONL artifacts can be regenerated with the provided scripts.
