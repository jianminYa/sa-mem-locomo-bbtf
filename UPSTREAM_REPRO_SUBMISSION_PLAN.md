# Upstream PR Submission Plan

**Target repo**: https://github.com/RichardWang11/SA-Mem-Research

## Recommended Files to Submit

### Scripts (new)

| File | Description |
|------|-------------|
| `scripts/run_locomo_b_btf.sh` | End-to-end B/B+TF pipeline |
| `scripts/retrieve_locomo_b_btf.py` | Retrieval wrapper with top-k and axis-mode |
| `scripts/analyze_repro_metrics.py` | Evidence metrics analysis (Complete-MRR, First-RR, latency) |

### Documentation (new)

| File | Description |
|------|-------------|
| `REPRODUCTION_REPORT_RECHECK.md` | Detailed reproduction report |
| `README.md` section | "LoCoMo B/B+TF Reproduction (Partial)" |

### Metrics (small files)

| File | Description |
|------|-------------|
| `out/locomo_b_btf_recheck_20260624/metrics_summary.json` | Evidence and latency metrics |
| `out/locomo_b_btf_recheck_20260624/metrics_summary.md` | Human-readable metrics |

### Code fixes (patches to existing files)

| File | Fix |
|------|-----|
| `retrieval/retrieval_impl_locomo.py` | `graph_person_relations` param, warm-cache, latency fields |
| `retrieval/retrieval_enhanced_locomo.py` | Bi-temporal filtering, warm-cache, query key fix, latency fields |
| `retrieval/query_pasing_byllm.py` | `time_axis` field, `dispatch_temporal_filter`, `_infer_time_axis` |
| `retrieval/retrieve_stage_enhanced_locomo.py` | `--axis-mode` argument |
| `generate_impl_locomo.py` | `LIMIT_CONVERSATIONS=-1` slice fix |

## Files NOT Recommended for Upstream

| File | Reason |
|------|--------|
| `dataset/locomo10.json` | May have license restrictions; authors should provide their own |
| `out/locomo_b_btf_full/vector_store/*.json` | Large cache files (~10MB total) |
| `out/locomo_b_btf_full/token_stream.jsonl` | Large debug log |
| `out/locomo_b_btf_full/trace_build_process.jsonl` | Large debug log |
| `out/locomo_b_btf_full/build_stats.jsonl` | Build-specific |
| `.env` | Contains API keys |
| `out/locomo_b_btf_*/retrieval_*.jsonl` | Large result files (submit metrics only) |
| `out/locomo_b_btf_*/generation_*.jsonl` | Large result files |
| `out/locomo_b_btf_*/evaluation_*.jsonl` | Large result files |

## PR Description Template

```markdown
## Summary

Partial reproduction of LoCoMo B/B+TF experiments from SA-Mem paper.

### Changes

1. Added `scripts/run_locomo_b_btf.sh` for end-to-end B/B+TF pipeline
2. Added `scripts/analyze_repro_metrics.py` for evidence metrics (Complete-MRR, First-RR)
3. Fixed `LIMIT_CONVERSATIONS=-1` slice bug in `generate_impl_locomo.py`
4. Added bi-temporal filtering support (`axis_mode`, `dispatch_temporal_filter`)
5. Added warm-cache optimization (EmbeddingStore per user)
6. Added fine-grained retrieval latency fields
7. Fixed B+TF query embedding cache key to avoid reusing B's embeddings

### Results

| Method | F1 | Hit@5 | Recall@5 | Complete-MRR |
|--------|-----|-------|----------|--------------|
| B | 0.5133 | 0.8201 | 0.7580 | 0.5608 |
| B+TF | 0.4879 | 0.7708 | 0.7065 | 0.5114 |

**Note**: B+TF temporal filtering gain is NOT reproduced. See REPRODUCTION_REPORT_RECHECK.md.

### Files

- New scripts: `scripts/run_locomo_b_btf.sh`, `scripts/retrieve_locomo_b_btf.py`, `scripts/analyze_repro_metrics.py`
- New report: `REPRODUCTION_REPORT_RECHECK.md`
- Metrics: `out/locomo_b_btf_recheck_20260624/metrics_summary.{json,md}`
```

## Checklist Before Submitting

- [ ] No `.env` or API keys in commit
- [ ] No large `out/` files (only metrics_summary)
- [ ] No `vector_store/*.json`
- [ ] No `token_stream.jsonl` or `trace_build_process.jsonl`
- [ ] README updated with partial reproduction note
- [ ] Report clearly states what's reproduced and what's not
- [ ] All code changes documented

---

*Created: 2026-06-24*
