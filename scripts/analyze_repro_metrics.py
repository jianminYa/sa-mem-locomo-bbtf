#!/usr/bin/env python3
"""Analyze retrieval/generation/evaluation metrics for LoCoMo B/B+TF experiments."""

import json
import os
import sys
from pathlib import Path
from collections import Counter


def percentile(data, p):
    """Calculate p-th percentile."""
    if not data:
        return 0.0
    data = sorted(data)
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[-1]
    return data[f] + (k - f) * (data[c] - data[f])


def analyze_retrieval_latency(jsonl_path):
    """Analyze retrieval latency from a JSONL file."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    if not rows:
        return {}
    
    # Extract latency fields
    latency_fields = [
        'retrieval_latency_parse',
        'retrieval_latency_filter',
        'retrieval_latency_store_init',
        'retrieval_latency_query_vector',
        'retrieval_latency_block_vector_fetch',
        'retrieval_latency_cosine',
        'retrieval_latency_sort',
        'retrieval_latency_flush',
        'retrieval_latency_rank_total',
        'retrieval_latency_search_no_parse',
        'retrieval_latency_total_with_parse',
    ]
    
    stats = {}
    for field in latency_fields:
        values = [r.get(field, 0.0) for r in rows if field in r]
        if values:
            stats[field] = {
                'mean': sum(values) / len(values),
                'p50': percentile(values, 50),
                'p95': percentile(values, 95),
                'max': max(values),
                'min': min(values),
            }
    
    # Parse source distribution
    parse_sources = Counter(r.get('parse_source', 'NONE') for r in rows)
    stats['parse_source_distribution'] = dict(parse_sources)
    
    # Time constraint type distribution
    time_types = Counter(r.get('time_constraint_type', 'NONE') for r in rows)
    stats['time_constraint_distribution'] = dict(time_types)
    
    # Time axis distribution
    time_axes = Counter(r.get('time_axis', 'NONE') for r in rows)
    stats['time_axis_distribution'] = dict(time_axes)
    
    # Fallback count
    fallback_count = sum(1 for r in rows if r.get('fallback_to_full_pool', False))
    stats['fallback_count'] = fallback_count
    
    # Pool sizes
    filtered_sizes = [r.get('filtered_pool_size', 0) for r in rows if 'filtered_pool_size' in r]
    if filtered_sizes:
        stats['filtered_pool_size'] = {
            'mean': sum(filtered_sizes) / len(filtered_sizes),
            'p50': percentile(filtered_sizes, 50),
            'p95': percentile(filtered_sizes, 95),
        }
    
    initial_sizes = [r.get('initial_pool_size', 0) for r in rows if 'initial_pool_size' in r]
    if initial_sizes:
        stats['initial_pool_size'] = {
            'mean': sum(initial_sizes) / len(initial_sizes),
        }
    
    return stats


def analyze_generation_metrics(jsonl_path):
    """Analyze generation metrics from a JSONL file."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    if not rows:
        return {}
    
    stats = {}
    
    # Generation latency
    latencies = [r.get('generation_latency', 0.0) for r in rows if 'generation_latency' in r]
    if latencies:
        stats['generation_latency'] = {
            'mean': sum(latencies) / len(latencies),
            'p50': percentile(latencies, 50),
            'p95': percentile(latencies, 95),
            'max': max(latencies),
        }
    
    # Context tokens
    tokens = [r.get('context_tokens', 0) for r in rows if 'context_tokens' in r]
    if tokens:
        stats['context_tokens'] = {
            'mean': sum(tokens) / len(tokens),
            'p50': percentile(tokens, 50),
            'p95': percentile(tokens, 95),
            'max': max(tokens),
        }
    
    return stats


def compute_evidence_metrics(retrieval_path, top_k=5, filter_time_constraint=None):
    """Compute evidence metrics from retrieval results.

    Args:
        retrieval_path: Path to retrieval JSONL file
        top_k: Top-k cutoff for ranking
        filter_time_constraint: If set, only include queries with this time_constraint_type
                                (e.g., "POINT", "RANGE" for temporal-constrained subset)

    Returns:
        dict with metrics
    """
    rows = []
    with open(retrieval_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Filter by time constraint if specified
    if filter_time_constraint is not None:
        rows = [r for r in rows if r.get('time_constraint_type') == filter_time_constraint]

    first_rr_values = []  # First-Relevant MRR
    complete_mrr_values = []  # Complete-MRR (paper definition)
    hit_at_k = 0
    recall_at_k_values = []

    for row in rows:
        targets = row.get('target_boxes', [])
        if not targets:
            continue

        target_ids = set()
        for t in targets:
            if isinstance(t, dict):
                tid = t.get('block_id')
                if tid is not None:
                    target_ids.add(int(tid))
            elif isinstance(t, (int, str)):
                try:
                    target_ids.add(int(t))
                except (ValueError, TypeError):
                    pass

        if not target_ids:
            continue

        rankings = row.get('rankings', {})
        ranked_ids = rankings.get('content_event_topic_kw', [])[:top_k]
        ranked_ids_int = [int(bid) for bid in ranked_ids]

        # First-Relevant MRR (reciprocal rank of first relevant item)
        first_rr = 0.0
        for i, bid in enumerate(ranked_ids_int):
            if bid in target_ids:
                first_rr = 1.0 / (i + 1)
                break
        first_rr_values.append(first_rr)

        # Complete-MRR (paper definition):
        # If ALL gold memories are in top-k, score = M / rank_max
        # where M = number of gold memories, rank_max = rank of last gold memory
        # Otherwise 0
        found_ranks = []
        for i, bid in enumerate(ranked_ids_int):
            if bid in target_ids:
                found_ranks.append(i + 1)  # 1-indexed rank

        if len(found_ranks) == len(target_ids):
            # All gold memories found in top-k
            rank_max = max(found_ranks)
            complete_mrr = len(target_ids) / rank_max
        else:
            complete_mrr = 0.0
        complete_mrr_values.append(complete_mrr)

        # Hit@k
        if any(bid in target_ids for bid in ranked_ids_int):
            hit_at_k += 1

        # Recall@k
        found = sum(1 for bid in ranked_ids_int if bid in target_ids)
        recall_at_k_values.append(found / len(target_ids))

    n = len(rows)
    n_with_targets = len(first_rr_values)
    return {
        'first_relevant_mrr': sum(first_rr_values) / len(first_rr_values) if first_rr_values else 0.0,
        'complete_mrr': sum(complete_mrr_values) / len(complete_mrr_values) if complete_mrr_values else 0.0,
        'hit_at_k': hit_at_k / n if n > 0 else 0.0,
        'recall_at_k': sum(recall_at_k_values) / len(recall_at_k_values) if recall_at_k_values else 0.0,
        'total_queries': n,
        'queries_with_targets': n_with_targets,
        'filter_time_constraint': filter_time_constraint,
    }


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "out/locomo_b_btf_fix_latency_20260623"
    
    print(f"Analyzing metrics from: {out_dir}")
    print("=" * 60)
    
    results = {}
    
    # A. B retrieval latency
    print("\n[A] B Retrieval Latency:")
    b_retrieval = os.path.join(out_dir, "retrieval_baseline.jsonl")
    if os.path.exists(b_retrieval):
        b_latency = analyze_retrieval_latency(b_retrieval)
        results['b_retrieval_latency'] = b_latency
        for field in ['retrieval_latency_rank', 'retrieval_latency_total_with_parse', 'retrieval_latency_core_no_parse']:
            if field in b_latency:
                s = b_latency[field]
                print(f"  {field}: mean={s['mean']:.3f}s, p50={s['p50']:.3f}s, p95={s['p95']:.3f}s, max={s['max']:.3f}s")
    
    # B. B+TF retrieval latency
    print("\n[B] B+TF Retrieval Latency:")
    btf_retrieval = os.path.join(out_dir, "retrieval_enhanced.jsonl")
    if os.path.exists(btf_retrieval):
        btf_latency = analyze_retrieval_latency(btf_retrieval)
        results['btf_retrieval_latency'] = btf_latency
        for field in ['retrieval_latency_parse', 'retrieval_latency_filter', 'retrieval_latency_rank',
                      'retrieval_latency_total_with_parse', 'retrieval_latency_core_no_parse']:
            if field in btf_latency:
                s = btf_latency[field]
                print(f"  {field}: mean={s['mean']:.3f}s, p50={s['p50']:.3f}s, p95={s['p95']:.3f}s, max={s['max']:.3f}s")
        
        print(f"\n  Parse source distribution: {btf_latency.get('parse_source_distribution', {})}")
        print(f"  Time constraint distribution: {btf_latency.get('time_constraint_distribution', {})}")
        print(f"  Time axis distribution: {btf_latency.get('time_axis_distribution', {})}")
        print(f"  Fallback count: {btf_latency.get('fallback_count', 0)}")
        if 'filtered_pool_size' in btf_latency:
            s = btf_latency['filtered_pool_size']
            print(f"  Filtered pool size: mean={s['mean']:.1f}, p50={s['p50']:.0f}, p95={s['p95']:.0f}")
    
    # C. Summary table
    print("\n[C] Retrieval Latency Summary Table:")
    print("| Method | Avg Parse | Avg Filter | Avg Search No Parse | Avg Total With Parse | p50 Total | p95 Total |")
    print("|--------|-----------|------------|---------------------|----------------------|-----------|-----------|")
    
    b_parse = b_latency.get('retrieval_latency_parse', {}).get('mean', 0)
    b_filter = b_latency.get('retrieval_latency_filter', {}).get('mean', 0)
    b_search = b_latency.get('retrieval_latency_search_no_parse', {}).get('mean', 0)
    b_total = b_latency.get('retrieval_latency_total_with_parse', {}).get('mean', 0)
    b_p50 = b_latency.get('retrieval_latency_total_with_parse', {}).get('p50', 0)
    b_p95 = b_latency.get('retrieval_latency_total_with_parse', {}).get('p95', 0)
    print(f"| B | {b_parse:.3f} | {b_filter:.3f} | {b_search:.3f} | {b_total:.3f} | {b_p50:.3f} | {b_p95:.3f} |")
    
    btf_parse = btf_latency.get('retrieval_latency_parse', {}).get('mean', 0)
    btf_filter = btf_latency.get('retrieval_latency_filter', {}).get('mean', 0)
    btf_search = btf_latency.get('retrieval_latency_search_no_parse', {}).get('mean', 0)
    btf_total = btf_latency.get('retrieval_latency_total_with_parse', {}).get('mean', 0)
    btf_p50 = btf_latency.get('retrieval_latency_total_with_parse', {}).get('p50', 0)
    btf_p95 = btf_latency.get('retrieval_latency_total_with_parse', {}).get('p95', 0)
    print(f"| B+TF | {btf_parse:.3f} | {btf_filter:.3f} | {btf_search:.3f} | {btf_total:.3f} | {btf_p50:.3f} | {btf_p95:.3f} |")
    
    results['latency_summary'] = {
        'b': {'parse': b_parse, 'filter': b_filter, 'search_no_parse': b_search, 'total_with_parse': b_total, 'p50': b_p50, 'p95': b_p95},
        'btf': {'parse': btf_parse, 'filter': btf_filter, 'search_no_parse': btf_search, 'total_with_parse': btf_total, 'p50': btf_p50, 'p95': btf_p95},
    }
    
    # D. Generation metrics
    print("\n[D] Generation Metrics:")
    for label, fname in [("B", "generation_results_locomo_baseline.jsonl"), 
                          ("B+TF", "generation_results_locomo_time_filtering.jsonl")]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            gen_stats = analyze_generation_metrics(path)
            results[f'{label.lower().replace("+", "_")}_generation'] = gen_stats
            if 'generation_latency' in gen_stats:
                s = gen_stats['generation_latency']
                print(f"  {label} generation_latency: mean={s['mean']:.2f}s, p50={s['p50']:.2f}s, p95={s['p95']:.2f}s, max={s['max']:.2f}s")
            if 'context_tokens' in gen_stats:
                s = gen_stats['context_tokens']
                print(f"  {label} context_tokens: mean={s['mean']:.0f}, p50={s['p50']:.0f}, p95={s['p95']:.0f}, max={s['max']:.0f}")
    
    # E. Evidence metrics (First-Relevant MRR, Complete-MRR, Hit@5, Recall@5)
    print("\n[E] Evidence Metrics (top-5):")
    
    # Load B+TF retrieval to get time_constraint_type for subset indexing
    btf_path = os.path.join(out_dir, "retrieval_enhanced.jsonl")
    btf_rows_map = {}  # (user_id, qa_idx) -> row
    if os.path.exists(btf_path):
        with open(btf_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    btf_rows_map[(r.get('user_id'), r.get('qa_idx'))] = r
    
    # Build subset indices from B+TF time_constraint_type
    subset_point_range = set()
    subset_any_temporal = set()
    for (uid, qidx), r in btf_rows_map.items():
        tc = r.get('time_constraint_type', 'NONE')
        if tc in ('POINT', 'RANGE'):
            subset_point_range.add((uid, qidx))
        if tc in ('POINT', 'RANGE', 'BEFORE', 'AFTER', 'ANCHOR'):
            subset_any_temporal.add((uid, qidx))
    
    print(f"\n  Temporal subset sizes (from B+TF): POINT+RANGE={len(subset_point_range)}, any_temporal={len(subset_any_temporal)}")
    
    # Helper: filter rows by subset
    def filter_rows_by_subset(path, subset_keys):
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if (r.get('user_id'), r.get('qa_idx')) in subset_keys:
                        rows.append(r)
        return rows
    
    # Helper: compute evidence metrics from rows
    def compute_metrics_from_rows(rows, top_k=5):
        first_rr_values = []
        complete_mrr_values = []
        hit_at_k = 0
        recall_at_k_values = []
        
        for row in rows:
            targets = row.get('target_boxes', [])
            if not targets:
                continue
            target_ids = set()
            for t in targets:
                if isinstance(t, dict):
                    tid = t.get('block_id')
                    if tid is not None:
                        target_ids.add(int(tid))
                elif isinstance(t, (int, str)):
                    try:
                        target_ids.add(int(t))
                    except (ValueError, TypeError):
                        pass
            if not target_ids:
                continue
            
            rankings = row.get('rankings', {})
            ranked_ids = rankings.get('content_event_topic_kw', [])[:top_k]
            ranked_ids_int = [int(bid) for bid in ranked_ids]
            
            first_rr = 0.0
            for i, bid in enumerate(ranked_ids_int):
                if bid in target_ids:
                    first_rr = 1.0 / (i + 1)
                    break
            first_rr_values.append(first_rr)
            
            found_ranks = [i + 1 for i, bid in enumerate(ranked_ids_int) if bid in target_ids]
            if len(found_ranks) == len(target_ids):
                complete_mrr = len(target_ids) / max(found_ranks)
            else:
                complete_mrr = 0.0
            complete_mrr_values.append(complete_mrr)
            
            if any(bid in target_ids for bid in ranked_ids_int):
                hit_at_k += 1
            
            found = sum(1 for bid in ranked_ids_int if bid in target_ids)
            recall_at_k_values.append(found / len(target_ids))
        
        n = len(rows)
        n_with_targets = len(first_rr_values)
        return {
            'first_relevant_mrr': sum(first_rr_values) / len(first_rr_values) if first_rr_values else 0.0,
            'complete_mrr': sum(complete_mrr_values) / len(complete_mrr_values) if complete_mrr_values else 0.0,
            'hit_at_k': hit_at_k / n if n > 0 else 0.0,
            'recall_at_k': sum(recall_at_k_values) / len(recall_at_k_values) if recall_at_k_values else 0.0,
            'total_queries': n,
            'queries_with_targets': n_with_targets,
        }
    
    # E.1 All queries
    print("\n  [E.1] All queries:")
    for label, fname in [("B", "retrieval_baseline.jsonl"), 
                          ("B+TF", "retrieval_enhanced.jsonl")]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            ev = compute_evidence_metrics(path, top_k=5)
            results[f'{label.lower().replace("+", "_")}_evidence_all'] = ev
            print(f"    {label}: First-RR={ev['first_relevant_mrr']:.4f}, Complete-MRR={ev['complete_mrr']:.4f}, Hit@5={ev['hit_at_k']:.4f}, Recall@5={ev['recall_at_k']:.4f} ({ev['queries_with_targets']}/{ev['total_queries']})")
    
    # E.2 Temporal-constrained subsets (using B+TF time_constraint_type as index)
    for subset_name, subset_keys in [("POINT+RANGE", subset_point_range), ("any_temporal", subset_any_temporal)]:
        print(f"\n  [E.2] Temporal-constrained subset ({subset_name}, n={len(subset_keys)}):")
        for label, fname in [("B", "retrieval_baseline.jsonl"), 
                              ("B+TF", "retrieval_enhanced.jsonl")]:
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                rows = filter_rows_by_subset(path, subset_keys)
                ev = compute_metrics_from_rows(rows, top_k=5)
                results[f'{label.lower().replace("+", "_")}_evidence_{subset_name}'] = ev
                print(f"      {label}: First-RR={ev['first_relevant_mrr']:.4f}, Complete-MRR={ev['complete_mrr']:.4f}, Hit@5={ev['hit_at_k']:.4f}, Recall@5={ev['recall_at_k']:.4f} ({ev['queries_with_targets']}/{ev['total_queries']})")
    
    # F. Evaluation summary
    print("\n[F] Evaluation Summary:")
    for label, fname in [("B", "evaluation_summary_locomo_baseline.json"),
                          ("B+TF", "evaluation_summary_locomo_time_filtering.json")]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                ev = json.load(f)
            results[f'{label.lower().replace("+", "_")}_evaluation'] = ev
            o = ev.get('overall', {})
            print(f"  {label}: F1={o.get('avg_f1', 0):.4f}, P={o.get('avg_precision', 0):.4f}, R={o.get('avg_recall', 0):.4f}, Acc={o.get('avg_accuracy', 0):.4f}, BLEU={o.get('avg_bleu', 0):.4f}")
    
    # Save JSON
    json_path = os.path.join(out_dir, "metrics_summary.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")
    
    # Save Markdown
    md_path = os.path.join(out_dir, "metrics_summary.md")
    with open(md_path, 'w') as f:
        f.write("# LoCoMo B/B+TF Metrics Summary (with latency fix)\n\n")
        f.write(f"Output directory: `{out_dir}`\n\n")
        
        f.write("## Retrieval Latency\n\n")
        f.write("| Method | Avg Parse | Avg Filter | Avg Search No Parse | Avg Total With Parse | p50 Total | p95 Total |\n")
        f.write("|--------|-----------|------------|---------------------|----------------------|-----------|----------|\n")
        f.write(f"| B | {b_parse:.3f}s | {b_filter:.3f}s | {b_search:.3f}s | {b_total:.3f}s | {b_p50:.3f}s | {b_p95:.3f}s |\n")
        f.write(f"| B+TF | {btf_parse:.3f}s | {btf_filter:.3f}s | {btf_search:.3f}s | {btf_total:.3f}s | {btf_p50:.3f}s | {btf_p95:.3f}s |\n\n")
        
        f.write("### B+TF Parse Source Distribution\n\n")
        f.write(f"{btf_latency.get('parse_source_distribution', {})}\n\n")
        f.write("### B+TF Time Constraint Distribution\n\n")
        f.write(f"{btf_latency.get('time_constraint_distribution', {})}\n\n")
        f.write(f"### Fallback Count: {btf_latency.get('fallback_count', 0)}\n\n")
        
        f.write("## Evidence Metrics (top-5)\n\n")
        f.write("### All Queries\n\n")
        f.write("| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |\n")
        f.write("|--------|----------|--------------|-------|----------|--------------------|\n")
        for label, fname in [("B", "retrieval_baseline.jsonl"), ("B+TF", "retrieval_enhanced.jsonl")]:
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                ev = compute_evidence_metrics(path, top_k=5)
                f.write(f"| {label} | {ev['first_relevant_mrr']:.4f} | {ev['complete_mrr']:.4f} | {ev['hit_at_k']:.4f} | {ev['recall_at_k']:.4f} | {ev['queries_with_targets']}/{ev['total_queries']} |\n")
        
        # Temporal-constrained subsets (using B+TF time_constraint_type as index)
        for subset_name, subset_keys in [("POINT+RANGE", subset_point_range), ("any_temporal", subset_any_temporal)]:
            f.write(f"\n### Temporal-Constrained Subset ({subset_name}, n={len(subset_keys)})\n\n")
            f.write("*Subset defined by B+TF time_constraint_type; same (user_id, qa_idx) used for both methods.*\n\n")
            f.write("| Method | First-RR | Complete-MRR | Hit@5 | Recall@5 | Queries w/ Targets |\n")
            f.write("|--------|----------|--------------|-------|----------|--------------------|\n")
            for label, fname in [("B", "retrieval_baseline.jsonl"), ("B+TF", "retrieval_enhanced.jsonl")]:
                path = os.path.join(out_dir, fname)
                if os.path.exists(path):
                    rows = filter_rows_by_subset(path, subset_keys)
                    ev = compute_metrics_from_rows(rows, top_k=5)
                    f.write(f"| {label} | {ev['first_relevant_mrr']:.4f} | {ev['complete_mrr']:.4f} | {ev['hit_at_k']:.4f} | {ev['recall_at_k']:.4f} | {ev['queries_with_targets']}/{ev['total_queries']} |\n")
        
        f.write("\n## Evaluation Results\n\n")
        f.write("| Method | F1 | Precision | Recall | Accuracy | BLEU | Samples |\n")
        f.write("|--------|-----|-----------|--------|----------|------|---------|\n")
        for label, fname in [("B", "evaluation_summary_locomo_baseline.json"), ("B+TF", "evaluation_summary_locomo_time_filtering.json")]:
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                with open(path) as ef:
                    ev = json.load(ef)
                o = ev.get('overall', {})
                f.write(f"| {label} | {o.get('avg_f1', 0):.4f} | {o.get('avg_precision', 0):.4f} | {o.get('avg_recall', 0):.4f} | {o.get('avg_accuracy', 0):.4f} | {o.get('avg_bleu', 0):.4f} | {ev.get('total_samples', 0)} |\n")
        
        f.write("\n## By Category\n\n")
        for label, fname in [("B", "evaluation_summary_locomo_baseline.json"), ("B+TF", "evaluation_summary_locomo_time_filtering.json")]:
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                with open(path) as ef:
                    ev = json.load(ef)
                f.write(f"\n### {label}\n\n")
                f.write("| Category | Count | F1 | Precision | Recall | Accuracy | BLEU |\n")
                f.write("|----------|-------|-----|-----------|--------|----------|------|\n")
                cat_names = {'1': 'Multi-hop', '2': 'Temporal', '3': 'Open reasoning', '4': 'Single-hop'}
                for cat_id in sorted(ev.get('by_category', {}).keys(), key=int):
                    c = ev['by_category'][cat_id]
                    cat_name = cat_names.get(cat_id, cat_id)
                    f.write(f"| {cat_name} ({cat_id}) | {c['count']} | {c['avg_f1']:.4f} | {c['avg_precision']:.4f} | {c['avg_recall']:.4f} | {c['avg_accuracy']:.4f} | {c['avg_bleu']:.4f} |\n")
    
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
