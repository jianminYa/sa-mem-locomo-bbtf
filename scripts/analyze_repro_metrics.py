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
        'retrieval_latency_rank',
        'retrieval_latency_total_with_parse',
        'retrieval_latency_core_no_parse',
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


def compute_evidence_metrics(retrieval_path, top_k=5):
    """Compute C-MRR, Hit@k, Recall@k from retrieval results."""
    rows = []
    with open(retrieval_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    
    cmrr_values = []
    hit_at_k = 0
    recall_at_k_values = []
    
    for row in rows:
        targets = row.get('target_boxes', [])
        if not targets:
            continue
        
        target_ids = set()
        for t in targets:
            if isinstance(t, dict):
                target_ids.add(t.get('block_id'))
            elif isinstance(t, (int, str)):
                target_ids.add(int(t))
        
        if not target_ids:
            continue
        
        rankings = row.get('rankings', {})
        ranked_ids = rankings.get('content_event_topic_kw', [])[:top_k]
        
        # C-MRR (reciprocal rank of first relevant item)
        rr = 0.0
        for i, bid in enumerate(ranked_ids):
            if int(bid) in target_ids:
                rr = 1.0 / (i + 1)
                break
        cmrr_values.append(rr)
        
        # Hit@k
        if any(int(bid) in target_ids for bid in ranked_ids):
            hit_at_k += 1
        
        # Recall@k
        found = sum(1 for bid in ranked_ids if int(bid) in target_ids)
        recall_at_k_values.append(found / len(target_ids))
    
    n = len(rows)
    return {
        'c_mrr': sum(cmrr_values) / len(cmrr_values) if cmrr_values else 0.0,
        'hit_at_k': hit_at_k / n if n > 0 else 0.0,
        'recall_at_k': sum(recall_at_k_values) / len(recall_at_k_values) if recall_at_k_values else 0.0,
        'total_queries': n,
        'queries_with_targets': len(cmrr_values),
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
        print(f"  Fallback count: {btf_latency.get('fallback_count', 0)}")
        if 'filtered_pool_size' in btf_latency:
            s = btf_latency['filtered_pool_size']
            print(f"  Filtered pool size: mean={s['mean']:.1f}, p50={s['p50']:.0f}, p95={s['p95']:.0f}")
    
    # C. Summary table
    print("\n[C] Retrieval Latency Summary Table:")
    print("| Method | Avg Parse | Avg Filter | Avg Rank | Avg Core No Parse | Avg Total With Parse | p50 Total | p95 Total |")
    print("|--------|-----------|------------|----------|-------------------|----------------------|-----------|-----------|")
    
    b_rank = b_latency.get('retrieval_latency_rank', {}).get('mean', 0)
    b_core = b_latency.get('retrieval_latency_core_no_parse', {}).get('mean', 0)
    b_total = b_latency.get('retrieval_latency_total_with_parse', {}).get('mean', 0)
    b_p50 = b_latency.get('retrieval_latency_total_with_parse', {}).get('p50', 0)
    b_p95 = b_latency.get('retrieval_latency_total_with_parse', {}).get('p95', 0)
    print(f"| B | 0 | 0 | {b_rank:.3f} | {b_core:.3f} | {b_total:.3f} | {b_p50:.3f} | {b_p95:.3f} |")
    
    btf_parse = btf_latency.get('retrieval_latency_parse', {}).get('mean', 0)
    btf_filter = btf_latency.get('retrieval_latency_filter', {}).get('mean', 0)
    btf_rank = btf_latency.get('retrieval_latency_rank', {}).get('mean', 0)
    btf_core = btf_latency.get('retrieval_latency_core_no_parse', {}).get('mean', 0)
    btf_total = btf_latency.get('retrieval_latency_total_with_parse', {}).get('mean', 0)
    btf_p50 = btf_latency.get('retrieval_latency_total_with_parse', {}).get('p50', 0)
    btf_p95 = btf_latency.get('retrieval_latency_total_with_parse', {}).get('p95', 0)
    print(f"| B+TF | {btf_parse:.3f} | {btf_filter:.3f} | {btf_rank:.3f} | {btf_core:.3f} | {btf_total:.3f} | {btf_p50:.3f} | {btf_p95:.3f} |")
    
    results['latency_summary'] = {
        'b': {'parse': 0, 'filter': 0, 'rank': b_rank, 'core_no_parse': b_core, 'total_with_parse': b_total, 'p50': b_p50, 'p95': b_p95},
        'btf': {'parse': btf_parse, 'filter': btf_filter, 'rank': btf_rank, 'core_no_parse': btf_core, 'total_with_parse': btf_total, 'p50': btf_p50, 'p95': btf_p95},
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
    
    # E. Evidence metrics (C-MRR, Hit@5, Recall@5)
    print("\n[E] Evidence Metrics (top-5):")
    for label, fname in [("B", "retrieval_baseline.jsonl"), 
                          ("B+TF", "retrieval_enhanced.jsonl")]:
        path = os.path.join(out_dir, fname)
        if os.path.exists(path):
            ev = compute_evidence_metrics(path, top_k=5)
            results[f'{label.lower().replace("+", "_")}_evidence'] = ev
            print(f"  {label}: C-MRR={ev['c_mrr']:.4f}, Hit@5={ev['hit_at_k']:.4f}, Recall@5={ev['recall_at_k']:.4f} ({ev['queries_with_targets']}/{ev['total_queries']} queries)")
    
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
        f.write("| Method | Avg Parse | Avg Filter | Avg Rank | Avg Core No Parse | Avg Total With Parse | p50 Total | p95 Total |\n")
        f.write("|--------|-----------|------------|----------|-------------------|----------------------|-----------|----------|\n")
        f.write(f"| B | 0 | 0 | {b_rank:.3f}s | {b_core:.3f}s | {b_total:.3f}s | {b_p50:.3f}s | {b_p95:.3f}s |\n")
        f.write(f"| B+TF | {btf_parse:.3f}s | {btf_filter:.3f}s | {btf_rank:.3f}s | {btf_core:.3f}s | {btf_total:.3f}s | {btf_p50:.3f}s | {btf_p95:.3f}s |\n\n")
        
        f.write("### B+TF Parse Source Distribution\n\n")
        f.write(f"{btf_latency.get('parse_source_distribution', {})}\n\n")
        f.write("### B+TF Time Constraint Distribution\n\n")
        f.write(f"{btf_latency.get('time_constraint_distribution', {})}\n\n")
        f.write(f"### Fallback Count: {btf_latency.get('fallback_count', 0)}\n\n")
        
        f.write("## Evidence Metrics (top-5)\n\n")
        f.write("| Method | C-MRR | Hit@5 | Recall@5 |\n")
        f.write("|--------|-------|-------|----------|\n")
        for label, fname in [("B", "retrieval_baseline.jsonl"), ("B+TF", "retrieval_enhanced.jsonl")]:
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                ev = compute_evidence_metrics(path, top_k=5)
                f.write(f"| {label} | {ev['c_mrr']:.4f} | {ev['hit_at_k']:.4f} | {ev['recall_at_k']:.4f} |\n")
        
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
