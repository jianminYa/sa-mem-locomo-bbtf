"""
Generate trace file for retrieval_enhanced.jsonl to enable ablation analysis.
"""
import json
import os
from typing import Dict, List, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results

def load_ground_truth(gt_path: str) -> Dict:
    """Load ground truth data."""
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    return gt_data.get("ground_truth", {})

def generate_trace_from_retrieval(
    retrieval_results: List[Dict],
    ground_truth: Dict,
    all_blocks_count: int = 273  # Default from the data
) -> List[Dict]:
    """
    Generate trace data from retrieval results.

    Since we don't have the actual filtering statistics, we'll create
    minimal trace data based on what we can infer from the results.
    """
    traces = []

    for result in retrieval_results:
        qa_idx = str(result.get("qa_idx", ""))
        rankings = result.get("rankings", {}).get("content_event_topic_kw", [])

        # Get ground truth for this query
        gt_entry = ground_truth.get(qa_idx, {})
        gt_blocks = set(gt_entry.get("block_ids", []))
        gt_total = len(gt_blocks)

        # Calculate GT in top-K
        gt_in_topk = len(set(rankings) & gt_blocks) if rankings else 0

        # Since we don't have actual filtering data, we'll use the final ranking size
        # as a proxy for all filtering stages
        final_size = len(rankings)

        # Create trace entry with minimal data
        trace = {
            "user_id": result.get("user_id"),
            "qa_idx": qa_idx,
            "pool_size": all_blocks_count,  # Assume all blocks in initial pool
            "time_filtered_size": final_size,  # We don't have this, use final size
            "event_filtered_size": final_size,  # We don't have this, use final size
            "final_size": final_size,
            "gt_total": gt_total,
            "gt_in_pool": gt_total,  # Assume all GT blocks were in initial pool
            "gt_after_time": gt_total,  # We don't have this, assume no loss
            "gt_after_event": gt_total,  # We don't have this, assume no loss
            "gt_in_topk": gt_in_topk,
            "gt_lost_time": 0,  # We don't have this data
            "gt_lost_event": 0,  # We don't have this data
            "fallback_triggered": False,  # We don't have this data
        }

        traces.append(trace)

    return traces

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate trace file for retrieval_enhanced.jsonl")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run ID to process"
    )
    parser.add_argument(
        "--all-blocks",
        type=int,
        default=273,
        help="Total number of blocks in the dataset"
    )

    args = parser.parse_args()

    output_dir = f"out/{args.run_id}"

    # Load retrieval results
    retrieval_file = os.path.join(output_dir, "retrieval_enhanced.jsonl")
    if not os.path.exists(retrieval_file):
        print(f"❌ Retrieval file not found: {retrieval_file}")
        return

    # Load ground truth
    gt_path = os.path.join(output_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        print(f"❌ Ground truth file not found: {gt_path}")
        return

    print(f"📖 Loading retrieval results from: {retrieval_file}")
    retrieval_results = load_jsonl(retrieval_file)
    print(f"   Found {len(retrieval_results)} queries")

    print(f"📖 Loading ground truth from: {gt_path}")
    ground_truth = load_ground_truth(gt_path)
    print(f"   Found {len(ground_truth)} ground truth entries")

    # Generate trace data
    print(f"🔧 Generating trace data...")
    traces = generate_trace_from_retrieval(retrieval_results, ground_truth, args.all_blocks)

    # Write trace file
    trace_file = os.path.join(output_dir, "retrieval_enhanced_trace.jsonl")
    with open(trace_file, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"✅ Trace file generated: {trace_file}")
    print(f"   {len(traces)} trace entries written")
    print()
    print("⚠️  Note: This trace file contains estimated filtering statistics.")
    print("   For accurate filtering metrics, run retrieval through compare_retrieval_methods.py")

if __name__ == "__main__":
    main()
