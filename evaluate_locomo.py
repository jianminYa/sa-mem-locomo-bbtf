#!/usr/bin/env python3
"""
Evaluation script for LoCoMo generation results using official LoCoMo evaluation metrics.
Uses category-specific F1 calculation methods to match the paper's reported results.
"""
import argparse
import json
import os
import sys
import re
import string
import regex
from typing import Dict, Any
from collections import defaultdict, Counter
from tenacity import retry, stop_after_attempt, wait_random_exponential
import numpy as np
from nltk.stem import PorterStemmer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import memblock_extractor as mx

# Initialize Porter Stemmer for token normalization
ps = PorterStemmer()


# Official LoCoMo evaluation functions (copied from /data/locomo/task_eval/evaluation.py)
def normalize_answer(s):
    """Normalize answer string by removing articles, punctuation, and lowercasing."""
    s = s.replace(',', "")

    def remove_articles(text):
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def locomo_f1_score(prediction, ground_truth):
    """
    Simple token-level F1 with stemming.
    Used for Categories 2, 3, 4 (Single-Hop, Temporal, Open Domain).
    """
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
    if precision + recall == 0:
        return 0
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def locomo_pr_score(prediction, ground_truth):
    """
    Token-level precision/recall with stemming.
    """
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
    recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0.0
    return precision, recall


def locomo_f1_multi(prediction, ground_truth):
    """
    Multi-answer F1 that splits by commas and computes partial F1.
    Used for Category 1 (Multi-Hop).
    """
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    return np.mean([max([locomo_f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def locomo_pr_multi(prediction, ground_truth):
    """
    Multi-answer precision/recall using best-matching prediction per ground-truth item.
    """
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    if not predictions or not ground_truths:
        return 0.0, 0.0
    precisions = []
    recalls = []
    for gt in ground_truths:
        best_f1 = -1.0
        best_p = 0.0
        best_r = 0.0
        for pred in predictions:
            p, r = locomo_pr_score(pred, gt)
            f1 = locomo_f1_score(pred, gt)
            if f1 > best_f1:
                best_f1 = f1
                best_p = p
                best_r = r
        precisions.append(best_p)
        recalls.append(best_r)
    return float(np.mean(precisions)), float(np.mean(recalls))


def exact_match_single(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def exact_match_multi(prediction, ground_truth):
    pred_items = [normalize_answer(p) for p in prediction.split(',') if p.strip()]
    gold_items = [normalize_answer(g) for g in ground_truth.split(',') if g.strip()]
    if not pred_items or not gold_items:
        return False
    return set(pred_items) == set(gold_items)


# LLM-as-judge prompt for binary classification
EVALUATION_PROMPT = """You are an evaluation expert for AI memory system question answering.

# TASK
Evaluate whether the "Memory System Response" correctly answers the "Question" based on the "Reference Answer".
Classify the response as either **"Correct"** or **"Incorrect"**.

# EVALUATION CRITERIA

## Correct
* The response accurately answers the question and is **semantically equivalent** to the reference answer
* Synonyms, paraphrasing, and reasonable summarization are acceptable
* Minor formatting differences are acceptable if the core meaning matches
* For multi-element questions, **all elements must be present and correct**

## Incorrect
* The response contradicts the reference answer
* The response is incomplete (missing key elements)
* The response provides wrong information
* The response states "don't know" or "no memory" when a reference answer exists
* For multi-element questions, missing **any** element makes it incorrect

# GUIDELINES
* Equivalent expressions of numbers, times, and units are acceptable
* The numerical values and facts must match the reference answer
* Do not use external knowledge - evaluate based only on the provided information

# INPUT

**Question:**
{question}

**Reference Answer:**
{reference_answer}

**Memory System Response:**
{response}

# OUTPUT

Provide your evaluation in JSON format:

```json
{{
  "reasoning": "Brief explanation of why the response is correct or incorrect",
  "evaluation_result": "Correct | Incorrect"
}}
```

Respond with ONLY the JSON block, no additional text.
"""


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(3),
    reraise=True
)
def evaluate_with_llm_judge(
    worker: Any,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    note: str = "judge"
) -> Dict[str, Any]:
    """
    Evaluate a single answer using LLM-as-judge with binary classification.

    Returns:
        Dict with 'reasoning', 'evaluation_result' (Correct/Incorrect), and 'score' (100/0)
    """
    prompt = EVALUATION_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        response=predicted_answer
    )

    try:
        response = worker.chat_completion(
            prompt,
            note=note,
            json_mode=False,  # Model returns markdown JSON block
            extra={"stage": "evaluation"}
        )

        # Extract JSON block from markdown
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not match:
            # Try without markdown wrapper
            match = re.search(r"(\{.*?\})", response, re.DOTALL)

        if not match:
            raise ValueError(f"No JSON found in response: {response}")

        json_str = match.group(1).strip()
        result = json.loads(json_str)

        # Validate and normalize result
        eval_result = result.get("evaluation_result", "").strip()
        if eval_result not in ["Correct", "Incorrect"]:
            mx.logger.warning(f"Invalid evaluation_result: {eval_result}, defaulting to Incorrect")
            eval_result = "Incorrect"

        # Convert to 100-point scale
        score = 100 if eval_result == "Correct" else 0

        return {
            "reasoning": result.get("reasoning", ""),
            "evaluation_result": eval_result,
            "score": score
        }

    except Exception as e:
        mx.logger.error(f"LLM judge evaluation failed: {e}")
        return {
            "reasoning": f"Evaluation failed: {str(e)}",
            "evaluation_result": "Error",
            "score": 0
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoCoMo generation results with LLM-as-Judge")
    parser.add_argument("--run-id", type=str, default="locomo", help="Run ID")
    parser.add_argument("--generation-file", type=str, default=None,
                       help="Path to generation results JSONL file")
    parser.add_argument("--use-llm-judge", action="store_true",
                       help="Enable LLM-as-judge evaluation (costs API calls)")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="Evaluate only first N samples (for testing)")

    parser.add_argument("--eval-output", type=str, default=None, help="Path to save detailed evaluation results (jsonl)")
    parser.add_argument("--summary-output", type=str, default=None, help="Path to save summary results (json)")

    args = parser.parse_args()

    # Apply run_id configuration
    mx.Config.apply_run_id(args.run_id)

    # Determine generation file path
    if args.generation_file:
        generation_file = args.generation_file
    else:
        generation_file = os.path.join(mx.Config.OUTPUT_DIR, "generation_results_locomo.jsonl")

    # # Output files
    # eval_output = os.path.join(mx.Config.OUTPUT_DIR, "evaluation_results_locomo.jsonl")
    # summary_output = os.path.join(mx.Config.OUTPUT_DIR, "evaluation_summary_locomo.json")
    # Output files
    if args.eval_output:
        eval_output = args.eval_output
    else:
        eval_output = os.path.join(mx.Config.OUTPUT_DIR, "evaluation_results_locomo.jsonl")
    
    if args.summary_output:
        summary_output = args.summary_output
    else:
        summary_output = os.path.join(mx.Config.OUTPUT_DIR, "evaluation_summary_locomo.json")
    

    mx.logger.info("=" * 60)
    mx.logger.info("📊 LoCoMo Evaluation with LLM-as-Judge")
    mx.logger.info("=" * 60)
    mx.logger.info(f"Generation file: {generation_file}")
    mx.logger.info(f"LLM-as-judge: {args.use_llm_judge}")
    mx.logger.info(f"Sample size: {args.sample_size or 'All'}")
    mx.logger.info("=" * 60)

    # Check if generation file exists
    if not os.path.exists(generation_file):
        mx.logger.error(f"❌ Generation file not found: {generation_file}")
        sys.exit(1)

    # Load generation results
    with open(generation_file, 'r', encoding='utf-8') as f:
        results = [json.loads(line) for line in f if line.strip()]

    if args.sample_size:
        results = results[:args.sample_size]

    mx.logger.info(f"📝 Loaded {len(results)} generation results")

    # Initialize worker if using LLM judge
    worker = None
    if args.use_llm_judge:
        worker = mx.LLMWorker()
        mx.logger.info("🤖 LLM judge initialized")

    # Evaluation metrics
    metrics = {
        "total": len(results),
        "f1_scores": [],
        "bleu_scores": [],
        "precision_scores": [],
        "recall_scores": [],
        "accuracy_scores": [],
        "judge_scores": [] if args.use_llm_judge else None,
        "judge_correct_count": 0 if args.use_llm_judge else None,
        "by_category": defaultdict(lambda: {
            "count": 0,
            "f1_sum": 0.0,
            "precision_sum": 0.0,
            "recall_sum": 0.0,
            "accuracy_sum": 0.0,
            "bleu_sum": 0.0,
            "judge_sum": 0.0 if args.use_llm_judge else None,
            "judge_correct": 0 if args.use_llm_judge else None
        })
    }

    # Evaluate each result
    evaluated_results = []
    for idx, result in enumerate(results):
        # Extract data for evaluation
        question = result.get("question", "")
        gold = str(result.get("gold", ""))
        pred = str(result.get("pred", ""))
        category = result.get("category", 0)
        bleu = result.get("bleu", 0.0)

        # Apply category-specific F1 calculation (matching official LoCoMo evaluation)
        if category == 3:
            # Temporal: split answer by ';' and use first part only
            gold = gold.split(';')[0].strip()
            f1 = locomo_f1_score(pred, gold)
            precision, recall = locomo_pr_score(pred, gold)
            accuracy = 1.0 if exact_match_single(pred, gold) else 0.0
        elif category in [2, 4]:
            # Single-Hop, Open Domain: simple token-level F1
            f1 = locomo_f1_score(pred, gold)
            precision, recall = locomo_pr_score(pred, gold)
            accuracy = 1.0 if exact_match_single(pred, gold) else 0.0
        elif category == 1:
            # Multi-Hop: split by commas and compute partial F1
            f1 = locomo_f1_multi(pred, gold)
            precision, recall = locomo_pr_multi(pred, gold)
            accuracy = 1.0 if exact_match_multi(pred, gold) else 0.0
        elif category == 5:
            # Counterfactual: binary check
            if 'no information available' in pred.lower() or 'not mentioned' in pred.lower():
                f1 = 1.0
                precision = 1.0
                recall = 1.0
                accuracy = 1.0
            else:
                f1 = 0.0
                precision = 0.0
                recall = 0.0
                accuracy = 0.0
        else:
            # Unknown category
            f1 = 0.0
            precision = 0.0
            recall = 0.0
            accuracy = 0.0

        metrics["f1_scores"].append(f1)
        metrics["precision_scores"].append(precision)
        metrics["recall_scores"].append(recall)
        metrics["accuracy_scores"].append(accuracy)
        metrics["bleu_scores"].append(bleu)
        metrics["by_category"][category]["count"] += 1
        metrics["by_category"][category]["f1_sum"] += f1
        metrics["by_category"][category]["precision_sum"] += precision
        metrics["by_category"][category]["recall_sum"] += recall
        metrics["by_category"][category]["accuracy_sum"] += accuracy
        metrics["by_category"][category]["bleu_sum"] += bleu

        # LLM-as-judge evaluation
        judge_result = None
        if args.use_llm_judge:
            judge_result = evaluate_with_llm_judge(
                worker=worker,
                question=result.get("question", ""),
                reference_answer=result.get("gold", ""),
                predicted_answer=result.get("pred", ""),
                note=f"judge_{result.get('user_id')}_{result.get('qa_idx')}"
            )

            score = judge_result["score"]
            metrics["judge_scores"].append(score)
            metrics["by_category"][category]["judge_sum"] += score

            if judge_result["evaluation_result"] == "Correct":
                metrics["judge_correct_count"] += 1
                metrics["by_category"][category]["judge_correct"] += 1

        # Add evaluation to result
        evaluated_result = {
            **result,
            "f1_corrected": f1,  # Store corrected F1 using official LoCoMo method
            "evaluation": {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1_original": result.get("f1", 0.0),  # Keep original F1 for comparison
                "bleu": bleu,
                "judge": judge_result
            }
        }
        evaluated_results.append(evaluated_result)

    # Calculate summary statistics
    summary = {
        "total_samples": metrics["total"],
        "overall": {
            "avg_f1": sum(metrics["f1_scores"]) / len(metrics["f1_scores"]) if metrics["f1_scores"] else 0,
            "avg_precision": sum(metrics["precision_scores"]) / len(metrics["precision_scores"]) if metrics["precision_scores"] else 0,
            "avg_recall": sum(metrics["recall_scores"]) / len(metrics["recall_scores"]) if metrics["recall_scores"] else 0,
            "avg_accuracy": sum(metrics["accuracy_scores"]) / len(metrics["accuracy_scores"]) if metrics["accuracy_scores"] else 0,
            "avg_bleu": sum(metrics["bleu_scores"]) / len(metrics["bleu_scores"]) if metrics["bleu_scores"] else 0,
        },
        "by_category": {}
    }

    if args.use_llm_judge:
        summary["overall"]["avg_judge_score"] = sum(metrics["judge_scores"]) / len(metrics["judge_scores"]) if metrics["judge_scores"] else 0
        summary["overall"]["judge_correct_count"] = metrics["judge_correct_count"]
        summary["overall"]["judge_correct_ratio"] = metrics["judge_correct_count"] / metrics["total"] if metrics["total"] > 0 else 0

    # Per-category statistics
    for category, cat_metrics in metrics["by_category"].items():
        count = cat_metrics["count"]
        summary["by_category"][str(category)] = {
            "count": count,
            "avg_f1": cat_metrics["f1_sum"] / count if count > 0 else 0,
            "avg_precision": cat_metrics["precision_sum"] / count if count > 0 else 0,
            "avg_recall": cat_metrics["recall_sum"] / count if count > 0 else 0,
            "avg_accuracy": cat_metrics["accuracy_sum"] / count if count > 0 else 0,
            "avg_bleu": cat_metrics["bleu_sum"] / count if count > 0 else 0,
        }
        if args.use_llm_judge:
            summary["by_category"][str(category)]["avg_judge_score"] = cat_metrics["judge_sum"] / count if count > 0 else 0
            summary["by_category"][str(category)]["judge_correct_count"] = cat_metrics["judge_correct"]
            summary["by_category"][str(category)]["judge_correct_ratio"] = cat_metrics["judge_correct"] / count if count > 0 else 0

    # Save evaluated results
    with open(eval_output, 'w', encoding='utf-8') as f:
        for result in evaluated_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Save summary
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    mx.logger.info("=" * 60)
    mx.logger.info("📊 Evaluation Summary")
    mx.logger.info("=" * 60)
    mx.logger.info(f"Total samples: {summary['total_samples']}")
    mx.logger.info(f"Average F1: {summary['overall']['avg_f1']:.4f}")
    mx.logger.info(f"Average Precision: {summary['overall']['avg_precision']:.4f}")
    mx.logger.info(f"Average Recall: {summary['overall']['avg_recall']:.4f}")
    mx.logger.info(f"Average Accuracy: {summary['overall']['avg_accuracy']:.4f}")
    mx.logger.info(f"Average BLEU: {summary['overall']['avg_bleu']:.4f}")
    if args.use_llm_judge:
        mx.logger.info(f"Average Judge Score: {summary['overall']['avg_judge_score']:.2f}/100")
        mx.logger.info(f"Judge Correct: {summary['overall']['judge_correct_count']}/{summary['total_samples']} ({summary['overall']['judge_correct_ratio']:.2%})")
    mx.logger.info("")
    mx.logger.info("By Category:")
    for category, cat_summary in summary["by_category"].items():
        mx.logger.info(f"  Category {category}: {cat_summary['count']} samples")
        mx.logger.info(f"    F1: {cat_summary['avg_f1']:.4f}")
        mx.logger.info(f"    Precision: {cat_summary['avg_precision']:.4f}")
        mx.logger.info(f"    Recall: {cat_summary['avg_recall']:.4f}")
        mx.logger.info(f"    Accuracy: {cat_summary['avg_accuracy']:.4f}")
        mx.logger.info(f"    BLEU: {cat_summary['avg_bleu']:.4f}")
        if args.use_llm_judge:
            mx.logger.info(f"    Judge: {cat_summary['avg_judge_score']:.2f}/100")
            mx.logger.info(f"    Correct: {cat_summary['judge_correct_count']}/{cat_summary['count']} ({cat_summary['judge_correct_ratio']:.2%})")
    mx.logger.info("=" * 60)
    mx.logger.info(f"✅ Results saved to:")
    mx.logger.info(f"   - {eval_output}")
    mx.logger.info(f"   - {summary_output}")
    mx.logger.info("=" * 60)


if __name__ == "__main__":
    main()

