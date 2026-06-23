#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-locomo_b_btf}"
DATA_FILE="${DATA_FILE:-dataset/locomo10.json}"
LIMIT_CONVERSATIONS="${LIMIT_CONVERSATIONS:--1}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
ANSWER_TOP_N="${ANSWER_TOP_N:-5}"
TEXT_MODE="${TEXT_MODE:-content}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"

echo "[1/6] Build memory blocks"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  python build_stage_locomo.py \
    --stage build \
    --run-id "$RUN_ID" \
    --raw-data-file "$DATA_FILE" \
    --limit-conversations "$LIMIT_CONVERSATIONS" \
    --disable-graph \
    --llm-model "$LLM_MODEL" \
    --embedding-model "$EMBEDDING_MODEL"
else
  echo "SKIP_BUILD=1, using existing out/${RUN_ID}/final_boxes_content.jsonl"
fi

echo "[2/6] Retrieval: B and B+TF"
python scripts/retrieve_locomo_b_btf.py \
  --top-k "$RETRIEVAL_TOP_K" \
  --mode both \
  --run-id "$RUN_ID" \
  --raw-data-file "$DATA_FILE" \
  --final-content-file "out/${RUN_ID}/final_boxes_content.jsonl" \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --overwrite

echo "[3/6] Generation: B"
python generate_stage_locomo.py \
  --run-id "$RUN_ID" \
  --raw-data-file "$DATA_FILE" \
  --final-content-file "out/${RUN_ID}/final_boxes_content.jsonl" \
  --retrieval-file "out/${RUN_ID}/retrieval_baseline.jsonl" \
  --answer-topn "$ANSWER_TOP_N" \
  --text-modes "$TEXT_MODE" \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --output-suffix baseline

echo "[4/6] Generation: B+TF"
python generate_stage_locomo.py \
  --run-id "$RUN_ID" \
  --raw-data-file "$DATA_FILE" \
  --final-content-file "out/${RUN_ID}/final_boxes_content.jsonl" \
  --retrieval-file "out/${RUN_ID}/retrieval_enhanced.jsonl" \
  --answer-topn "$ANSWER_TOP_N" \
  --text-modes "$TEXT_MODE" \
  --limit-conversations "$LIMIT_CONVERSATIONS" \
  --output-suffix time_filtering

echo "[5/6] Evaluation: B"
python evaluate_locomo.py \
  --run-id "$RUN_ID" \
  --generation-file "out/${RUN_ID}/generation_results_locomo_baseline.jsonl" \
  --eval-output "out/${RUN_ID}/evaluation_results_locomo_baseline.jsonl" \
  --summary-output "out/${RUN_ID}/evaluation_summary_locomo_baseline.json"

echo "[6/6] Evaluation: B+TF"
python evaluate_locomo.py \
  --run-id "$RUN_ID" \
  --generation-file "out/${RUN_ID}/generation_results_locomo_time_filtering.jsonl" \
  --eval-output "out/${RUN_ID}/evaluation_results_locomo_time_filtering.jsonl" \
  --summary-output "out/${RUN_ID}/evaluation_summary_locomo_time_filtering.json"

echo "Done. Outputs are under out/${RUN_ID}/"
