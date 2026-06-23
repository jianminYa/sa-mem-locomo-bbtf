# LoCoMo B / B+TF Reproduction on a Server

This package is based on `RichardWang11/SA-Mem-Research` `origin/main` commit
`bdf50de` (`upload build_impl_graph.py`), plus small non-graph compatibility
patches needed to run LoCoMo B / B+TF without graph modules.

## What B / B+TF Means Here

- `B`: baseline plain vector retrieval, generated from `retrieval_baseline.jsonl`.
- `B+TF`: enhanced retrieval with query parsing and temporal filtering, generated
  from `retrieval_enhanced.jsonl`.
- Graph is disabled for both. Do not use `--enable-graph`, `--graph-expand`,
  `--graph-index-expand`, or `--use-graph-context` for this run.

## Configuration Used

- Dataset: `dataset/locomo10.json`
- Build model: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`
- Retrieval top-k: `5`
- Generation answer top-n: `5`
- Generation context: `content`
- LoCoMo category 5: skipped by the existing retrieval/generation code

Note: the upstream README does not expose retrieval `top-k` as a CLI argument.
This package includes `scripts/retrieve_locomo_b_btf.py` to set
`Config.TOP_K_RETRIEVE=5` before calling the official retrieval entrypoint.

## Package Compatibility Notes

The new upstream `build_impl_graph.py` imports graph modules at import time:

- `graph_storage.py`
- `graph_entities_extractor.py`

The public repo still does not include those files. This package includes
minimal no-op stubs for non-graph B / B+TF. They are only for import
compatibility. Replace them with the author's real files before running graph or
HTM experiments.

This package also applies two small LoCoMo fixes:

- `build_stage_locomo.py` now also overwrites `mx.Config.PROMPT_*`, not only the
  module-level prompt variables.
- `build_impl_graph.py` accepts both `mentions` and `explicit_mentions` from the
  first-pass extraction prompt.

## DigitalOcean Setup

Example on Ubuntu:

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip unzip tmux
unzip SA-Mem-LoCoMo-B-BTF-server.zip
cd SA-Mem-Research
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-locomo.txt
python -m nltk.downloader punkt punkt_tab
cp .env.example .env
nano .env
```

Fill in at least:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
```

If you use an OpenAI-compatible proxy, replace `OPENAI_BASE_URL` with that
endpoint. The run script sets the model through CLI flags; override with shell
variables such as `LLM_MODEL=...` and `EMBEDDING_MODEL=...` when needed.

## Recommended First Test: One Conversation

Run inside `tmux` or `screen` because build/generation can take a while.

```bash
source .venv/bin/activate
LIMIT_CONVERSATIONS=1 RUN_ID=locomo_b_btf_sample1 bash scripts/run_locomo_b_btf.sh
```

Expected outputs:

```text
out/locomo_b_btf_sample1/final_boxes_content.jsonl
out/locomo_b_btf_sample1/retrieval_baseline.jsonl
out/locomo_b_btf_sample1/retrieval_enhanced.jsonl
out/locomo_b_btf_sample1/generation_results_locomo_baseline.jsonl
out/locomo_b_btf_sample1/generation_results_locomo_time_filtering.jsonl
out/locomo_b_btf_sample1/evaluation_summary_locomo_baseline.json
out/locomo_b_btf_sample1/evaluation_summary_locomo_time_filtering.json
```

## Full LoCoMo10 Run

```bash
source .venv/bin/activate
RUN_ID=locomo_b_btf_full bash scripts/run_locomo_b_btf.sh
```

Full LoCoMo10 has 1540 non-category-5 QA pairs. Expect several hours depending
on API latency and rate limits. The build stage calls the LLM for topic
splitting and event extraction; generation then calls the LLM once per
question per retrieval variant.

## Stage-by-Stage Commands

Build only:

```bash
python build_stage_locomo.py \
  --stage build \
  --run-id locomo_b_btf_full \
  --raw-data-file dataset/locomo10.json \
  --limit-conversations -1 \
  --disable-graph \
  --llm-model gpt-4o-mini \
  --embedding-model text-embedding-3-small
```

Retrieval only:

```bash
python scripts/retrieve_locomo_b_btf.py \
  --top-k 5 \
  --mode both \
  --run-id locomo_b_btf_full \
  --raw-data-file dataset/locomo10.json \
  --final-content-file out/locomo_b_btf_full/final_boxes_content.jsonl \
  --limit-conversations -1 \
  --overwrite
```

Generate and evaluate B:

```bash
python generate_stage_locomo.py \
  --run-id locomo_b_btf_full \
  --raw-data-file dataset/locomo10.json \
  --final-content-file out/locomo_b_btf_full/final_boxes_content.jsonl \
  --retrieval-file out/locomo_b_btf_full/retrieval_baseline.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --limit-conversations -1 \
  --output-suffix baseline

python evaluate_locomo.py \
  --run-id locomo_b_btf_full \
  --generation-file out/locomo_b_btf_full/generation_results_locomo_baseline.jsonl \
  --eval-output out/locomo_b_btf_full/evaluation_results_locomo_baseline.jsonl \
  --summary-output out/locomo_b_btf_full/evaluation_summary_locomo_baseline.json
```

Generate and evaluate B+TF:

```bash
python generate_stage_locomo.py \
  --run-id locomo_b_btf_full \
  --raw-data-file dataset/locomo10.json \
  --final-content-file out/locomo_b_btf_full/final_boxes_content.jsonl \
  --retrieval-file out/locomo_b_btf_full/retrieval_enhanced.jsonl \
  --answer-topn 5 \
  --text-modes content \
  --limit-conversations -1 \
  --output-suffix time_filtering

python evaluate_locomo.py \
  --run-id locomo_b_btf_full \
  --generation-file out/locomo_b_btf_full/generation_results_locomo_time_filtering.jsonl \
  --eval-output out/locomo_b_btf_full/evaluation_results_locomo_time_filtering.jsonl \
  --summary-output out/locomo_b_btf_full/evaluation_summary_locomo_time_filtering.json
```

## Approximate Runtime

For one conversation, expect roughly:

- Build: tens of minutes, depending on LLM latency/rate limits.
- Retrieval B/B+TF: minutes to tens of minutes.
- Generation B + B+TF: tens of minutes.

For full LoCoMo10, expect a multi-hour run.
