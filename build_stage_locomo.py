#!/usr/bin/env python3
"""
LoCoMo-specific build stage using build_prompts_locomo.py
This version uses prompts optimized for LoCoMo dataset (no user persona).
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and override prompts BEFORE importing memblock_cli
import build_prompts_locomo
import memblock_extractor as mx

# Override the prompts in memblock_extractor
mx.PROMPT_MSG_CONTINUATION = build_prompts_locomo.PROMPT_MSG_CONTINUATION
mx.PROMPT_DIALOG_EXTRACT = build_prompts_locomo.PROMPT_DIALOG_EXTRACT
mx.PROMPT_DIALOG_CLASSIFICATION = build_prompts_locomo.PROMPT_DIALOG_CLASSIFICATION
mx.Config.PROMPT_MSG_CONTINUATION = build_prompts_locomo.PROMPT_MSG_CONTINUATION
mx.Config.PROMPT_DIALOG_EXTRACT = build_prompts_locomo.PROMPT_DIALOG_EXTRACT
mx.Config.PROMPT_DIALOG_CLASSIFICATION = build_prompts_locomo.PROMPT_DIALOG_CLASSIFICATION

print("LoCoMo Build Mode")
print("   Using prompts from build_prompts_locomo.py")
print(f"   TOPIC_CLASSIFY_TIMEOUT: {mx.Config.TOPIC_CLASSIFY_TIMEOUT}s")
print(f"   TOPIC_CLASSIFY_MAX_RETRIES: {mx.Config.TOPIC_CLASSIFY_MAX_RETRIES}")
print()

# Now import and run memblock_cli
from memblock_cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]

    # Auto-detect LoCoMo dataset format
    if "--dataset-format" not in argv:
        argv = ["--dataset-format", "locomo", *argv]

    main(argv=argv, default_stage="build")
