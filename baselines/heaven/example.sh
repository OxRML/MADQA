#!/bin/bash
# Example: Full HEAVEN pipeline (paper configuration)
# Uses DSE + ColQwen2.5 + VS-Pages

set -e

# Set your OpenAI API key for VLM inference
export OPENAI_API_KEY="your-api-key-here"

echo "=== HEAVEN Full Pipeline Example ==="
echo ""

# Build index with paper defaults (strongest configuration)
echo "Step 1: Building index with DSE + ColQwen2.5 + VS-Pages + DLA..."
python heaven.py index \
    --stage1-encoder dse \
    --stage2-encoder colqwen2.5 \
    --use-vs-pages \
    --reduction-factor 15 \
    --use-dla \
    --limit 100 \
    --device 0

# Ask a question (paper defaults)
echo ""
echo "Step 2: Asking question..."
python heaven.py ask "What is the total revenue?" \
    --stage1-encoder dse \
    --stage2-encoder colqwen2.5 \
    --vlm-model gpt-4o \
    --stage1-k 200 \
    --stage2-k 5 \
    --alpha 0.1 \
    --beta 0.3 \
    --vs-page-filter-ratio 0.5 \
    --query-filter-ratio 0.25 \
    --device 0 \
    -o example_result.json

echo ""
echo "Done! Result saved to example_result.json"

# Optionally run full evaluation (paper defaults)
# echo ""
# echo "Step 3: Running evaluation on test set..."
# python heaven.py evaluate results.jsonl \
#     --split test \
#     --stage1-encoder dse \
#     --stage2-encoder colqwen2.5 \
#     --vlm-model gpt-4o \
#     --stage1-k 200 \
#     --stage2-k 5 \
#     --alpha 0.1 \
#     --beta 0.3 \
#     --vs-page-filter-ratio 0.5 \
#     --query-filter-ratio 0.25 \
#     --device 0
