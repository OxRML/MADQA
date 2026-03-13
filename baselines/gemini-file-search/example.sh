#!/bin/bash
# Example for Gemini File Search Baseline

export GOOGLE_API_KEY=""
export HF_TOKEN=""

python gemini_file_search_agent.py index

MODEL="gemini-2.5-flash"
python gemini_file_search_agent.py evaluate $MODEL.jsonl --split test
