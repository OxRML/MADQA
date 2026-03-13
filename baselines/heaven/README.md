# HEAVEN

Implementation of **HEAVEN** (Hybrid-Vector Retrieval for Visually Rich Documents), a non-agentic two-stage visual retrieval baseline. Corresponds to **HEAVEN** in the paper (Section 4, Appendix G.3).

Based on [juyeonnn/HEAVEN](https://github.com/juyeonnn/HEAVEN) and the paper *"Hybrid-Vector Retrieval for Visually Rich Documents"* ([arXiv:2510.22215](https://arxiv.org/abs/2510.22215)).

## Pipeline

1. **Stage 1 (Single-Vector):** Fast candidate retrieval using DSE encoder, optionally with VS-Page filtering
2. **Stage 2 (Multi-Vector):** Re-ranking with ColQwen2.5 via MaxSim scoring and query token filtering
3. **Answer generation:** Top pages are passed to a MLLM (e.g., GPT-4o)

| Stage | Encoder | Model |
|-------|---------|-------|
| Stage 1 | DSE | MrLight/dse-qwen2-2b-mrl-v1 |
| Stage 2 | ColQwen2.5 | vidore/colqwen2.5-v0.2 |

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-key"
```

## Usage

```bash
# Build index
python heaven.py index \
    --stage1-encoder dse \
    --stage2-encoder colqwen2.5 \
    --use-vs-pages \
    --reduction-factor 15

# Ask a question
python heaven.py ask "What is the total revenue?" \
    --stage1-encoder dse \
    --stage2-encoder colqwen2.5 \
    --vlm-model gpt-4o \
    -o result.json
```

Output is compatible with `eval/evaluate.py`.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--stage1-encoder` | dse | Stage 1 encoder |
| `--stage2-encoder` | colqwen2.5 | Stage 2 re-ranker |
| `--reduction-factor` | 15 | Pages per VS-page |
| `--stage1-k` | 100 | Stage 1 candidates |
| `--stage2-k` | 5–10 | Final results |
