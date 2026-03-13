# Recursive Language Models (RLM)

Agentic baseline using [Recursive Language Models](https://github.com/alexzhang13/rlm) (Zhang et al., 2025a). A task-agnostic approach that enables LLMs to programmatically examine and recursively process the document collection. Corresponds to **Recursive Language Models** in the paper (Section 4, Appendix G.5).

## Setup

```bash
# Install RLM from source
git clone https://github.com/alexzhang13/rlm.git
cd rlm && pip install -e . && cd ..

# Set API keys
export OPENAI_API_KEY="sk-..."        # OpenAI backend
export ANTHROPIC_API_KEY="sk-ant-..." # Anthropic backend
```

Uses a markdown corpus from HuggingFace: `agentic-document-ai/pdfs-to-markdown-mistral-ocr`.

## Usage

```bash
# Single question
python rlm_agent.py "What is the registration number?" -v

# Process dataset (outputs JSONL compatible with eval/evaluate.py)
python process_dataset.py -o results.jsonl --split test

# Different model/backend
python process_dataset.py -o results.jsonl --backend anthropic --model claude-sonnet-4-5-20250929
```

## Options

| Flag | Description |
|------|-------------|
| `--backend` | RLM backend: openai, anthropic (default: openai) |
| `--model` | Model name (default: gpt-5-mini-2025-08-07) |
| `--split` | Dataset split: dev, test, train |
| `--resume` | Resume from existing output file |
