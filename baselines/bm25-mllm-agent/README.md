# BM25 MLLM Agent

Iterative agentic system coupling text-based BM25 retrieval with a multimodal LLM. The agent formulates search queries, receives rendered page images, and decides whether to search further or answer. Corresponds to **BM25 MLLM Agent** in the paper (Section 4, Appendix G.1).

Supports OpenAI, Anthropic, Google Gemini, and self-hosted models via vLLM.

## Setup

```bash
pip install -r requirements.txt
```

Set an API key for the provider you want to use:

```bash
export OPENAI_API_KEY='sk-...'       # OpenAI
export ANTHROPIC_API_KEY='sk-ant-...' # Anthropic
export GOOGLE_API_KEY='...'           # Gemini
```

## Usage

```bash
# OpenAI
python openai_agent.py "What is the incorporation date of JOY HOUSE ENTERPRISES?"

# Anthropic
python anthropic_agent.py "What is the incorporation date of JOY HOUSE ENTERPRISES?"

# Gemini
python gemini_agent.py "What is the incorporation date of JOY HOUSE ENTERPRISES?"

# vLLM (self-hosted models via OpenAI-compatible API)
python vllm_agent.py "What is the incorporation date of JOY HOUSE ENTERPRISES?" \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --base-url http://localhost:8000/v1
```

### Options

- `--model` — Model name
- `--top-k` — Pages returned per search (default: 3)
- `--max-iterations` — Max search cycles (default: 5)
- `--output` — Save results as JSON

For the vLLM variant, you can also set `VLLM_BASE_URL` and `VLLM_API_KEY` environment variables.

### Batch evaluation

```bash
python process_dataset.py -o results.jsonl --provider openai
```

Output is compatible with `eval/evaluate.py`.

## How It Works

1. Whoosh indexes OCR text from the corpus (cached to `data/whoosh_index/`)
2. Agent issues text search queries
3. Only page images are returned — no text is provided to the agent
4. Agent analyzes images and decides: refine search or produce an answer
