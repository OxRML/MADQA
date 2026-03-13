# OpenAI File Search

Non-agentic baseline using OpenAI's [Assistants API with File Search](https://platform.openai.com/docs/guides/tools-file-search), a "RAG-as-a-Service" solution where embedding, chunking, and retrieval are handled by OpenAI. Corresponds to **Managed RAG Services** in the paper (Section 4, Appendix G.2).

## Setup

```bash
pip install openai datasets tqdm
export OPENAI_API_KEY="sk-..."
```

Requires a direct OpenAI API key (starts with `sk-`). Proxy services are not supported because vector stores are server-side resources.

## Usage

```bash
# Index PDFs from the dataset
python openai_file_search_agent.py index

# Ask a single question
python openai_file_search_agent.py ask "What is the total revenue?"

# Run evaluation (outputs JSONL compatible with eval/evaluate.py)
python openai_file_search_agent.py evaluate results.jsonl --split test
```
