# Gemini File Search

Non-agentic baseline using Google's managed [File Search API](https://ai.google.dev/gemini-api/docs/file-search), a "RAG-as-a-Service" solution where embedding, chunking, and retrieval are handled by Google. Corresponds to **Managed RAG Services** in the paper (Section 4, Appendix G.2).

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-api-key"
```

## Usage

```bash
# Index PDFs from the dataset
python gemini_file_search_agent.py index

# Ask a single question
python gemini_file_search_agent.py ask "What is the total revenue?"

# Run evaluation (outputs JSONL compatible with eval/evaluate.py)
python gemini_file_search_agent.py evaluate results.jsonl --split dev
```
