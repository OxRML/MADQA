# Evaluation

Command-line evaluation harness for the [MADQA benchmark](https://huggingface.co/datasets/OxRML/MADQA). Implements the metrics described in Section 3 of the paper.

## Setup

```bash
pip install -r requirements.txt
```

For semantic accuracy with the LLM judge, set:

```bash
export GOOGLE_API_KEY="your-api-key"
```

## Usage

```bash
# Basic evaluation against dev split
python evaluate.py results.jsonl

# Evaluate against test split
python evaluate.py results.jsonl --split test

# Breakdown by category or domain
python evaluate.py results.jsonl --by-category --by-domain

# Compare multiple systems side-by-side
python evaluate.py model1.jsonl model2.jsonl model3.jsonl --compare

# Use semantic accuracy (ANLS* + LLM judge)
python evaluate.py results.jsonl --semantic

# Output as JSON
python evaluate.py results.jsonl --json
```

## Metrics

| Metric | Paper | Description |
|--------|-------|-------------|
| **Accuracy** | §3.1 | Fraction with ANLS* >= 0.5 |
| **ANLS\*** | §3.1 | Answer Normalized Levenshtein Similarity (0–1) |
| **Semantic Accuracy** | §3.1 | ANLS* + LLM judge with bias correction (`--semantic`) |
| **Document F1** | §3.2 | Citation accuracy at document level |
| **Page F1** | §3.2 | Citation accuracy at page level |
| **Kuiper Statistic** | §3.3 | Effort-accuracy calibration (lower = better) |
| **Wasted Effort Ratio** | §3.3 | mean_steps(incorrect) / mean_steps(correct) |

Results are also broken down by **hop type** (single, cross-page, cross-document) by default.

### Semantic Accuracy

When `--semantic` is enabled, the evaluator:

1. Computes ANLS* for each prediction
2. If ANLS* < 1.0, calls a Gemini LLM judge to assess semantic equivalence
3. Applies Rogan-Gladen bias correction using calibration from a 200-sample human evaluation
4. Reports 95% confidence intervals

Based on *"How to Correctly Report LLM-as-a-Judge Evaluations"* ([arXiv:2511.21140](https://arxiv.org/abs/2511.21140)).

## Input Format

JSONL with one prediction per line:

```json
{"question": "What is the total revenue?", "answer": "$1.2M", "citations": [{"document": "report.pdf", "page": 5}], "search_history": [{"query": "revenue", "num_results": 3}]}
```

Required fields:
- `question` — question text (used to match with gold standard)
- `answer` — predicted answer (string or list)

Optional fields:
- `id` — question ID (fallback if question text doesn't match)
- `citations` — list of `{document, page}` or `{file, page}` for citation F1
- `search_history` — list of search steps (for Kuiper / wasted effort analysis)
- `iterations` — alternative to `search_history` length

## Python API

```python
from metrics import anls_star, anls_star_llm, citation_f1, kuiper_statistic

# ANLS* score
score = anls_star("$1.2 million", [["$1.2M", "1.2 million dollars"]])

# Semantic accuracy with LLM judge
result = anls_star_llm("$1.2 million", [["$1.2M"]], question="What is the total revenue?")
# result['score'] -> 1.0, result['used_llm'] -> True/False

# Citation F1
f1 = citation_f1(
    predicted=[{"document": "a.pdf", "page": 1}],
    gold_locations=[{"document": "a.pdf", "page": 1}, {"document": "a.pdf", "page": 2}],
    level='page'
)

# Kuiper statistic
results = [{"steps": 3, "correct": True}, {"steps": 7, "correct": False}]
kuiper = kuiper_statistic(results)
```
