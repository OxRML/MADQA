# OCR Documents

Extracts text from MADQA corpus PDFs using Amazon Textract. The resulting OCR is used by the BM25 MLLM Agent baseline for building the search index.

## Setup

```bash
pip install -r requirements.txt
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## Usage

```bash
# Process all PDFs
make

# Or run directly
python textract.py -o ocr_output.jsonl
```

## Output

JSONL with one page per line:

```json
{"file": "document.pdf", "service": "textract", "page_number": 1, "total_pages": 3, "text": "..."}
```
