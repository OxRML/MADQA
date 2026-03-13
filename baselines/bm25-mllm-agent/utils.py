"""
Shared utilities for BM25 MLLM Agent baselines.

Provides PDF/image helpers, the Whoosh search engine, and common Pydantic models
used across all provider-specific agent implementations.
"""

import json
import os
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional

from pydantic import BaseModel
from PIL import Image
from pdf2image import convert_from_path
from datasets import load_dataset, DownloadManager, disable_progress_bar
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.qparser import QueryParser

# Disable progress bars
disable_progress_bar()

# Module-level cache for document URL mapping and DownloadManager
_doc_urls: dict | None = None
_dm: DownloadManager | None = None


def _get_doc_urls(dataset_name: str = "OxRML/MADQA") -> tuple[dict, DownloadManager]:
    """Return (doc_urls dict, DownloadManager), loading once and caching."""
    global _doc_urls, _dm
    if _doc_urls is None:
        docs = load_dataset(dataset_name, "documents", split="links")
        _doc_urls = {r["document"]: r["url"] for r in docs}
        _dm = DownloadManager()
    return _doc_urls, _dm


# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """Citation for a source document."""
    file: str  # Must be the exact PDF filename (e.g., "1007969.pdf")
    page: int  # Page number


class Answer(BaseModel):
    """Structured answer with citations."""
    answer: List[str]
    citations: List[Citation]


# ---------------------------------------------------------------------------
# Whoosh search engine
# ---------------------------------------------------------------------------

class WhooshSearchEngine:
    """Whoosh-based search engine for OCR results."""

    def __init__(self, ocr_jsonl_path: str, index_dir: str = None):
        """Initialize search engine and build/load index."""
        if index_dir is None:
            index_dir = os.path.join(os.path.dirname(ocr_jsonl_path), "whoosh_index")

        self.index_dir = index_dir

        if exists_in(index_dir):
            print(f"Loading cached index from {index_dir}...")
            self.ix = open_dir(index_dir)
            print("Index loaded")
        else:
            os.makedirs(index_dir, exist_ok=True)

            schema = Schema(
                file=ID(stored=True),
                page_number=NUMERIC(stored=True),
                total_pages=NUMERIC(stored=True),
                text=TEXT(stored=False)
            )

            print(f"Building search index from {ocr_jsonl_path}...")
            ix = create_in(index_dir, schema)
            writer = ix.writer(limitmb=512, procs=1, multisegment=False)

            count = 0
            with open(ocr_jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        page = json.loads(line)
                        writer.add_document(
                            file=page['file'],
                            page_number=page['page_number'],
                            total_pages=page['total_pages'],
                            text=page.get('text', '')
                        )
                        count += 1

            writer.commit()
            print(f"Indexed {count} pages (cached to {index_dir})")
            self.ix = ix

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search and return page metadata."""
        with self.ix.searcher() as searcher:
            query_parser = QueryParser("text", self.ix.schema)
            parsed_query = query_parser.parse(query)
            results = searcher.search(parsed_query, limit=top_k)

            return [
                {
                    "file": hit['file'],
                    "page_number": hit['page_number'],
                    "total_pages": hit['total_pages']
                }
                for hit in results
            ]


# ---------------------------------------------------------------------------
# PDF / image helpers
# ---------------------------------------------------------------------------


def get_pdf_page_as_png(
    pdf_filename: str, 
    page_number: int, 
    dpi: int = 200,
    dataset_name: str = "OxRML/MADQA"
) -> Image.Image:
    """
    Get a specific page from a PDF in the HuggingFace dataset as a PIL Image.
    
    Args:
        pdf_filename: Name of the PDF file (e.g., "document.pdf")
        page_number: Page number (1-indexed)
        dpi: DPI for rendering (default: 200)
        dataset_name: HuggingFace dataset name
        
    Returns:
        PIL Image object
    """
    doc_urls, dm = _get_doc_urls(dataset_name)

    if pdf_filename not in doc_urls:
        raise ValueError(f"PDF '{pdf_filename}' not found in dataset")

    pdf_path = dm.download(doc_urls[pdf_filename])

    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
    if not images:
        raise ValueError(f"Could not extract page {page_number} from {pdf_filename}")
    
    return images[0]


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string (PNG format)."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

