#!/usr/bin/env python3
"""
HEAVEN Document QA Agent

Based on: https://github.com/juyeonnn/HEAVEN
Paper: "Hybrid-Vector Retrieval for Visually Rich Documents: 
        Combining Single-Vector Efficiency and Multi-Vector Accuracy"
        https://arxiv.org/pdf/2510.22215

This agent implements the full HEAVEN pipeline:
1. VS-Pages (Visually-Summarized Pages) for efficient Stage 1 filtering
2. Query token filtering for efficient Stage 2 re-ranking
3. Two-stage hybrid retrieval (DSE + ColQwen2.5)
4. VLM-based question answering on retrieved pages

Output format is compatible with eval/evaluate.py
"""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm


class HEAVENAgent:
    """Agent using HEAVEN two-stage retrieval for document QA."""
    
    def __init__(
        self,
        index_dir: str = "./heaven_index",
        stage1_encoder_type: str = "dse",
        stage2_encoder_type: Optional[str] = "colqwen2.5",
        vlm_model: str = "gpt-4o",
        stage1_k: int = 200,  # Paper default: 200
        stage2_k: int = 5,
        device: str = "cuda",
        openai_api_key: Optional[str] = None,
        # HEAVEN-specific parameters (paper defaults)
        use_vs_pages: bool = True,
        reduction_factor: int = 15,
        vs_page_filter_ratio: float = 0.5,
        use_query_filtering: bool = True,
        query_filter_ratio: float = 0.25,  # Paper default: 0.25
        alpha: float = 0.1,  # Paper default: 0.1 (Stage 1 hybrid weight)
        beta: float = 0.3,   # Paper default: 0.3 (Stage 2 score combination weight)
    ):
        """
        Initialize HEAVEN agent.
        
        Args:
            index_dir: Directory for index storage
            stage1_encoder_type: Encoder for fast retrieval (dse, siglip)
            stage2_encoder_type: Encoder for re-ranking (colqwen2.5), None for single-stage
            vlm_model: Vision-language model for QA (gpt-4o, etc.)
            stage1_k: Number of candidates from stage 1
            stage2_k: Number of final results for QA
            device: Device for encoders (cuda/cpu)
            openai_api_key: API key for VLM (uses env var if not provided)
            
            HEAVEN-specific parameters:
            use_vs_pages: Whether to use VS-Pages for efficient Stage 1
            reduction_factor: Number of pages per VS-page (paper default: 15)
            vs_page_filter_ratio: Fraction of VS-pages to consider
            use_query_filtering: Whether to filter query tokens in Stage 2
            query_filter_ratio: Fraction of query tokens to keep
            alpha: Stage 1 hybrid weight (0 = visual only)
            beta: Stage 2 score combination weight
        """
        self.index_dir = Path(index_dir)
        self.stage1_encoder_type = stage1_encoder_type
        self.stage2_encoder_type = stage2_encoder_type
        self.vlm_model = vlm_model
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.device = device
        
        # HEAVEN-specific parameters
        self.use_vs_pages = use_vs_pages
        self.reduction_factor = reduction_factor
        self.vs_page_filter_ratio = vs_page_filter_ratio
        self.use_query_filtering = use_query_filtering
        self.query_filter_ratio = query_filter_ratio
        self.alpha = alpha
        self.beta = beta
        
        # Initialize OpenAI client for VLM
        self.openai_client = None
        if openai_api_key or os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Lazy-loaded components
        self._index = None
        self._retriever = None
        self._stage1_encoder = None
        self._stage2_encoder = None
        
        print(f"Initialized HEAVEN agent")
        print(f"  Stage 1 encoder: {stage1_encoder_type}")
        print(f"  Stage 2 encoder: {stage2_encoder_type or 'None (single-stage)'}")
        print(f"  VS-Pages: {use_vs_pages} (reduction_factor={reduction_factor})")
        print(f"  Query filtering: {use_query_filtering} (ratio={query_filter_ratio})")
        print(f"  VLM model: {vlm_model}")
    
    @property
    def index(self):
        """Lazy-load document index."""
        if self._index is None:
            from index import DocumentIndex
            self._index = DocumentIndex(self.index_dir)
            if (self.index_dir / "index_metadata.json").exists():
                self._index.load()
        return self._index
    
    @property
    def stage1_encoder(self):
        """Lazy-load stage 1 encoder."""
        if self._stage1_encoder is None:
            from encoders import get_encoder
            self._stage1_encoder = get_encoder(self.stage1_encoder_type, device=self.device)
        return self._stage1_encoder
    
    @property
    def stage2_encoder(self):
        """Lazy-load stage 2 encoder."""
        if self._stage2_encoder is None and self.stage2_encoder_type:
            from encoders import get_encoder
            self._stage2_encoder = get_encoder(self.stage2_encoder_type, device=self.device)
        return self._stage2_encoder
    
    @property
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None:
            from retrieval import HEAVENRetriever
            self._retriever = HEAVENRetriever(
                index=self.index,
                stage1_encoder=self.stage1_encoder,
                stage2_encoder=self.stage2_encoder,
                stage1_k=self.stage1_k,
                stage2_k=self.stage2_k,
                alpha=self.alpha,
                beta=self.beta,
                vs_page_filter_ratio=self.vs_page_filter_ratio,
                query_filter_ratio=self.query_filter_ratio,
                use_query_filtering=self.use_query_filtering
            )
        return self._retriever
    
    def download_pdfs_from_hf(
        self,
        repo_id: str = "OxRML/MADQA",
        local_dir: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Path]:
        """Download PDFs from HuggingFace dataset."""
        import shutil
        from datasets import load_dataset, DownloadManager
        
        local_dir = Path(local_dir or self.index_dir / "pdf_cache")
        local_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading document URLs from {repo_id}...")
        docs = load_dataset(repo_id, "documents", split="links")
        doc_urls = {r["document"]: r["url"] for r in docs}
        
        filenames = sorted(doc_urls.keys())
        if limit:
            filenames = filenames[:limit]
        
        print(f"Found {len(filenames)} PDF files")
        
        dm = DownloadManager()
        local_paths = []
        for filename in tqdm(filenames, desc="Downloading PDFs"):
            try:
                cached_path = dm.download(doc_urls[filename])
                dest = local_dir / filename
                if not dest.exists():
                    shutil.copy2(cached_path, dest)
                local_paths.append(dest)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        
        return local_paths
    
    def build_index(
        self,
        pdf_dir: Optional[Path] = None,
        repo_id: str = "OxRML/MADQA",
        limit: Optional[int] = None,
        batch_size: int = 4,
        use_dla: bool = False,
        skip_download: bool = False,
        skip_pdf_conversion: bool = False
    ):
        """
        Build document index from PDFs.
        
        Args:
            pdf_dir: Directory with PDFs (downloads from HF if None)
            repo_id: HuggingFace repo to download from
            limit: Maximum number of PDFs to index
            batch_size: Batch size for encoding
            use_dla: Whether to use Document Layout Analysis for VS-pages
            skip_download: Skip downloading PDFs (use existing pdf_cache)
            skip_pdf_conversion: Skip PDF to image conversion (use existing images)
        """
        from index import DocumentIndex
        
        # Download PDFs if needed (unless skipped)
        if pdf_dir is None:
            pdf_dir = self.index_dir / "pdf_cache"
            if not skip_download:
                pdf_paths = self.download_pdfs_from_hf(repo_id=repo_id, limit=limit)
            else:
                print(f"Skipping PDF download, using existing files in {pdf_dir}")
        
        # Build index with HEAVEN-specific options
        self._index = DocumentIndex(self.index_dir)
        self._index.build_index(
            pdf_dir=pdf_dir,
            stage1_encoder=self.stage1_encoder,
            stage2_encoder=self.stage2_encoder,
            limit=limit,
            batch_size=batch_size,
            save_images=True,
            use_vs_pages=self.use_vs_pages,
            reduction_factor=self.reduction_factor,
            use_dla=use_dla,
            skip_pdf_conversion=skip_pdf_conversion
        )
        self._index.save()
        
        print(f"Index built with {len(self._index.doc_metadata)} document pages")
        if self._index.vs_page_embeddings is not None:
            print(f"  VS-pages: {len(self._index.vs_page_embeddings)}")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _call_vlm(
        self,
        question: str,
        images: List[Image.Image],
        doc_info: List[Dict[str, Any]]
    ) -> str:
        """
        Call VLM to answer question based on retrieved images.
        
        Args:
            question: The question to answer
            images: List of document page images
            doc_info: List of document metadata dicts
        
        Returns:
            VLM response text
        """
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")
        
        # Build context string
        context_parts = []
        for i, info in enumerate(doc_info):
            context_parts.append(f"Image {i+1}: {info['pdf_name']}, Page {info['page_num'] + 1}")
        context = "\n".join(context_parts)
        
        # Build message with images
        content = [
            {
                "type": "text",
                "text": f"""You are a document QA assistant. Answer the question based ONLY on the provided document images.

Document sources:
{context}

Question: {question}

Instructions:
1. Carefully examine ALL provided document images
2. Find the specific information that answers the question
3. Respond with a JSON object in this exact format:
{{
  "answer": ["answer value 1", "answer value 2", ...],
  "citations": [
    {{"file": "exact_filename.pdf", "page": 1}},
    {{"file": "another_file.pdf", "page": 3}}
  ]
}}

Where:
- answer: list of answer values (use as few words as possible, exact document wording preferred)
- citations: list of sources with exact PDF filename and page number (1-indexed)

IMPORTANT: The answer to the question is definitely in the documents. Examine all provided pages carefully.
"""
            }
        ]
        
        # Add images
        for img in images:
            base64_img = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "high"
                }
            })
        
        # Use max_completion_tokens for newer models (gpt-5, o1, o3, etc.)
        # Use max_tokens for older models (gpt-4o, gpt-4, etc.)
        newer_models = ('gpt-5', 'o1', 'o3', 'o4')
        if any(self.vlm_model.startswith(m) for m in newer_models):
            response = self.openai_client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=1024
            )
        else:
            response = self.openai_client.chat.completions.create(
                model=self.vlm_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1024
            )
        
        return response.choices[0].message.content
    
    def _parse_answer(self, text: str) -> Dict[str, Any]:
        """Parse the answer from VLM response."""
        # Try to extract JSON
        json_match = re.search(r'\{[\s\S]*"answer"[\s\S]*"citations"[\s\S]*\}', text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if "answer" in parsed and "citations" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Fallback: return text as answer
        return {
            "answer": [text.strip()],
            "citations": []
        }
    
    def answer_question(self, question: str, question_id: str = "") -> Dict[str, Any]:
        """
        Answer a question using HEAVEN retrieval + VLM.
        
        Args:
            question: The question to answer
            question_id: Optional question ID for output
        
        Returns:
            Dict with answer, citations, retrieval_info compatible with eval/evaluate.py
        """
        print(f"\nQuestion: {question}")
        
        # Retrieve relevant documents
        results, retrieval_metadata = self.retriever.retrieve_with_scores(
            query=question,
            k=self.stage2_k
        )
        
        print(f"Retrieved {len(results)} documents")
        for i, r in enumerate(results[:3]):
            print(f"  {i+1}. {r.pdf_name} p{r.page_num + 1} (score: {r.score:.4f})")
        
        # Get images and metadata for retrieved docs
        images = []
        doc_info = []
        for result in results:
            info = self.index.get_document_info(result.index)
            if info["image"] is not None:
                images.append(info["image"])
                doc_info.append(info)
        
        if not images:
            return {
                "id": question_id,
                "question": question,
                "answer": ["No document images available"],
                "citations": [],
                "model": self.vlm_model,
                "error": "No images in index"
            }
        
        # Call VLM
        try:
            vlm_response = self._call_vlm(question, images, doc_info)
            answer_data = self._parse_answer(vlm_response)
        except Exception as e:
            print(f"VLM error: {e}")
            return {
                "id": question_id,
                "question": question,
                "answer": ["Error calling VLM"],
                "citations": [],
                "model": self.vlm_model,
                "error": str(e)
            }
        
        # Build output compatible with eval/evaluate.py
        return {
            "id": question_id,
            "question": question,
            "answer": answer_data["answer"],
            "citations": answer_data["citations"],
            "model": self.vlm_model
        }


def main():
    parser = argparse.ArgumentParser(
        description="HEAVEN two-stage hybrid-vector retrieval for document QA"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build document index')
    index_parser.add_argument('--repo', default='OxRML/MADQA',
                              help='HuggingFace repository ID')
    index_parser.add_argument('--pdf-dir', help='Local directory with PDFs (skips download)')
    index_parser.add_argument('--limit', type=int, help='Limit number of PDFs')
    index_parser.add_argument('--index-dir', default='./heaven_index',
                              help='Index output directory')
    index_parser.add_argument('--stage1-encoder', default='dse',
                              choices=['siglip', 'dse'],
                              help='Stage 1 encoder (paper uses DSE)')
    index_parser.add_argument('--stage2-encoder', default='colqwen2.5',
                              choices=['colqwen', 'colqwen2.5', 'none'],
                              help='Stage 2 encoder (paper uses ColQwen2.5)')
    index_parser.add_argument('--device', default='0', help='Device')
    index_parser.add_argument('--batch-size', type=int, default=16,
                              help='Batch size for Stage 1 encoding (default for 140GB+ VRAM)')
    # VS-Page options
    index_parser.add_argument('--use-vs-pages', action='store_true', default=True,
                              help='Enable VS-Pages for efficient retrieval')
    index_parser.add_argument('--no-vs-pages', action='store_false', dest='use_vs_pages',
                              help='Disable VS-Pages')
    index_parser.add_argument('--reduction-factor', type=int, default=15,
                              help='Pages per VS-page (paper default: 15)')
    index_parser.add_argument('--use-dla', action='store_true',
                              help='Use Document Layout Analysis for VS-pages')
    # Skip options for faster re-indexing
    index_parser.add_argument('--skip-download', action='store_true',
                              help='Skip PDF download (use existing pdf_cache)')
    index_parser.add_argument('--skip-pdf-conversion', action='store_true',
                              help='Skip PDF conversion (use existing saved images)')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('question', help='Question to answer')
    ask_parser.add_argument('--index-dir', default='./heaven_index',
                            help='Index directory')
    ask_parser.add_argument('--stage1-encoder', default='dse',
                            help='Stage 1 encoder type')
    ask_parser.add_argument('--stage2-encoder', default='colqwen2.5',
                            help='Stage 2 encoder type')
    ask_parser.add_argument('--vlm-model', default='gpt-4o',
                            help='VLM model for QA')
    ask_parser.add_argument('--stage1-k', type=int, default=200,
                            help='Stage 1 candidates (paper default: 200)')
    ask_parser.add_argument('--stage2-k', type=int, default=5,
                            help='Final results')
    ask_parser.add_argument('--device', default='cuda', help='Device')
    ask_parser.add_argument('--output', '-o', help='Output JSON file')
    # HEAVEN-specific options (paper defaults)
    ask_parser.add_argument('--vs-page-filter-ratio', type=float, default=0.5,
                            help='Fraction of VS-pages to consider (paper default: 0.5)')
    ask_parser.add_argument('--query-filter-ratio', type=float, default=0.25,
                            help='Fraction of query tokens to keep (paper default: 0.25)')
    ask_parser.add_argument('--no-query-filtering', action='store_true',
                            help='Disable query token filtering')
    ask_parser.add_argument('--alpha', type=float, default=0.1,
                            help='Stage 1 hybrid weight (paper default: 0.1)')
    ask_parser.add_argument('--beta', type=float, default=0.3,
                            help='Stage 2 score combination weight (paper default: 0.3)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation on questions')
    eval_parser.add_argument('output', help='Output JSONL file')
    eval_parser.add_argument('--dataset', default='OxRML/MADQA',
                             help='HuggingFace dataset name')
    eval_parser.add_argument('--split', default='test',
                             help='Dataset split (dev, test, train)')
    eval_parser.add_argument('--index-dir', default='./heaven_index',
                             help='Index directory')
    eval_parser.add_argument('--stage1-encoder', default='dse',
                             help='Stage 1 encoder type')
    eval_parser.add_argument('--stage2-encoder', default='colqwen2.5',
                             help='Stage 2 encoder type')
    eval_parser.add_argument('--vlm-model', default='gpt-4o',
                             help='VLM model for QA')
    eval_parser.add_argument('--stage1-k', type=int, default=200,
                             help='Stage 1 candidates (paper default: 200)')
    eval_parser.add_argument('--stage2-k', type=int, default=5,
                             help='Final results')
    eval_parser.add_argument('--device', default='cuda', help='Device')
    eval_parser.add_argument('--limit', type=int, help='Limit number of questions')
    # HEAVEN-specific options (paper defaults)
    eval_parser.add_argument('--vs-page-filter-ratio', type=float, default=0.5,
                             help='Fraction of VS-pages to consider (paper default: 0.5)')
    eval_parser.add_argument('--query-filter-ratio', type=float, default=0.25,
                             help='Fraction of query tokens to keep (paper default: 0.25)')
    eval_parser.add_argument('--no-query-filtering', action='store_true',
                             help='Disable query token filtering')
    eval_parser.add_argument('--alpha', type=float, default=0.1,
                             help='Stage 1 hybrid weight (paper default: 0.1)')
    eval_parser.add_argument('--beta', type=float, default=0.3,
                             help='Stage 2 score combination weight (paper default: 0.3)')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        stage2 = None if args.stage2_encoder == 'none' else args.stage2_encoder
        
        agent = HEAVENAgent(
            index_dir=args.index_dir,
            stage1_encoder_type=args.stage1_encoder,
            stage2_encoder_type=stage2,
            device=args.device,
            use_vs_pages=args.use_vs_pages,
            reduction_factor=args.reduction_factor
        )
        
        pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
        agent.build_index(
            pdf_dir=pdf_dir,
            repo_id=args.repo,
            limit=args.limit,
            batch_size=args.batch_size,
            use_dla=args.use_dla,
            skip_download=args.skip_download,
            skip_pdf_conversion=args.skip_pdf_conversion
        )
        
    elif args.command == 'ask':
        stage2 = None if args.stage2_encoder == 'none' else args.stage2_encoder
        
        agent = HEAVENAgent(
            index_dir=args.index_dir,
            stage1_encoder_type=args.stage1_encoder,
            stage2_encoder_type=stage2,
            vlm_model=args.vlm_model,
            stage1_k=args.stage1_k,
            stage2_k=args.stage2_k,
            device=args.device,
            vs_page_filter_ratio=args.vs_page_filter_ratio,
            query_filter_ratio=args.query_filter_ratio,
            use_query_filtering=not args.no_query_filtering,
            alpha=args.alpha,
            beta=args.beta
        )
        
        result = agent.answer_question(args.question)
        
        print("\n" + "=" * 80)
        print("QUESTION:", result["question"])
        print("\nANSWER:", json.dumps(result["answer"], indent=2))
        print("\nCITATIONS:", json.dumps(result["citations"], indent=2))
        print("=" * 80)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {args.output}")
    
    elif args.command == 'evaluate':
        stage2 = None if args.stage2_encoder == 'none' else args.stage2_encoder
        
        agent = HEAVENAgent(
            index_dir=args.index_dir,
            stage1_encoder_type=args.stage1_encoder,
            stage2_encoder_type=stage2,
            vlm_model=args.vlm_model,
            stage1_k=args.stage1_k,
            stage2_k=args.stage2_k,
            device=args.device,
            vs_page_filter_ratio=args.vs_page_filter_ratio,
            query_filter_ratio=args.query_filter_ratio,
            use_query_filtering=not args.no_query_filtering,
            alpha=args.alpha,
            beta=args.beta
        )
        
        run_evaluation(
            agent,
            output_file=args.output,
            dataset_name=args.dataset,
            split=args.split,
            limit=args.limit
        )
    
    else:
        parser.print_help()


def run_evaluation(
    agent: HEAVENAgent,
    output_file: str,
    dataset_name: str = "OxRML/MADQA",
    split: str = "test",
    limit: Optional[int] = None
):
    """Run evaluation on questions from HuggingFace dataset.
    
    Args:
        agent: The HEAVEN agent
        output_file: Path to output JSONL file
        dataset_name: HuggingFace dataset name
        split: Dataset split to evaluate on (dev, test, train)
        limit: Maximum number of questions to process
    """
    from datasets import load_dataset
    
    print(f"Loading {dataset_name} ({split} split)...")
    dataset = load_dataset(dataset_name, split=split)
    
    questions = list(dataset)
    if limit:
        questions = questions[:limit]
    
    print(f"Processing {len(questions)} questions...")
    
    # Process questions
    with open(output_file, 'w') as f:
        for q in tqdm(questions, desc="Answering"):
            question_text = q.get('question', '')
            question_id = q.get('id', '')
            
            result = agent.answer_question(question_text, question_id=question_id)
            
            # Format output compatible with eval/evaluate.py
            output = {
                'id': question_id,
                'question': question_text,
                'answer': result['answer'],
                'citations': result['citations'],
                'model': result.get('model', agent.vlm_model),
            }
            
            if 'error' in result:
                output['error'] = result['error']
            
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
