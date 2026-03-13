#!/usr/bin/env python3
"""
Gemini File Search Baseline for Document QA

Uses Google's File Search API for RAG-based document retrieval.
See: https://ai.google.dev/gemini-api/docs/file-search

This baseline:
1. Downloads externally-hosted PDFs referenced by the OxRML/MADQA dataset
2. Indexes them in a Gemini File Search store
3. Answers questions using the file search tool
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from datasets import load_dataset, DownloadManager
from tqdm import tqdm


class GeminiFileSearchAgent:
    """Agent using Gemini File Search for document QA."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        store_name: str = "agentic-doc-ai-store",
        api_key: Optional[str] = None
    ):
        """Initialize the agent.
        
        Args:
            model: Gemini model to use (must support File Search)
            store_name: Display name for the file search store
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided)
        """
        self.model = model
        self.store_name = store_name
        
        # Initialize client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
        
        self.file_search_store = None
        self.indexed_files = set()
        
        print(f"Initialized agent with model: {self.model}")
    
    def get_or_create_store(self) -> str:
        """Get existing store or create a new one.
        
        Returns:
            The file search store name/ID
        """
        # List existing stores
        try:
            stores = list(self.client.file_search_stores.list())
            for store in stores:
                if store.display_name == self.store_name:
                    print(f"Found existing store: {store.name}")
                    self.file_search_store = store
                    return store.name
        except Exception as e:
            print(f"Error listing stores: {e}")
        
        # Create new store
        print(f"Creating new file search store: {self.store_name}")
        self.file_search_store = self.client.file_search_stores.create(
            config={'display_name': self.store_name}
        )
        print(f"Created store: {self.file_search_store.name}")
        return self.file_search_store.name
    
    def list_indexed_files(self) -> set:
        """List files already indexed in the store."""
        if not self.file_search_store:
            return set()
        
        indexed = set()
        try:
            # List documents in the store
            docs = list(self.client.file_search_stores.list_documents(
                file_search_store_name=self.file_search_store.name
            ))
            for doc in docs:
                # Extract filename from display_name or name
                name = doc.display_name or doc.name
                indexed.add(name)
        except Exception as e:
            print(f"Error listing indexed files: {e}")
        
        return indexed
    
    def download_pdfs_from_hf(
        self,
        repo_id: str = "OxRML/MADQA",
        local_dir: str = "./pdf_cache",
        limit: Optional[int] = None
    ) -> List[Path]:
        """Download PDFs from HuggingFace dataset.
        
        Args:
            repo_id: HuggingFace repository ID
            local_dir: Local directory to cache PDFs
            limit: Maximum number of PDFs to download (None for all)
            
        Returns:
            List of local PDF file paths
        """
        import shutil

        local_dir = Path(local_dir)
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
    
    def index_pdfs(
        self,
        pdf_paths: List[Path],
        skip_existing: bool = True
    ) -> int:
        """Index PDFs in the file search store.
        
        Args:
            pdf_paths: List of local PDF file paths
            skip_existing: Skip files already indexed
            
        Returns:
            Number of files indexed
        """
        store_name = self.get_or_create_store()
        
        if skip_existing:
            self.indexed_files = self.list_indexed_files()
            print(f"Already indexed: {len(self.indexed_files)} files")
        
        indexed_count = 0
        pending_operations = []
        
        for pdf_path in tqdm(pdf_paths, desc="Uploading PDFs"):
            # Use just the filename for display
            display_name = pdf_path.name
            
            if skip_existing and display_name in self.indexed_files:
                continue
            
            try:
                # Upload directly to file search store
                operation = self.client.file_search_stores.upload_to_file_search_store(
                    file=str(pdf_path),
                    file_search_store_name=store_name,
                    config={'display_name': display_name}
                )
                pending_operations.append((display_name, operation))
                indexed_count += 1
                
                # Process in batches to avoid too many pending operations
                if len(pending_operations) >= 10:
                    self._wait_for_operations(pending_operations)
                    pending_operations = []
                    
            except Exception as e:
                print(f"Error uploading {pdf_path.name}: {e}")
        
        # Wait for remaining operations
        if pending_operations:
            self._wait_for_operations(pending_operations)
        
        print(f"Indexed {indexed_count} new files")
        return indexed_count
    
    def _wait_for_operations(self, operations: List[tuple], timeout: int = 300):
        """Wait for upload operations to complete."""
        for display_name, operation in operations:
            start_time = time.time()
            while not operation.done:
                if time.time() - start_time > timeout:
                    print(f"Timeout waiting for {display_name}")
                    break
                time.sleep(2)
                try:
                    operation = self.client.operations.get(operation)
                except Exception as e:
                    print(f"Error checking operation for {display_name}: {e}")
                    break
    
    def _parse_answer(self, text: str) -> Dict[str, Any]:
        """Parse the answer from model response.
        
        Args:
            text: Raw text response from model
            
        Returns:
            Dict with 'answer' and 'citations' keys
        """
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
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using file search.
        
        File Search handles retrieval automatically in a single API call.
        Unlike agentic approaches, there are no explicit iterations.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict with answer, citations, grounding_info, and metadata
        """
        if not self.file_search_store:
            self.get_or_create_store()
        
        print(f"\nQuestion: {question}")
        
        system_prompt = """You are a document QA assistant with access to a file search tool.
The file search tool retrieves relevant content from a collection of PDF documents.

IMPORTANT: The answer to the question is definitely in the documents. Search carefully using different terms if needed.

When you have the answer, respond with a JSON object in this exact format:
{
  "answer": ["answer value 1", "answer value 2", ...],
  "citations": [
    {"file": "exact_filename.pdf", "page": 1},
    {"file": "another_file.pdf", "page": 3}
  ]
}

Where:
- answer: list of answer values (use as few words as possible, exact document wording preferred)
- citations: list of sources with exact PDF filename and page number
"""
        
        grounding_info = {
            "retrieval_queries": [],
            "grounding_chunks": [],
            "num_chunks": 0
        }
        answer_data = {"answer": ["Unable to find answer"], "citations": []}
        
        # File Search is single-shot - one API call handles retrieval automatically
        prompt = f"{system_prompt}\n\nQuestion: {question}"
        
        try:
            # Generate with file search tool
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[
                        types.Tool(
                            file_search=types.FileSearch(
                                file_search_store_names=[self.file_search_store.name]
                            )
                        )
                    ]
                )
            )
            
            # Extract text response - handle different response formats
            text = ""
            try:
                if hasattr(response, 'text') and response.text:
                    text = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        for part in candidate.content.parts or []:
                            if hasattr(part, 'text') and part.text:
                                text += part.text
            except Exception as e:
                print(f"Error extracting text: {e}")
            
            print(f"Response text: {text[:300]}..." if len(text) > 300 else f"Response text: {text}")
            
            # Extract grounding metadata - this shows what File Search retrieved
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    # Check for grounding_metadata
                    if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                        metadata = candidate.grounding_metadata
                        
                        # Retrieval queries used
                        if hasattr(metadata, 'retrieval_queries') and metadata.retrieval_queries:
                            grounding_info["retrieval_queries"] = list(metadata.retrieval_queries)
                        
                        # Grounding chunks (retrieved content)
                        if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                            grounding_info["num_chunks"] = len(metadata.grounding_chunks)
                            for chunk in metadata.grounding_chunks[:5]:  # Log first 5
                                chunk_info = {}
                                if hasattr(chunk, 'retrieved_context'):
                                    ctx = chunk.retrieved_context
                                    if hasattr(ctx, 'uri'):
                                        chunk_info['uri'] = ctx.uri
                                    if hasattr(ctx, 'title'):
                                        chunk_info['title'] = ctx.title
                                grounding_info["grounding_chunks"].append(chunk_info)
                        
                        # Grounding supports (what parts are grounded)
                        if hasattr(metadata, 'grounding_supports') and metadata.grounding_supports:
                            print(f"  Grounding supports: {len(metadata.grounding_supports)}")
                    else:
                        print("No grounding_metadata in response")
                        # Print available attributes for debugging
                        print(f"  Candidate attributes: {[a for a in dir(candidate) if not a.startswith('_')]}")
            except Exception as e:
                print(f"Error extracting grounding metadata: {e}")
                import traceback
                traceback.print_exc()
            
            # Try to parse as final answer
            answer_data = self._parse_answer(text)
            
            return {
                "question": question,
                "answer": answer_data["answer"],
                "citations": answer_data["citations"],
                "iterations": 1,  # File Search is single-shot
                "grounding_info": grounding_info,
                "model": self.model
            }
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "question": question,
                "answer": ["Error occurred"],
                "citations": [],
                "iterations": 1,
                "grounding_info": grounding_info,
                "model": self.model,
                "error": str(e)
            }


def run_evaluation(
    agent: GeminiFileSearchAgent,
    output_file: str,
    dataset_name: str = "OxRML/MADQA",
    split: str = "test",
    limit: Optional[int] = None
):
    """Run evaluation on questions from HuggingFace dataset.
    
    Args:
        agent: The file search agent
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
            
            result = agent.answer_question(question_text)
            
            # Format output compatible with eval/evaluate.py
            output = {
                'id': question_id,
                'question': question_text,
                'answer': result['answer'],
                'citations': result['citations'],
                'iterations': result.get('iterations', 1),
                'model': result.get('model', agent.model),
            }
            
            # Include grounding info for analysis
            if 'grounding_info' in result:
                output['grounding_info'] = result['grounding_info']
            
            if 'error' in result:
                output['error'] = result['error']
            
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini File Search baseline for document QA"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index PDFs from HuggingFace')
    index_parser.add_argument('--repo', default='OxRML/MADQA',
                              help='HuggingFace repository ID')
    index_parser.add_argument('--limit', type=int, help='Limit number of PDFs')
    index_parser.add_argument('--cache-dir', default='./pdf_cache',
                              help='Local cache directory for PDFs')
    index_parser.add_argument('--store-name', default='agentic-doc-ai-store',
                              help='File search store name')
    index_parser.add_argument('--api-key', help='Google API key')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('question', help='Question to answer')
    ask_parser.add_argument('--model', default='gemini-2.5-flash',
                            help='Gemini model name')
    ask_parser.add_argument('--store-name', default='agentic-doc-ai-store',
                            help='File search store name')
    ask_parser.add_argument('--api-key', help='Google API key')
    ask_parser.add_argument('--output', '-o', help='Output JSON file')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation on questions')
    eval_parser.add_argument('output', help='Output JSONL file')
    eval_parser.add_argument('--dataset', default='OxRML/MADQA',
                             help='HuggingFace dataset name')
    eval_parser.add_argument('--split', default='test',
                             help='Dataset split (dev, test, train)')
    eval_parser.add_argument('--model', default='gemini-2.5-flash',
                             help='Gemini model name')
    eval_parser.add_argument('--store-name', default='agentic-doc-ai-store',
                             help='File search store name')
    eval_parser.add_argument('--limit', type=int, help='Limit number of questions')
    eval_parser.add_argument('--api-key', help='Google API key')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        agent = GeminiFileSearchAgent(
            store_name=args.store_name,
            api_key=args.api_key
        )
        
        # Download PDFs
        pdf_paths = agent.download_pdfs_from_hf(
            repo_id=args.repo,
            local_dir=args.cache_dir,
            limit=args.limit
        )
        
        # Index them
        agent.index_pdfs(pdf_paths)
        
    elif args.command == 'ask':
        agent = GeminiFileSearchAgent(
            model=args.model,
            store_name=args.store_name,
            api_key=args.api_key
        )
        
        result = agent.answer_question(args.question)
        
        # Print result
        print("\n" + "=" * 80)
        print("QUESTION:", result["question"])
        print("\nANSWER:", json.dumps(result["answer"], indent=2))
        print("\nCITATIONS:", json.dumps(result["citations"], indent=2))
        print("\nMETADATA:")
        print(f"  Model: {result['model']}")
        print(f"  Iterations: {result['iterations']}")
        if result.get("search_history"):
            print("  Searches:")
            for search in result["search_history"]:
                print(f"    [{search['iteration']}] '{search['query']}'")
        if "error" in result:
            print(f"  Error: {result['error']}")
        print("=" * 80)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {args.output}")
            
    elif args.command == 'evaluate':
        agent = GeminiFileSearchAgent(
            model=args.model,
            store_name=args.store_name,
            api_key=args.api_key
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


if __name__ == "__main__":
    main()

