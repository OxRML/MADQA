#!/usr/bin/env python3
"""
OpenAI File Search Baseline for Document QA

Uses OpenAI's Assistants API with File Search tool for RAG-based document retrieval.
See: https://platform.openai.com/docs/guides/tools-file-search

This baseline:
1. Downloads externally-hosted PDFs referenced by the OxRML/MADQA dataset
2. Uploads them to an OpenAI vector store
3. Answers questions using an Assistant with file_search tool
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from datasets import load_dataset, DownloadManager
from tqdm import tqdm


class OpenAIFileSearchAgent:
    """Agent using OpenAI Assistants File Search for document QA."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        vector_store_name: str = "agentic-doc-ai-store-new",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """Initialize the agent.
        
        Args:
            model: OpenAI model to use
            vector_store_name: Name for the vector store
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            base_url: Custom base URL (default: api.openai.com)
        
        Note: This baseline requires direct OpenAI API access. Vector stores and
        Assistants are server-side resources that don't work through proxy services.
        API keys should start with 'sk-'.
        """
        self.model = model
        self.vector_store_name = vector_store_name
        
        # Check for potential proxy usage
        actual_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if actual_key and not actual_key.startswith("sk-"):
            print("WARNING: API key doesn't start with 'sk-'. You may be using a proxy service.")
            print("         OpenAI File Search requires direct OpenAI API access.")
            print("         Vector stores won't work through proxy services.")
        
        # Initialize client
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs) if client_kwargs else OpenAI()
        
        self.vector_store = None
        self.assistant = None
        self.indexed_files = set()
        
        print(f"Initialized agent with model: {self.model}")
        if base_url:
            print(f"Using custom base URL: {base_url}")
    
    def get_or_create_vector_store(self) -> str:
        """Get existing vector store or create a new one.
        
        Returns:
            The vector store ID
            
        Raises:
            RuntimeError: If vector stores API is not available (e.g., using proxy)
        """
        # Try both client.vector_stores (newer) and client.beta.vector_stores (older)
        vector_stores_api = getattr(self.client, 'vector_stores', None) or \
                           getattr(self.client.beta, 'vector_stores', None)
        
        if vector_stores_api is None:
            raise RuntimeError(
                "Vector stores API not available.\n"
                "This likely means you're using a proxy service instead of direct OpenAI API.\n"
                "OpenAI File Search requires direct access to api.openai.com with an 'sk-' API key.\n"
                "Vector stores are server-side resources that can't be replicated by proxies."
            )
        
        # List existing vector stores
        try:
            stores = vector_stores_api.list()
            for store in stores.data:
                if store.name == self.vector_store_name:
                    print(f"Found existing vector store: {store.id}")
                    self.vector_store = store
                    self._vector_stores_api = vector_stores_api
                    return store.id
        except Exception as e:
            print(f"Error listing vector stores: {e}")
        
        # Create new vector store
        print(f"Creating new vector store: {self.vector_store_name}")
        self.vector_store = vector_stores_api.create(
            name=self.vector_store_name
        )
        self._vector_stores_api = vector_stores_api
        print(f"Created vector store: {self.vector_store.id}")
        return self.vector_store.id
    
    def get_or_create_assistant(self) -> str:
        """Get existing assistant or create a new one.
        
        Returns:
            The assistant ID
        """
        assistant_name = f"DocQA-{self.vector_store_name}"
        
        # List existing assistants
        try:
            assistants = self.client.beta.assistants.list()
            for asst in assistants.data:
                if asst.name == assistant_name:
                    print(f"Found existing assistant: {asst.id}")
                    self.assistant = asst
                    # Update with current vector store
                    if self.vector_store:
                        self.client.beta.assistants.update(
                            assistant_id=asst.id,
                            tool_resources={
                                "file_search": {
                                    "vector_store_ids": [self.vector_store.id]
                                }
                            }
                        )
                    return asst.id
        except Exception as e:
            print(f"Error listing assistants: {e}")
        
        # Create new assistant
        print(f"Creating new assistant: {assistant_name}")
        
        tool_resources = {}
        if self.vector_store:
            tool_resources = {
                "file_search": {
                    "vector_store_ids": [self.vector_store.id]
                }
            }
        
        self.assistant = self.client.beta.assistants.create(
            name=assistant_name,
            instructions="""You are a document QA assistant with access to a file search tool.
The file search tool retrieves relevant content from a collection of PDF documents.

IMPORTANT: The answer to the question is definitely in the documents. Search carefully.

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
- citations: list of sources with exact PDF filename and page number""",
            model=self.model,
            tools=[{"type": "file_search"}],
            tool_resources=tool_resources
        )
        print(f"Created assistant: {self.assistant.id}")
        return self.assistant.id
    
    def list_indexed_files(self) -> set:
        """List files already indexed in the vector store."""
        if not self.vector_store:
            return set()
        
        indexed = set()
        try:
            # Use the cached API reference or find it
            vector_stores_api = getattr(self, '_vector_stores_api', None) or \
                               getattr(self.client, 'vector_stores', None) or \
                               getattr(self.client.beta, 'vector_stores', None)
            
            if vector_stores_api:
                files = vector_stores_api.files.list(
                    vector_store_id=self.vector_store.id
                )
                for f in files.data:
                    # Get the original filename
                    try:
                        file_obj = self.client.files.retrieve(f.id)
                        indexed.add(file_obj.filename)
                    except:
                        pass
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
        skip_existing: bool = True,
        batch_size: int = 100
    ) -> int:
        """Index PDFs in the vector store.
        
        Args:
            pdf_paths: List of local PDF file paths
            skip_existing: Skip files already indexed
            batch_size: Number of files to upload per batch
            
        Returns:
            Number of files indexed
        """
        vector_store_id = self.get_or_create_vector_store()
        
        if skip_existing:
            self.indexed_files = self.list_indexed_files()
            print(f"Already indexed: {len(self.indexed_files)} files")
        
        # Filter files to upload
        files_to_upload = []
        for pdf_path in pdf_paths:
            display_name = pdf_path.name
            if skip_existing and display_name in self.indexed_files:
                continue
            files_to_upload.append(pdf_path)
        
        if not files_to_upload:
            print("No new files to index")
            return 0
        
        print(f"Uploading {len(files_to_upload)} files...")
        
        indexed_count = 0
        
        # Process in batches
        for i in range(0, len(files_to_upload), batch_size):
            batch = files_to_upload[i:i + batch_size]
            print(f"\nBatch {i // batch_size + 1}: {len(batch)} files")
            
            # Open file streams
            file_streams = []
            for pdf_path in batch:
                try:
                    file_streams.append(open(pdf_path, "rb"))
                except Exception as e:
                    print(f"Error opening {pdf_path}: {e}")
            
            if not file_streams:
                continue
            
            try:
                # Upload batch to vector store
                vector_stores_api = getattr(self, '_vector_stores_api', None) or \
                                   getattr(self.client, 'vector_stores', None) or \
                                   getattr(self.client.beta, 'vector_stores', None)
                
                file_batch = vector_stores_api.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=file_streams
                )
                
                print(f"  Status: {file_batch.status}")
                print(f"  File counts: {file_batch.file_counts}")
                indexed_count += file_batch.file_counts.completed
                
            except Exception as e:
                print(f"Error uploading batch: {e}")
            finally:
                # Close file streams
                for fs in file_streams:
                    try:
                        fs.close()
                    except:
                        pass
        
        # Create/update assistant with this vector store
        self.get_or_create_assistant()
        
        print(f"\nIndexed {indexed_count} new files")
        return indexed_count
    
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
    
    def _extract_citations_from_annotations(self, message) -> List[Dict[str, Any]]:
        """Extract citations from message annotations.
        
        Args:
            message: OpenAI message object
            
        Returns:
            List of citation dicts with file and page info
        """
        citations = []
        
        try:
            for content_block in message.content:
                if content_block.type != "text":
                    continue
                    
                text_content = content_block.text
                if not hasattr(text_content, 'annotations'):
                    continue
                
                for annotation in text_content.annotations:
                    if annotation.type == "file_citation":
                        file_citation = annotation.file_citation
                        try:
                            # Get the file object to retrieve filename
                            file_obj = self.client.files.retrieve(file_citation.file_id)
                            citation = {
                                "file": file_obj.filename,
                                "file_id": file_citation.file_id
                            }
                            # Note: OpenAI doesn't provide page numbers directly
                            # We might be able to extract from quote if available
                            if hasattr(file_citation, 'quote'):
                                citation["quote"] = file_citation.quote[:200]
                            citations.append(citation)
                        except Exception as e:
                            print(f"Error retrieving file info: {e}")
                            citations.append({
                                "file_id": file_citation.file_id
                            })
        except Exception as e:
            print(f"Error extracting annotations: {e}")
        
        return citations
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using file search.
        
        Uses OpenAI Assistants API which handles retrieval automatically.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict with answer, citations, and metadata
        """
        if not self.vector_store:
            self.get_or_create_vector_store()
        
        if not self.assistant:
            self.get_or_create_assistant()
        
        print(f"\nQuestion: {question}")
        
        retrieval_info = {
            "annotations": [],
            "num_citations": 0
        }
        
        try:
            # Create a thread
            thread = self.client.beta.threads.create()
            
            # Add the question as a message
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=question
            )
            
            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant.id
            )
            
            # Poll for completion
            start_time = time.time()
            timeout = 120  # 2 minutes timeout
            
            while run.status in ["queued", "in_progress"]:
                if time.time() - start_time > timeout:
                    print("Timeout waiting for response")
                    break
                time.sleep(1)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
            
            if run.status != "completed":
                print(f"Run ended with status: {run.status}")
                if hasattr(run, 'last_error') and run.last_error:
                    print(f"Error: {run.last_error}")
                return {
                    "question": question,
                    "answer": ["Error: Run did not complete"],
                    "citations": [],
                    "iterations": 1,
                    "model": self.model,
                    "error": f"Run status: {run.status}"
                }
            
            # Get the assistant's response
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc",
                limit=1
            )
            
            if not messages.data:
                return {
                    "question": question,
                    "answer": ["No response received"],
                    "citations": [],
                    "iterations": 1,
                    "model": self.model
                }
            
            assistant_message = messages.data[0]
            
            # Extract text response
            text = ""
            for content_block in assistant_message.content:
                if content_block.type == "text":
                    text += content_block.text.value
            
            print(f"Response: {text[:300]}..." if len(text) > 300 else f"Response: {text}")
            
            # Extract citations from annotations
            api_citations = self._extract_citations_from_annotations(assistant_message)
            retrieval_info["annotations"] = api_citations
            retrieval_info["num_citations"] = len(api_citations)
            
            # Parse the answer
            answer_data = self._parse_answer(text)
            
            # Merge API citations with parsed citations
            # Prefer parsed citations if they have page numbers
            final_citations = answer_data["citations"] if answer_data["citations"] else api_citations
            
            # Clean up thread
            try:
                self.client.beta.threads.delete(thread.id)
            except:
                pass
            
            return {
                "question": question,
                "answer": answer_data["answer"],
                "citations": final_citations,
                "iterations": 1,  # Assistants API is single-shot
                "retrieval_info": retrieval_info,
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
                "model": self.model,
                "error": str(e)
            }


def run_evaluation(
    agent: OpenAIFileSearchAgent,
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
            
            # Include retrieval info for analysis
            if 'retrieval_info' in result:
                output['retrieval_info'] = result['retrieval_info']
            
            if 'error' in result:
                output['error'] = result['error']
            
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI File Search baseline for document QA"
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
                              help='Vector store name')
    index_parser.add_argument('--api-key', help='OpenAI API key (must start with sk-)')
    index_parser.add_argument('--base-url', help='Custom API base URL')
    index_parser.add_argument('--batch-size', type=int, default=100,
                              help='Files per upload batch')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a single question')
    ask_parser.add_argument('question', help='Question to answer')
    ask_parser.add_argument('--model', default='gpt-4o',
                            help='OpenAI model name')
    ask_parser.add_argument('--store-name', default='agentic-doc-ai-store',
                            help='Vector store name')
    ask_parser.add_argument('--api-key', help='OpenAI API key (must start with sk-)')
    ask_parser.add_argument('--base-url', help='Custom API base URL')
    ask_parser.add_argument('--output', '-o', help='Output JSON file')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation on questions')
    eval_parser.add_argument('output', help='Output JSONL file')
    eval_parser.add_argument('--dataset', default='OxRML/MADQA',
                             help='HuggingFace dataset name')
    eval_parser.add_argument('--split', default='test',
                             help='Dataset split (dev, test, train)')
    eval_parser.add_argument('--model', default='gpt-4o',
                             help='OpenAI model name')
    eval_parser.add_argument('--store-name', default='agentic-doc-ai-store',
                             help='Vector store name')
    eval_parser.add_argument('--limit', type=int, help='Limit number of questions')
    eval_parser.add_argument('--api-key', help='OpenAI API key (must start with sk-)')
    eval_parser.add_argument('--base-url', help='Custom API base URL')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        agent = OpenAIFileSearchAgent(
            vector_store_name=args.store_name,
            api_key=args.api_key,
            base_url=getattr(args, 'base_url', None)
        )
        
        # Download PDFs
        pdf_paths = agent.download_pdfs_from_hf(
            repo_id=args.repo,
            local_dir=args.cache_dir,
            limit=args.limit
        )
        
        # Index them
        agent.index_pdfs(pdf_paths, batch_size=args.batch_size)
        
    elif args.command == 'ask':
        agent = OpenAIFileSearchAgent(
            model=args.model,
            vector_store_name=args.store_name,
            api_key=args.api_key,
            base_url=getattr(args, 'base_url', None)
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
        if "error" in result:
            print(f"  Error: {result['error']}")
        print("=" * 80)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {args.output}")
            
    elif args.command == 'evaluate':
        agent = OpenAIFileSearchAgent(
            model=args.model,
            vector_store_name=args.store_name,
            api_key=args.api_key,
            base_url=getattr(args, 'base_url', None)
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

