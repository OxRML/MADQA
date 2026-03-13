#!/usr/bin/env python3
"""
Process the MADQA dataset using SearchAgent baseline.

Supports loading from HuggingFace dataset with splits (dev, test, train).
Logs full trajectories of model outputs, reasoning traces, and metadata.
"""

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import traceback

from utils import WhooshSearchEngine
from openai_agent import SearchAgent
from anthropic_agent import AnthropicSearchAgent
from gemini_agent import GeminiSearchAgent
from vllm_agent import VLLMSearchAgent

try:
    from mistral_agent import MistralSearchAgent
except ImportError:
    MistralSearchAgent = None


def process_question(agent, item, max_iterations, top_k):
    """Process a single question and return the result with full trajectory."""
    question = item['question']
    question_id = item.get('id', '')
    
    try:
        result = agent.answer_question(
            question, 
            max_iterations=max_iterations,
            top_k=top_k
        )
        
        # Use ID from dataset if available, otherwise use the one from result
        result['id'] = question_id if question_id else result.get('id', '')
        
        # Add ground truth and metadata from dataset
        if 'answers' in item:
            result['ground_truth_answers'] = item['answers']
        if 'answer_locations' in item:
            result['ground_truth_locations'] = item['answer_locations']
        if 'category' in item:
            result['category'] = item['category']
        if 'documents' in item:
            result['documents'] = item['documents']
        
        return {'success': True, 'result': result}
        
    except Exception as e:
        error_tb = traceback.format_exc()
        return {
            'success': False,
            'result': {
                'question': question,
                'id': question_id,
                'error': str(e),
                'error_traceback': error_tb,
                'ground_truth_answers': item.get('answers'),
                'category': item.get('category')
            },
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Process dataset with SearchAgent")
    parser.add_argument("--output", "-o", required=True,
                       help="Output JSONL file")
    parser.add_argument("--limit", type=int,
                       help="Limit number of questions")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "vllm", "mistral"], default="openai",
                       help="Model provider to use (default: openai)")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07",
                       help="Model to use (default: gpt-5-mini-2025-08-07 for OpenAI, claude-sonnet-4-5-20250929 for Anthropic, gemini-2.0-flash-exp for Gemini, pixtral-large-latest for Mistral)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Max search iterations")
    parser.add_argument("--top-k", type=int, default=4,
                       help="Number of results per search")
    parser.add_argument("--dataset", default="OxRML/MADQA",
                       help="HuggingFace dataset name")
    parser.add_argument("--split", default="test",
                       help="Dataset split (dev, test, train)")
    parser.add_argument("--workers", type=int, default=5,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Change to this directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Load dataset
    print(f"Loading dataset: {args.dataset} ({args.split} split)")
    dataset = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(dataset)} questions from {args.split} split")
    
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {len(dataset)} questions")
    
    # Initialize SearchAgent
    print(f"\nInitializing {args.provider.capitalize()} SearchAgent...")
    ocr_path = script_dir / "data" / "ocr_output.jsonl"
    search_engine = WhooshSearchEngine(str(ocr_path))
    
    if args.provider == "anthropic":
        # Use default Anthropic model if OpenAI default was not changed
        model = args.model if args.model != "gpt-5-mini-2025-08-07" else "claude-sonnet-4-5-20250929"
        agent = AnthropicSearchAgent(search_engine, model=model)
    elif args.provider == "gemini":
        # Use default Gemini model if OpenAI default was not changed
        model = args.model if args.model != "gpt-5-mini-2025-08-07" else "gemini-2.0-flash-exp"
        agent = GeminiSearchAgent(search_engine, model=model)
    elif args.provider == "vllm":
        # Use default model name if OpenAI default was not changed
        model = args.model if args.model != "gpt-5-mini-2025-08-07" else "default"
        agent = VLLMSearchAgent(search_engine, model=model)
    elif args.provider == "mistral":
        if MistralSearchAgent is None:
            raise ImportError("mistral_agent module not found. Please add mistral_agent.py to use this provider.")
        model = args.model if args.model != "gpt-5-mini-2025-08-07" else "pixtral-large-latest"
        agent = MistralSearchAgent(search_engine, model=model)
    else:
        agent = SearchAgent(search_engine, model=args.model)
    
    # Check resume - use question ID for deduplication
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    # Use ID if available, fall back to question text
                    processed_ids.add(result.get('id') or result.get('question'))
        print(f"Skipping {len(processed_ids)} already processed")
    
    # Process
    print(f"\nProcessing questions with {args.workers} parallel workers...")
    mode = 'a' if args.resume else 'w'
    
    # File writing lock for thread safety
    write_lock = threading.Lock()
    
    # Prepare items to process
    items_to_process = []
    skipped = 0
    for item in dataset:
        item_id = item.get('id') or item.get('question')
        if item_id in processed_ids:
            skipped += 1
            continue
        items_to_process.append(item)
    
    if skipped > 0:
        print(f"Skipped {skipped} already processed questions")
    
    successful = failed = 0
    
    with open(args.output, mode, encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(
                    process_question, 
                    agent, 
                    item, 
                    args.max_iterations, 
                    args.top_k
                ): item
                for item in items_to_process
            }
            
            # Process results as they complete
            with tqdm(total=len(items_to_process), desc="Processing") as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    
                    try:
                        result_data = future.result()
                        
                        # Write immediately to file (thread-safe)
                        with write_lock:
                            f.write(json.dumps(result_data['result'], ensure_ascii=False) + '\n')
                            f.flush()
                        
                        if result_data['success']:
                            successful += 1
                        else:
                            failed += 1
                            print(f"\nError on '{item['question'][:50]}...': {result_data.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        print(f"\nUnexpected error processing '{item['question'][:50]}...': {e}")
                        # Write error result
                        with write_lock:
                            f.write(json.dumps({
                                'question': item['question'],
                                'id': item.get('id', ''),
                                'error': str(e),
                                'error_traceback': traceback.format_exc(),
                                'ground_truth_answers': item.get('answers'),
                                'category': item.get('category')
                            }, ensure_ascii=False) + '\n')
                            f.flush()
                        failed += 1
                    
                    pbar.update(1)
    
    print(f"\nDone: {successful} successful, {failed} failed, {skipped} skipped")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

