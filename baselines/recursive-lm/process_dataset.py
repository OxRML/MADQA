#!/usr/bin/env python3
"""
Process the MADQA dataset using RLM baseline.

Supports loading from HuggingFace dataset with splits (dev, test, train).
Logs full trajectories of model outputs and metadata.
"""

import argparse
import json
import os
import signal
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import traceback

from rlm_agent import RLMAgent

# Timeout settings
TIMEOUT_SECONDS = 5000  # ~83 minutes
MAX_RETRIES = 2  # Retry up to 2 times (3 total attempts)


class TimeoutException(Exception):
    """Raised when a function call times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for SIGALRM."""
    raise TimeoutException(f"Operation timed out after {TIMEOUT_SECONDS} seconds")


def main():
    parser = argparse.ArgumentParser(description="Process dataset with RLM Agent")
    parser.add_argument("--output", "-o", required=True,
                       help="Output JSONL file")
    parser.add_argument("--limit", type=int,
                       help="Limit number of questions")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output")
    parser.add_argument("--backend", default="openai",
                       help="RLM backend (default: openai)")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07",
                       help="Model to use")
    parser.add_argument("--dataset", default="OxRML/MADQA",
                       help="HuggingFace dataset name for questions")
    parser.add_argument("--corpus-dataset", default="OxRML/pdfs-to-markdown-mistral-ocr",
                       help="HuggingFace dataset name for document corpus (markdown)")
    parser.add_argument("--split", default="test",
                       help="Dataset split (dev, test, train)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
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
    
    # Initialize RLM Agent
    print(f"\nInitializing RLM Agent with {args.backend}/{args.model}...")
    print(f"Using corpus from HuggingFace: {args.corpus_dataset}")
    
    agent = RLMAgent(
        hf_dataset=args.corpus_dataset,
        backend=args.backend,
        model_name=args.model,
        verbose=args.verbose
    )
    
    # Check resume
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        print(f"Resuming from {args.output}")
        with open(args.output, 'r') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    processed_ids.add(result.get('id') or result.get('question'))
        print(f"Skipping {len(processed_ids)} already processed")
    
    # Process questions sequentially
    print(f"\nProcessing {len(dataset)} questions...")
    mode = 'a' if args.resume else 'w'
    successful = failed = skipped = 0
    
    with open(args.output, mode, encoding='utf-8') as f:
        for item in tqdm(dataset, desc="Processing"):
            question = item['question']
            question_id = item.get('id', '')
            
            # Skip if already processed
            item_key = question_id or question
            if item_key in processed_ids:
                skipped += 1
                continue
            
            result = None
            
            # Retry loop with timeout using SIGALRM (hard timeout on Linux)
            for attempt in range(MAX_RETRIES + 1):
                try:
                    # Set up signal-based timeout (works on Unix/Linux)
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TIMEOUT_SECONDS)
                    
                    try:
                        result = agent.answer_question(question)
                    finally:
                        signal.alarm(0)  # Cancel the alarm
                    
                    result['id'] = question_id
                    
                    # Add ground truth metadata
                    if 'answers' in item:
                        result['ground_truth_answers'] = item['answers']
                    if 'answer_locations' in item:
                        result['ground_truth_locations'] = item['answer_locations']
                    if 'category' in item:
                        result['category'] = item['category']
                    if 'documents' in item:
                        result['documents'] = item['documents']
                    
                    successful += 1
                    break  # Success, exit retry loop
                    
                except TimeoutException:
                    signal.alarm(0)  # Ensure alarm is cancelled
                    tqdm.write(f"Timeout on '{question[:50]}...' (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                    if attempt < MAX_RETRIES:
                        continue  # Retry
                    # Max retries reached, mark as timeout
                    result = {
                        'question': question,
                        'id': question_id,
                        'error': 'timeout',
                        'error_details': f"Timed out after {MAX_RETRIES + 1} attempts ({TIMEOUT_SECONDS}s each)",
                        'ground_truth_answers': item.get('answers'),
                        'category': item.get('category')
                    }
                    failed += 1
                    
                except Exception as e:
                    signal.alarm(0)  # Ensure alarm is cancelled
                    tqdm.write(f"Error on '{question[:50]}...' (attempt {attempt + 1}): {e}")
                    if attempt < MAX_RETRIES:
                        continue  # Retry
                    # Max retries reached
                    result = {
                        'question': question,
                        'id': question_id,
                        'error': str(e),
                        'error_traceback': traceback.format_exc(),
                        'ground_truth_answers': item.get('answers'),
                        'category': item.get('category')
                    }
                    failed += 1
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
    
    print(f"\nDone: {successful} successful, {failed} failed, {skipped} skipped")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
