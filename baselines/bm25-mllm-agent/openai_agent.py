#!/usr/bin/env python3
"""
Search Agent Baseline for Document QA

Provides image-only search results to OpenAI models using Whoosh search.
"""

import argparse
import json
import os
from typing import List, Dict, Any

import openai

from utils import get_pdf_page_as_png, image_to_base64, WhooshSearchEngine, Citation, Answer


class SearchAgent:
    """OpenAI agent with image-only search tool."""
    
    def __init__(self, search_engine: WhooshSearchEngine, model: str = "gpt-5-mini-2025-08-07", api_key: str = None):
        """Initialize the agent."""
        self.search_engine = search_engine
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)
        print(f"Initialized agent with model: {self.model}")
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a single page as an image in OpenAI vision format."""
        image = get_pdf_page_as_png(file, page)
        base64_image = image_to_base64(image)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def answer_question(self, question: str, max_iterations: int = 5, top_k: int = 3) -> Dict[str, Any]:
        """Answer a question using the search tool.
        
        Returns a dict with:
            - question: The original question
            - answer: List of answer values
            - citations: List of citation dicts
            - iterations: Number of iterations used
            - search_history: Summary of searches (for backward compatibility)
            - trajectory: Full trajectory log with model outputs and reasoning
            - model: Model name used
        """
        print(f"\nQuestion: {question}")
        
        # Define search tool
        tools = [{
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Search document collection and return images of matching pages. Supports: terms and phrases (use quotes for exact match), boolean operators (AND, OR, NOT - AND is default), wildcards (* for multiple chars, ? for single char). Examples: 'engine specifications', '\"Bell 407\" AND accessories', 'Bell*', 'incorporation NOT date'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query using keywords, phrases in quotes, and boolean operators"
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]
        
        system_prompt = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms. Be creative with queries - use synonyms, abbreviations, or different phrasings.

Once you find relevant pages, analyze the images carefully and provide:
- answer: list of answer values (one or more items)
  * if there is a single answer, the output should be a one-element list
  * if the answer refers to multiple items or entities, the list will have several elements
  * do not write a full sentence there, use as few words as possible
  * if possible, use the exact words from the document
- citations: list of sources where EACH citation must have:
  * file: the exact PDF filename shown in the image (e.g., "1007969.pdf", "doc_name.pdf")
  * page: the page number (integer)"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        search_history = []
        trajectory = []  # Full trajectory log
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")
            
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="none" if iteration == max_iterations else "auto",
                response_format=Answer
            )
            message = response.choices[0].message
            
            # Build trajectory step
            step = {
                "iteration": iteration,
                "role": "assistant",
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') and response.usage else None
            }
            
            # Extract content and tool calls
            step["content"] = message.content if message.content else None
            
            if message.tool_calls:
                step["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
                    }
                    for tc in message.tool_calls
                ]
            
            # Check for reasoning/thinking if available (o1 models)
            if hasattr(message, 'reasoning') and message.reasoning:
                step["thinking"] = message.reasoning
            
            if not message.tool_calls:
                step["parsed_answer"] = {
                    "answer": message.parsed.answer,
                    "citations": [{"file": c.file, "page": c.page} for c in message.parsed.citations]
                } if message.parsed else None
                trajectory.append(step)
                
                return {
                    "question": question,
                    "answer": message.parsed.answer,
                    "citations": [{"file": c.file, "page": c.page} for c in message.parsed.citations],
                    "iterations": iteration,
                    "search_history": search_history,
                    "trajectory": trajectory,
                    "model": self.model
                }
            
            messages.append(message)
            image_content = [{"type": "text", "text": "Here are the matching pages:\n"}]
            tool_results_for_trajectory = []
            
            for tool_call in message.tool_calls:
                if tool_call.function.name == "search_documents":
                    query = json.loads(tool_call.function.arguments)["query"]
                    print(f"Searching: {query}")
                    
                    results = self.search_engine.search(query, top_k)
                    print(f"Returning {len(results)} images to model")
                    
                    search_history.append({
                        "iteration": iteration,
                        "query": query,
                        "num_results": len(results)
                    })
                    
                    # Log search results in trajectory
                    tool_results_for_trajectory.append({
                        "tool_call_id": tool_call.id,
                        "name": "search_documents",
                        "query": query,
                        "results": [
                            {"file": r["file"], "page": r["page_number"]}
                            for r in results
                        ]
                    })
                    
                    # Add tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Found {len(results)} matching pages."
                    })
                    
                    # Build interleaved content: metadata text + image
                    for result in results:
                        image_content.append({
                            "type": "text",
                            "text": f"\n[Image] File: {result['file']}, Page: {result['page_number']}"
                        })
                        try:
                            image_content.append(self._load_page_image(result['file'], result['page_number']))
                        except Exception as e:
                            print(f"Warning: Could not load {result['file']} page {result['page_number']}: {e}")
            
            step["tool_results"] = tool_results_for_trajectory
            trajectory.append(step)
            
            if len(image_content) > 1:
                messages.append({"role": "user", "content": image_content})


def main():
    parser = argparse.ArgumentParser(description="Search agent baseline for document QA")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--ocr-file", default="data/ocr_output.jsonl",
                       help="Path to OCR results JSONL file")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07",
                       help="OpenAI model name (default: gpt-5-mini-2025-08-07)")
    parser.add_argument("--api-key", help="OpenAI API key (optional, uses OPENAI_API_KEY env var)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Maximum search iterations (default: 5)")
    parser.add_argument("--top-k", type=int, default=3,
                       help="Number of results per search (default: 3)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Initialize search engine
    ocr_path = os.path.join(os.path.dirname(__file__), args.ocr_file)
    search_engine = WhooshSearchEngine(ocr_path)
    
    # Initialize agent
    agent = SearchAgent(search_engine, args.model, args.api_key)
    
    # Answer question
    result = agent.answer_question(args.question, args.max_iterations, args.top_k)
    
    # Print result
    print("\n" + "="*80)
    print("QUESTION:", result["question"])
    print("\nANSWER:", json.dumps(result["answer"], indent=2))
    print("\nCITATIONS:", json.dumps(result["citations"], indent=2))
    print("\nMETADATA:")
    print(f"  Model: {result['model']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Searches:")
    for search in result["search_history"]:
        print(f"    [{search['iteration']}] '{search['query']}' -> {search['num_results']} images")
    print("="*80)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
