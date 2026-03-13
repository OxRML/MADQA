#!/usr/bin/env python3
"""
Search Agent Baseline for Document QA (Anthropic Version)

Provides image-only search results to Anthropic models using Whoosh search.
"""

import argparse
import json
import os
import time
from typing import List, Dict, Any
from PIL import Image

import anthropic

from utils import get_pdf_page_as_png, image_to_base64, WhooshSearchEngine


class AnthropicSearchAgent:
    """Anthropic agent with image-only search tool."""
    
    def __init__(self, search_engine: WhooshSearchEngine, model: str = "claude-sonnet-4-5-20250929", api_key: str = None):
        """Initialize the agent."""
        self.search_engine = search_engine
        self.model = model
        # Only pass api_key if explicitly provided, otherwise let SDK use ANTHROPIC_API_KEY env var
        if api_key is not None:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic()
        print(f"Initialized agent with model: {self.model}")
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a single page as an image in Anthropic vision format."""
        image = get_pdf_page_as_png(file, page)
        base64_image = image_to_base64(image)
        
        # Check if image exceeds 5MB limit and resize if needed
        # Anthropic's limit is on the base64 encoded size
        max_base64_size = 5 * 1024 * 1024  # 5MB in base64
        
        # Keep resizing until we're under the limit
        scale_factor = 1.0
        while len(base64_image) > max_base64_size:
            # Reduce by 10% each iteration (0.9^2 ≈ 0.81 area reduction)
            scale_factor *= 0.9
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            print(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height} (current size: {len(base64_image)/1024/1024:.1f}MB)")
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            base64_image = image_to_base64(resized_image)
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_image
            }
        }
    
    def _call_api_with_retry(self, max_retries: int = 3, **kwargs) -> Any:
        """Call Anthropic API with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(**kwargs)
                return response
            except anthropic.InternalServerError as e:
                error_str = str(e)
                print(f"API Error (500) on attempt {attempt + 1}/{max_retries}: {error_str}")
                
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 9 seconds
                    print(f"Server error detected. Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached.")
                    raise
            except anthropic.APIStatusError as e:
                # For other API errors (rate limits, etc.), also retry
                if e.status_code >= 500 or e.status_code == 429:
                    print(f"API Error ({e.status_code}) on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1
                        print(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    # For client errors (4xx except 429), don't retry
                    raise
        
        raise Exception(f"Failed after {max_retries} retries")
    
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
        search_tool = {
            "name": "search_documents",
            "description": "Search document collection and return images of matching pages. Supports: terms and phrases (use quotes for exact match), boolean operators (AND, OR, NOT - AND is default), wildcards (* for multiple chars, ? for single char). Examples: 'engine specifications', '\"Bell 407\" AND accessories', 'Bell*', 'incorporation NOT date'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query using keywords, phrases in quotes, and boolean operators"
                    }
                },
                "required": ["query"]
            }
        }
        
        # Define answer tool for structured output
        answer_tool = {
            "name": "provide_answer",
            "description": "Provide the final structured answer with citations after analyzing the document pages.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of answer values (one or more items)"
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "description": "The exact PDF filename (e.g., '1007969.pdf')"
                                },
                                "page": {
                                    "type": "integer",
                                    "description": "The page number"
                                }
                            },
                            "required": ["file", "page"]
                        },
                        "description": "List of citations with file and page information"
                    }
                },
                "required": ["answer", "citations"]
            }
        }
        
        system_prompt = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms. Be creative with queries - use synonyms, abbreviations, or different phrasings.

Once you find relevant pages, analyze the images carefully. When you have the answer, use the provide_answer tool with:
- answer: list of answer values (one or more items)
  * if there is a single answer, the output should be a one-element list
  * if the answer refers to multiple items or entities, the list will have several elements
  * do not write a full sentence there, use as few words as possible
  * if possible, use the exact words from the document
- citations: list of sources where EACH citation must have:
  * file: the exact PDF filename shown in the image (e.g., "1007969.pdf", "doc_name.pdf")
  * page: the page number (integer)"""

        # Additional instruction ensuring the model follows the exact tool schema
        system_prompt += """

Always call the search_documents tool with an input object that contains ONLY the key "query" with a string value. Do not include any other fields (no lists, alternate keys, or metadata)."""
        
        messages = [
            {"role": "user", "content": question}
        ]
        
        search_history = []
        trajectory = []  # Full trajectory log
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")
            
            # For the last iteration, force the model to provide an answer
            if iteration == max_iterations:
                tools = [answer_tool]
                try:
                    response = self._call_api_with_retry(
                        model=self.model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=messages,
                        tools=tools,
                        tool_choice={"type": "tool", "name": "provide_answer"}
                    )
                except Exception as e:
                    print(f"API Error on final iteration: {e}")
                    print(f"Number of messages: {len(messages)}")
                    if messages:
                        print(f"Last message role: {messages[-1]['role']}")
                        print(f"Last message content type: {type(messages[-1]['content'])}")
                    raise
            else:
                tools = [search_tool, answer_tool]
                try:
                    response = self._call_api_with_retry(
                        model=self.model,
                        max_tokens=4096,
                        system=system_prompt,
                        messages=messages,
                        tools=tools
                    )
                except Exception as e:
                    print(f"API Error on iteration {iteration}: {e}")
                    raise
            
            # Build trajectory step
            step = {
                "iteration": iteration,
                "role": "assistant",
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                } if hasattr(response, 'usage') and response.usage else None
            }
            
            # Extract text content and tool calls from response
            text_content = ""
            tool_calls = []
            thinking_content = None
            
            for block in response.content:
                if block.type == "text":
                    text_content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                elif block.type == "thinking":
                    # Claude's extended thinking (if enabled)
                    thinking_content = block.thinking if hasattr(block, 'thinking') else str(block)
            
            step["content"] = text_content if text_content else None
            step["tool_calls"] = tool_calls if tool_calls else None
            step["thinking"] = thinking_content
            
            # Check if we got a final answer or tool calls
            if response.stop_reason == "tool_use":
                # Collect all tool uses
                tool_uses = [block for block in response.content if block.type == "tool_use"]
                
                if not tool_uses:
                    print(f"Unexpected: tool_use stop reason but no tool_use blocks")
                    trajectory.append(step)
                    continue
                
                # Check if any tool use is provide_answer - but DON'T return yet
                # We need to check if there are ONLY provide_answer calls, no search_documents
                answer_tool_use = next((t for t in tool_uses if t.name == "provide_answer"), None)
                search_tool_uses = [t for t in tool_uses if t.name == "search_documents"]
                
                # If we have provide_answer AND no search_documents, we can return the answer
                if answer_tool_use and not search_tool_uses:
                    # Extract the structured answer with safe defaults
                    answer_data = answer_tool_use.input
                    # Handle malformed responses gracefully
                    answer = answer_data.get("answer", [])
                    citations = answer_data.get("citations", [])
                    # Ensure answer is a list
                    if isinstance(answer, str):
                        answer = [answer]
                    elif not isinstance(answer, list):
                        answer = [str(answer)] if answer else ["Unable to parse answer"]
                    trajectory.append(step)
                    return {
                        "question": question,
                        "answer": answer,
                        "citations": citations if isinstance(citations, list) else [],
                        "iterations": iteration,
                        "search_history": search_history,
                        "trajectory": trajectory,
                        "model": self.model
                    }
                
                # Handle search tool calls (and possibly ignore provide_answer if it came with search)
                # Must add assistant message first
                # Reconstruct content blocks to avoid extra fields from response
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # Build tool results - combine tool_result blocks and images into ONE user message
                # Anthropic API requires alternating user/assistant messages, so we can't split this
                # IMPORTANT: We must provide tool_result for EVERY tool_use, not just search_documents
                # CRITICAL: ALL tool_result blocks must come FIRST, then images/text after
                tool_result_blocks = []
                image_blocks = []
                tool_results_for_trajectory = []
                
                for tool_use in tool_uses:
                    if tool_use.name == "search_documents":
                        query = tool_use.input["query"]
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
                            "tool_use_id": tool_use.id,
                            "name": "search_documents",
                            "query": query,
                            "results": [
                                {"file": r["file"], "page": r["page_number"]}
                                for r in results
                            ]
                        })
                        
                        # Add tool_result block (will be placed first in user_content)
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Found {len(results)} matching pages."
                        })
                        
                        # If we have results, collect images (will be placed after all tool_results)
                        if len(results) > 0:
                            for result in results:
                                image_blocks.append({
                                    "type": "text",
                                    "text": f"\nFile: {result['file']}, Page: {result['page_number']}"
                                })
                                try:
                                    image_blocks.append(
                                        self._load_page_image(result['file'], result['page_number'])
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not load {result['file']} page {result['page_number']}: {e}")
                    elif tool_use.name == "provide_answer":
                        # Model tried to call provide_answer alongside search - tell it to wait
                        print(f"Note: Model called provide_answer alongside search, asking it to wait for results")
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": "Please wait for the search results before providing an answer.",
                            "is_error": True
                        })
                        tool_results_for_trajectory.append({
                            "tool_use_id": tool_use.id,
                            "name": "provide_answer",
                            "error": "Concurrent with search - asked to wait"
                        })
                    else:
                        # For any other unexpected tool, provide an error result
                        print(f"Warning: Unexpected tool call: {tool_use.name}")
                        tool_result_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Error: Tool '{tool_use.name}' is not available in this context.",
                            "is_error": True
                        })
                        tool_results_for_trajectory.append({
                            "tool_use_id": tool_use.id,
                            "name": tool_use.name,
                            "error": f"Tool not available"
                        })
                
                step["tool_results"] = tool_results_for_trajectory
                trajectory.append(step)
                
                # Combine: ALL tool_result blocks first, then images
                user_content = tool_result_blocks + image_blocks
                
                # Sanity check: We must have content if we added an assistant message with tool_use
                if not user_content:
                    raise RuntimeError(f"Failed to generate tool_results for {len(tool_uses)} tool_uses")
                
                messages.append({
                    "role": "user",
                    "content": user_content
                })
            else:
                # If we didn't get a tool use, something went wrong
                print(f"Unexpected stop reason: {response.stop_reason}")
                # Try to extract any text response
                text_content = ""
                for block in response.content:
                    if hasattr(block, 'text'):
                        text_content += block.text
                
                trajectory.append(step)
                return {
                    "question": question,
                    "answer": [text_content if text_content else "Unable to generate answer"],
                    "citations": [],
                    "iterations": iteration,
                    "search_history": search_history,
                    "trajectory": trajectory,
                    "model": self.model,
                    "error": f"Unexpected stop reason: {response.stop_reason}"
                }
        
        # Should not reach here, but handle gracefully
        return {
            "question": question,
            "answer": ["Maximum iterations reached without answer"],
            "citations": [],
            "iterations": max_iterations,
            "search_history": search_history,
            "trajectory": trajectory,
            "model": self.model,
            "error": "Maximum iterations reached"
        }


def main():
    parser = argparse.ArgumentParser(description="Search agent baseline for document QA (Anthropic)")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--ocr-file", default="data/ocr_output.jsonl",
                       help="Path to OCR results JSONL file")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929",
                       help="Anthropic model name (default: claude-sonnet-4-5-20250929)")
    parser.add_argument("--api-key", help="Anthropic API key (optional, uses ANTHROPIC_API_KEY env var)")
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
    agent = AnthropicSearchAgent(search_engine, args.model, args.api_key)
    
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
    if "error" in result:
        print(f"  Error: {result['error']}")
    print("="*80)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

