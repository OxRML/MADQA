#!/usr/bin/env python3
"""
Search Agent Baseline for Document QA (vLLM Version)

Provides image-only search results to vLLM models using Whoosh search.
Uses OpenAI-compatible API that vLLM provides.
"""

import argparse
import json
import os
import re
from typing import List, Dict, Any
from PIL import Image

import openai

from utils import get_pdf_page_as_png, image_to_base64, WhooshSearchEngine


class VLLMSearchAgent:
    """vLLM agent with image-only search tool using OpenAI-compatible API."""
    
    def __init__(
        self, 
        search_engine: WhooshSearchEngine, 
        model: str = "default",
        base_url: str = None,
        api_key: str = None
    ):
        """Initialize the agent.
        
        Args:
            search_engine: WhooshSearchEngine instance
            model: Model name running on vLLM server
            base_url: vLLM server URL (default: VLLM_BASE_URL env or http://localhost:8000/v1)
            api_key: API key (default: VLLM_API_KEY env or "EMPTY")
        """
        self.search_engine = search_engine
        self.model = model
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = api_key or os.environ.get("VLLM_API_KEY", "abc123")
        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)
        print(f"Initialized vLLM agent with model: {self.model} at {self.base_url}")
    
    def _load_page_image(self, file: str, page: int) -> Dict:
        """Load a single page as an image in OpenAI vision format."""
        image = get_pdf_page_as_png(file, page)
        base64_image = image_to_base64(image)
        
        # Check if image is too large and resize if needed
        # Many models have size limits, be conservative
        max_base64_size = 4 * 1024 * 1024  # 4MB
        
        scale_factor = 1.0
        while len(base64_image) > max_base64_size:
            scale_factor *= 0.9
            new_width = int(image.width * scale_factor)
            new_height = int(image.height * scale_factor)
            
            print(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height}")
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            base64_image = image_to_base64(resized_image)
        
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def _parse_answer_from_text(self, text: str) -> Dict[str, Any]:
        """Try to parse structured answer from text response.
        
        Falls back to returning the raw text if parsing fails.
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*"answer"[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                answer = data.get("answer", [])
                if isinstance(answer, str):
                    answer = [answer]
                citations = data.get("citations", [])
                return {"answer": answer, "citations": citations}
            except json.JSONDecodeError:
                pass
        
        # Look for answer patterns
        answer_match = re.search(r'(?:answer|Answer|ANSWER)[:\s]+([^\n]+)', text)
        if answer_match:
            return {"answer": [answer_match.group(1).strip()], "citations": []}
        
        # Return raw text as answer
        return {"answer": [text.strip()], "citations": []}
    
    def answer_question(
        self, 
        question: str, 
        max_iterations: int = 5, 
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Answer a question using the search tool.
        
        Args:
            question: Question to answer
            max_iterations: Maximum search iterations
            top_k: Number of results per search
            
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
        
        # Define tools - search and answer
        tools = [
            {
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
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "provide_answer",
                    "description": "Provide the final structured answer with citations after analyzing the document pages.",
                    "parameters": {
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
            }
        ]
        
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
  * page: the page number (integer)

Always use one of the available tools (search_documents or provide_answer)."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        search_history = []
        trajectory = []  # Full trajectory log
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")
            
            try:
                # On last iteration, try to force an answer
                if iteration == max_iterations:
                    # Add a prompt to force answer
                    force_message = {
                        "role": "user", 
                        "content": "You must now provide your final answer using the provide_answer tool based on what you've found."
                    }
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages + [force_message],
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": "provide_answer"}}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
            except Exception as e:
                print(f"API Error on iteration {iteration}: {e}")
                # Try without tool_choice if it fails
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools
                    )
                except Exception as e2:
                    print(f"Retry also failed: {e2}")
                    raise
            
            message = response.choices[0].message
            
            # Build trajectory step
            step = {
                "iteration": iteration,
                "role": "assistant",
                "finish_reason": response.choices[0].finish_reason,
                "content": message.content if message.content else None,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') and response.usage else None
            }
            
            # Check for tool calls
            if message.tool_calls:
                step["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
                    }
                    for tc in message.tool_calls
                ]
                
                # Check if provide_answer was called
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "provide_answer":
                        try:
                            answer_data = json.loads(tool_call.function.arguments)
                            step["parsed_answer"] = answer_data
                            trajectory.append(step)
                            return {
                                "question": question,
                                "answer": answer_data.get("answer", []),
                                "citations": answer_data.get("citations", []),
                                "iterations": iteration,
                                "search_history": search_history,
                                "trajectory": trajectory,
                                "model": self.model
                            }
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse provide_answer arguments: {e}")
                
                # Handle search tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                tool_results_for_trajectory = []
                
                for tool_call in message.tool_calls:
                    if tool_call.function.name == "search_documents":
                        try:
                            query = json.loads(tool_call.function.arguments)["query"]
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Warning: Could not parse search query: {e}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": "Error: Invalid search query format"
                            })
                            tool_results_for_trajectory.append({
                                "tool_call_id": tool_call.id,
                                "name": "search_documents",
                                "error": f"Invalid query format: {e}"
                            })
                            continue
                        
                        print(f"Searching: {query}")
                        results = self.search_engine.search(query, top_k)
                        print(f"Returning {len(results)} results to model")
                        
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
                        
                        # Add images if we have results
                        if results:
                            image_content = [{"type": "text", "text": "Here are the matching pages:\n"}]
                            for result in results:
                                image_content.append({
                                    "type": "text",
                                    "text": f"\nFile: {result['file']}, Page: {result['page_number']}"
                                })
                                try:
                                    image_content.append(
                                        self._load_page_image(result['file'], result['page_number'])
                                    )
                                except Exception as e:
                                    print(f"Warning: Could not load {result['file']} page {result['page_number']}: {e}")
                            
                            messages.append({"role": "user", "content": image_content})
                    
                    elif tool_call.function.name == "provide_answer":
                        # Already handled above, but add tool response for consistency
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Answer recorded."
                        })
                
                step["tool_results"] = tool_results_for_trajectory
                trajectory.append(step)
            
            else:
                # No tool calls - try to parse answer from text
                text_content = message.content or ""
                print(f"No tool call, got text response: {text_content[:200]}...")
                
                parsed = self._parse_answer_from_text(text_content)
                step["raw_response"] = text_content
                step["parsed_answer"] = parsed
                trajectory.append(step)
                
                return {
                    "question": question,
                    "answer": parsed["answer"],
                    "citations": parsed["citations"],
                    "iterations": iteration,
                    "search_history": search_history,
                    "trajectory": trajectory,
                    "model": self.model,
                    "raw_response": text_content
                }
        
        # Max iterations reached
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
    parser = argparse.ArgumentParser(description="Search agent baseline for document QA (vLLM)")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--ocr-file", default="data/ocr_output.jsonl",
                       help="Path to OCR results JSONL file")
    parser.add_argument("--model", default="default",
                       help="Model name on vLLM server")
    parser.add_argument("--base-url",
                       help="vLLM server URL (default: VLLM_BASE_URL env or http://localhost:8000/v1)")
    parser.add_argument("--api-key",
                       help="API key (default: VLLM_API_KEY env or EMPTY)")
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
    agent = VLLMSearchAgent(
        search_engine, 
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    # Answer question
    result = agent.answer_question(
        args.question, 
        args.max_iterations, 
        args.top_k
    )
    
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
        print(f"    [{search['iteration']}] '{search['query']}' -> {search['num_results']} results")
    if "error" in result:
        print(f"  Error: {result['error']}")
    print("="*80)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

