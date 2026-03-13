#!/usr/bin/env python3
"""
Search Agent Baseline for Document QA (Gemini Version)

Provides image-only search results to Google Gemini models using Whoosh search.
"""

import argparse
import json
import os
from typing import List, Dict, Any
from PIL import Image

import google.generativeai as genai

from utils import get_pdf_page_as_png, image_to_base64, WhooshSearchEngine


class GeminiSearchAgent:
    """Gemini agent with image-only search tool."""
    
    def __init__(self, search_engine: WhooshSearchEngine, model: str = "gemini-2.0-flash-exp", api_key: str = None):
        """Initialize the agent."""
        self.search_engine = search_engine
        self.model = model
        
        # Configure API key
        if api_key is not None:
            genai.configure(api_key=api_key)
        # Otherwise, it will use GOOGLE_API_KEY environment variable
        
        # Initialize model with tools
        self.gemini_model = genai.GenerativeModel(
            model_name=model,
            tools=[self._create_search_tool()]
        )
        print(f"Initialized agent with model: {self.model}")
    
    def _create_search_tool(self):
        """Create the search tool definition for Gemini."""
        return genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="search_documents",
                    description="Search document collection and return images of matching pages. Supports: terms and phrases (use quotes for exact match), boolean operators (AND, OR, NOT - AND is default), wildcards (* for multiple chars, ? for single char). Examples: 'engine specifications', '\"Bell 407\" AND accessories', 'Bell*', 'incorporation NOT date'.",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "query": genai.protos.Schema(
                                type=genai.protos.Type.STRING,
                                description="Search query using keywords, phrases in quotes, and boolean operators"
                            )
                        },
                        required=["query"]
                    )
                )
            ]
        )
    
    def _load_page_image(self, file: str, page: int) -> Image.Image:
        """Load a single page as a PIL Image."""
        return get_pdf_page_as_png(file, page)
    
    def _parse_final_answer(self, text: str) -> Dict[str, Any]:
        """Parse the final answer from model text response."""
        # Try to extract JSON from the response
        # Look for patterns like {"answer": [...], "citations": [...]}
        import re
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[\s\S]*"answer"[\s\S]*"citations"[\s\S]*\}', text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if "answer" in parsed and "citations" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing fails, return the text as a single answer with no citations
        return {
            "answer": [text.strip()],
            "citations": []
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
        
        system_instruction = """You are a document QA assistant with access to a search tool.
The search tool returns images of document pages.

IMPORTANT: The answer to the question is definitely in the documents. If your search returns no results or unhelpful pages, try different search terms. Be creative with queries - use synonyms, abbreviations, or different phrasings.

Once you find relevant pages, analyze the images carefully. When you have the answer, respond with a JSON object in this exact format:
{
  "answer": ["answer value 1", "answer value 2", ...],
  "citations": [
    {"file": "exact_filename.pdf", "page": 1},
    {"file": "another_file.pdf", "page": 3}
  ]
}

Where:
- answer: list of answer values (one or more items)
  * if there is a single answer, the output should be a one-element list
  * if the answer refers to multiple items or entities, the list will have several elements
  * do not write a full sentence there, use as few words as possible
  * if possible, use the exact words from the document
- citations: list of sources where EACH citation must have:
  * file: the exact PDF filename shown in the image (e.g., "1007969.pdf", "doc_name.pdf")
  * page: the page number (integer)
  
"""
        
        # Start chat with system instruction
        chat = self.gemini_model.start_chat(history=[])
        
        search_history = []
        trajectory = []  # Full trajectory log
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n=== Iteration {iteration} ===")
            
            # On first iteration, send the question
            if iteration == 1:
                prompt = f"{system_instruction}\n\nQuestion: {question}"
            else:
                # On subsequent iterations, just continue the conversation
                prompt = "Please continue. If you have found the answer, provide the JSON response as specified."
            
            # For the last iteration, force answer without tools
            if iteration == max_iterations:
                try:
                    # Disable tool use and force final answer
                    response = chat.send_message(
                        prompt,
                        tools=[]  # No tools available, must provide answer
                    )
                    
                    # Parse the text response
                    answer_data = self._parse_final_answer(response.text)
                    
                    # Safely get answer and citations with defaults
                    answer = answer_data.get("answer", [])
                    citations = answer_data.get("citations", [])
                    if isinstance(answer, str):
                        answer = [answer]
                    elif not isinstance(answer, list):
                        answer = [str(answer)] if answer else ["Unable to parse answer"]
                    
                    # Log final step
                    step = {
                        "iteration": iteration,
                        "role": "assistant",
                        "content": response.text,
                        "parsed_answer": answer_data,
                        "forced_answer": True
                    }
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
                except Exception as e:
                    print(f"Error on final iteration: {e}")
                    trajectory.append({
                        "iteration": iteration,
                        "role": "assistant",
                        "error": str(e)
                    })
                    # Return error result
                    return {
                        "question": question,
                        "answer": ["Error: Unable to generate answer"],
                        "citations": [],
                        "iterations": iteration,
                        "search_history": search_history,
                        "trajectory": trajectory,
                        "model": self.model,
                        "error": str(e)
                    }
            
            # Send message with tools available
            try:
                response = chat.send_message(prompt)
            except Exception as e:
                print(f"Error sending message on iteration {iteration}: {e}")
                trajectory.append({
                    "iteration": iteration,
                    "role": "assistant",
                    "error": str(e)
                })
                # Try to continue without tools on the next iteration
                continue
            
            # Build trajectory step
            step = {
                "iteration": iteration,
                "role": "assistant"
            }
            
            # Check for finish reason issues
            try:
                candidate = response.candidates[0]
                step["finish_reason"] = str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else None
                
                # Check for problematic finish reasons
                if hasattr(candidate, 'finish_reason'):
                    finish_reason_str = str(candidate.finish_reason)
                    if 'MALFORMED_FUNCTION_CALL' in finish_reason_str or finish_reason_str == '12':
                        print(f"Warning: Problematic finish reason detected: {finish_reason_str}")
                        step["error"] = f"Problematic finish reason: {finish_reason_str}"
                        trajectory.append(step)
                        # Skip this iteration and try again
                        continue
            except Exception as e:
                print(f"Error checking finish reason: {e}")
            
            # Check if model wants to use a function
            try:
                if not response.candidates[0].content.parts:
                    print("Warning: Empty response parts, skipping iteration")
                    step["error"] = "Empty response parts"
                    trajectory.append(step)
                    continue
                    
                has_function_call = False
                has_text = False
                text_content = ""
                function_calls = []
                
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        has_function_call = True
                        function_calls.append({
                            "name": part.function_call.name,
                            "args": dict(part.function_call.args) if hasattr(part.function_call, 'args') else {}
                        })
                    if part.text:
                        has_text = True
                        text_content += part.text
                
                step["content"] = text_content if text_content else None
                step["tool_calls"] = function_calls if function_calls else None
                
                # If model provides text without function call, it might be the final answer
                if has_text and not has_function_call:
                    answer_data = self._parse_final_answer(text_content)
                    
                    # Safely get answer and citations with defaults
                    answer = answer_data.get("answer", [])
                    citations = answer_data.get("citations", [])
                    if isinstance(answer, str):
                        answer = [answer]
                    
                    # Check if this looks like a valid final answer
                    if citations or (answer and len(answer[0]) > 10):
                        step["parsed_answer"] = answer_data
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
                
                # Process function calls
                if has_function_call:
                    function_responses = []
                    images_to_send = []
                    tool_results_for_trajectory = []
                    
                    for part in response.candidates[0].content.parts:
                        if part.function_call:
                            try:
                                function_call = part.function_call
                                
                                if function_call.name == "search_documents":
                                    # Check if args are valid
                                    if not hasattr(function_call, 'args') or 'query' not in function_call.args:
                                        print("Warning: Malformed function call - missing query argument")
                                        tool_results_for_trajectory.append({
                                            "name": "search_documents",
                                            "error": "Missing query argument"
                                        })
                                        continue
                                    
                                    query = function_call.args["query"]
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
                                        "name": "search_documents",
                                        "query": query,
                                        "results": [
                                            {"file": r["file"], "page": r["page_number"]}
                                            for r in results
                                        ]
                                    })
                                    
                                    # Prepare function response
                                    function_responses.append(
                                        genai.protos.Part(
                                            function_response=genai.protos.FunctionResponse(
                                                name="search_documents",
                                                response={"result": f"Found {len(results)} matching pages. Images attached below."}
                                            )
                                        )
                                    )
                                    
                                    # Load images
                                    for result in results:
                                        try:
                                            image = self._load_page_image(result['file'], result['page_number'])
                                            images_to_send.append({
                                                "image": image,
                                                "metadata": f"File: {result['file']}, Page: {result['page_number']}"
                                            })
                                        except Exception as e:
                                            print(f"Warning: Could not load {result['file']} page {result['page_number']}: {e}")
                            except Exception as e:
                                print(f"Error processing function call: {e}")
                                tool_results_for_trajectory.append({
                                    "name": function_call.name if hasattr(function_call, 'name') else "unknown",
                                    "error": str(e)
                                })
                                continue
                    
                    step["tool_results"] = tool_results_for_trajectory
                    trajectory.append(step)
                    
                    # Send function response and images back to model
                    if function_responses:
                        try:
                            # Build content parts: function responses + text labels + images
                            content_parts = function_responses
                            
                            for img_data in images_to_send:
                                # Add text label before image
                                content_parts.append(f"\n{img_data['metadata']}")
                                # Add image
                                content_parts.append(img_data['image'])
                            
                            # Send as continuation of chat
                            response = chat.send_message(content_parts)
                            
                            # Check if this response contains the final answer
                            if response.candidates[0].content.parts:
                                has_text_response = False
                                text_content = ""
                                
                                for part in response.candidates[0].content.parts:
                                    if part.text:
                                        has_text_response = True
                                        text_content += part.text
                                
                                # If we get a text response after sending images, check if it's the final answer
                                if has_text_response and text_content.strip():
                                    answer_data = self._parse_final_answer(text_content)
                                    
                                    # Check if this looks like a valid final answer with citations
                                    if answer_data["citations"]:
                                        # Log this as a separate step
                                        follow_up_step = {
                                            "iteration": iteration,
                                            "role": "assistant",
                                            "content": text_content,
                                            "parsed_answer": answer_data,
                                            "after_tool_response": True
                                        }
                                        trajectory.append(follow_up_step)
                                        
                                        return {
                                            "question": question,
                                            "answer": answer_data["answer"],
                                            "citations": answer_data["citations"],
                                            "iterations": iteration,
                                            "search_history": search_history,
                                            "trajectory": trajectory,
                                            "model": self.model
                                        }
                        except Exception as e:
                            print(f"Error sending function response on iteration {iteration}: {e}")
                            # Continue to next iteration
                            continue
                else:
                    trajectory.append(step)
            except Exception as e:
                print(f"Error processing response on iteration {iteration}: {e}")
                step["error"] = str(e)
                trajectory.append(step)
                # Continue to next iteration
                continue
        
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
    parser = argparse.ArgumentParser(description="Search agent baseline for document QA (Gemini)")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--ocr-file", default="data/ocr_output.jsonl",
                       help="Path to OCR results JSONL file")
    parser.add_argument("--model", default="gemini-2.0-flash-exp",
                       help="Gemini model name (default: gemini-2.0-flash-exp)")
    parser.add_argument("--api-key", help="Google API key (optional, uses GOOGLE_API_KEY env var)")
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
    agent = GeminiSearchAgent(search_engine, args.model, args.api_key)
    
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

