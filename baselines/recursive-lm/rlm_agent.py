#!/usr/bin/env python3
"""
RLM (Recursive Language Model) Agent for Document QA.

Uses the RLM library to process document corpus and answer questions.
Uses OpenAI structured outputs to ensure consistent answer format.
"""

import json
import os
from typing import Dict, Any, List
from pydantic import BaseModel
import openai
import anthropic


class Citation(BaseModel):
    """Citation for a source document."""
    file: str
    page: int


class Answer(BaseModel):
    """Structured answer with citations."""
    answer: List[str]
    citations: List[Citation]


class RLMAgent:
    """Document QA agent using Recursive Language Models."""
    
    def __init__(
        self, 
        hf_dataset: str = "agentic-document-ai/pdfs-to-markdown-mistral-ocr",
        backend: str = "openai",
        model_name: str = "gpt-5-mini-2025-08-07",
        verbose: bool = False
    ):
        """
        Initialize the RLM agent.
        
        Args:
            hf_dataset: HuggingFace dataset name for markdown corpus
            backend: RLM backend ("openai", "anthropic", etc.)
            model_name: Model name for the backend
            verbose: Enable verbose output
        """
        from rlm import RLM
        
        self.hf_dataset = hf_dataset
        self.backend = backend
        self.model_name = model_name
        self.verbose = verbose
        
        # Load corpus data into memory
        self.corpus = self._load_corpus()
        total_pages = sum(doc.get('num_pages', 1) for doc in self.corpus)
        print(f"Loaded {len(self.corpus)} files ({total_pages} total pages) from corpus")
        
        # Initialize RLM
        self.rlm = RLM(
            backend=backend,
            backend_kwargs={"model_name": model_name},
            verbose=verbose
        )
        print(f"Initialized RLM with {backend}/{model_name}")
        
        # Initialize clients for structured output
        self.openai_client = None
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)

        self.anthropic_client = None
        if backend == "anthropic":
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if not anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required for Anthropic structured output."
                )
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        elif backend == "anthropic_bedrock":
            bedrock_kwargs = {}
            if os.getenv("AWS_REGION"):
                bedrock_kwargs["aws_region"] = os.getenv("AWS_REGION")
            if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                bedrock_kwargs["aws_access_key"] = os.getenv("AWS_ACCESS_KEY_ID")
                bedrock_kwargs["aws_secret_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")
            self.anthropic_client = anthropic.AnthropicBedrock(**bedrock_kwargs)
    
    def _load_corpus(self) -> List[Dict[str, Any]]:
        """Load corpus from HuggingFace dataset (one entry per PDF file)."""
        from datasets import load_dataset
        import json as json_module
        
        print(f"Loading corpus from HuggingFace: {self.hf_dataset}")
        dataset = load_dataset(self.hf_dataset, split="train")
        
        corpus = []
        for item in dataset:
            # Dataset has: pdf_name, markdown (full doc), metadata (JSON string)
            metadata = {}
            if "metadata" in item and item["metadata"]:
                try:
                    metadata = json_module.loads(item["metadata"])
                except:
                    pass
            
            corpus.append({
                "file": item.get("pdf_name", item.get("file", "")),
                "text": item.get("markdown", ""),
                "num_pages": metadata.get("num_pages", 1),
            })
        return corpus
    
    def _build_corpus_text(self) -> str:
        """Build corpus text for RLM context with explicit page markers.
        
        Mistral OCR separates pages with '---' (markdown horizontal rule).
        We split on this and add explicit [PAGE N] markers.
        """
        parts = []
        for doc in self.corpus:
            file_name = doc['file']
            markdown = doc.get('text', '')
            num_pages = doc.get('num_pages', 1)
            
            # Split by page separator (---)
            pages = markdown.split('\n---\n')
            
            parts.append(f"[FILE: {file_name} | TOTAL PAGES: {num_pages}]")
            
            for i, page_content in enumerate(pages, 1):
                parts.append(f"[PAGE {i}]")
                parts.append(page_content.strip())
                parts.append(f"[END PAGE {i}]")
            
            parts.append(f"[END FILE: {file_name}]\n")
        
        return "\n".join(parts)
    
    def _format_answer_with_structured_output(self, question: str, rlm_response: str) -> Answer:
        """Use OpenAI structured output to format the RLM response."""
        system_prompt = """You are a precise answer formatter. Given an RLM analysis response, extract the answer and citations.

Rules:
- answer: List of answer strings. Extract the actual answer values found.
- citations: List of {file, page} objects. Use exact PDF filename and page number.
- Page numbers are from [PAGE N] markers in the corpus (1-indexed).
- If the response contains no valid answer, return empty lists.
- Do NOT include raw data dumps, analysis metadata, or suggestions as answers.
- Return ONLY valid JSON with keys: answer, citations."""

        user_prompt = (
            f"Question: {question}\n\nRLM Analysis Response:\n{rlm_response[:50000]}"
        )

        if self.openai_client:
            response = self.openai_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=Answer,
            )
            return response.choices[0].message.parsed

        if self.anthropic_client:
            response = self.anthropic_client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parsed = self._parse_json_answer(response.content[0].text)
            return parsed

        # For Corvo and other backends, use the same client for JSON extraction
        if self.backend == "corvo":
            from rlm.clients.corvo import CorvoClient
            corvo_client = CorvoClient(model_name=self.model_name, max_output_tokens=2048)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = corvo_client.completion(full_prompt)
            parsed = self._parse_json_answer(response)
            return parsed

        raise ValueError(
            "No structured output client available. Set OPENAI_API_KEY or use an Anthropic backend."
        )

    def _parse_json_answer(self, text: str) -> Answer:
        """Parse an Answer object from a JSON string (or JSON block) output."""
        import re
        
        if not text or not text.strip():
            # Empty response - return empty answer
            return Answer(answer=[], citations=[])
        
        cleaned = text.strip()
        
        # Try to extract JSON from code blocks first
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                if "json" in part[:10]:
                    cleaned = part.split("json", 1)[-1].strip()
                    break
            else:
                # Try the content between first pair of ```
                if len(parts) > 1:
                    cleaned = parts[1].strip()
        
        # Try to find JSON object in the text
        json_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        
        # If still no valid JSON found, try to find any JSON object
        if not cleaned.startswith('{'):
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # If JSON parsing fails, return empty answer
            return Answer(answer=[], citations=[])
        
        if hasattr(Answer, "model_validate"):
            return Answer.model_validate(data)
        return Answer.parse_obj(data)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using RLM + structured output.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict with question, answer, citations, and metadata
        """
        # Build the corpus context
        corpus_text = self._build_corpus_text()
        
        # Run RLM to analyze the corpus
        full_prompt = f"""
# The document corpus is available as the variable 'corpus'
corpus = '''{corpus_text}'''

# Question: {question}
#
# The corpus is structured with markers:
# - [FILE: xxx.pdf | TOTAL PAGES: N] ... [END FILE: xxx.pdf] for each document
# - [PAGE N] ... [END PAGE N] for each page within a document
#
# Search the corpus to find the answer. Return:
# 1. The specific answer(s) to the question
# 2. The source file AND page number where the answer was found
#    Example: "Found in 24357345.pdf, page 5"
"""
        
        try:
            # Try up to 2 times if response is empty
            rlm_response = None
            result = None
            for attempt in range(2):
                result = self.rlm.completion(full_prompt, root_prompt=question)
                rlm_response = result.response
                if rlm_response and rlm_response.strip():
                    break
                if self.verbose:
                    print(f"Empty response on attempt {attempt + 1}, retrying...")
            
            # Use structured output to format the answer
            structured_answer = self._format_answer_with_structured_output(question, rlm_response)
            
            # If structured extraction returned empty but raw response has content,
            # try to extract answers directly from the raw response
            if not structured_answer.answer and rlm_response and rlm_response.strip():
                fallback_answers = self._extract_answers_from_raw(rlm_response)
                if fallback_answers:
                    structured_answer = Answer(answer=fallback_answers, citations=[])
            
            # Extract trajectory from RLM result
            trajectory = self._extract_trajectory(result)
            
            return {
                "question": question,
                "answer": structured_answer.answer,
                "citations": [{"file": c.file, "page": c.page} for c in structured_answer.citations],
                "model": f"{self.backend}/{self.model_name}",
                "raw_response": rlm_response,
                "trajectory": trajectory
            }
            
        except Exception as e:
            import traceback
            return {
                "question": question,
                "answer": [],
                "citations": [],
                "error": str(e),
                "error_traceback": traceback.format_exc(),
                "model": f"{self.backend}/{self.model_name}"
            }
    
    def _extract_answers_from_raw(self, raw_response: str) -> List[str]:
        """Extract answers directly from raw response when structured extraction fails.
        
        This handles cases where the raw response is a simple comma-separated list or
        a single answer without JSON formatting.
        """
        if not raw_response:
            return []
        
        cleaned = raw_response.strip()
        
        # If the response looks like a simple answer (no JSON, no complex formatting)
        # and is relatively short, treat it as a direct answer
        if len(cleaned) < 1000 and '{' not in cleaned and '[' not in cleaned:
            # Check if it's a comma-separated list
            if ',' in cleaned:
                # Split by comma and clean up each part
                parts = [p.strip() for p in cleaned.split(',')]
                # Filter out empty parts and parts that look like metadata
                answers = [p for p in parts if p and not p.lower().startswith(('found in', 'page', 'source:'))]
                if answers:
                    return answers
            else:
                # Single answer
                if cleaned and not cleaned.lower().startswith(('found in', 'page', 'source:', 'no answer', 'i could not')):
                    return [cleaned]
        
        return []
    
    def _extract_trajectory(self, result) -> Dict[str, Any]:
        """Extract trajectory/usage stats from RLM result (RLMChatCompletion)."""
        trajectory = {
            "execution_time": getattr(result, 'execution_time', None),
        }
        
        # Extract usage summary - RLMChatCompletion has usage_summary.model_usage_summaries
        if hasattr(result, 'usage_summary') and result.usage_summary:
            usage = result.usage_summary
            if hasattr(usage, 'model_usage_summaries'):
                total_calls = 0
                total_input_tokens = 0
                total_output_tokens = 0
                
                for model_name, model_usage in usage.model_usage_summaries.items():
                    total_calls += getattr(model_usage, 'total_calls', 0)
                    total_input_tokens += getattr(model_usage, 'total_input_tokens', 0)
                    total_output_tokens += getattr(model_usage, 'total_output_tokens', 0)
                
                trajectory["llm_calls"] = total_calls
                trajectory["input_tokens"] = total_input_tokens
                trajectory["output_tokens"] = total_output_tokens
            elif hasattr(usage, 'to_dict'):
                trajectory["usage"] = usage.to_dict()
        
        return trajectory


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RLM Agent for Document QA")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--corpus-dataset", default="agentic-document-ai/pdfs-to-markdown-mistral-ocr",
                       help="HuggingFace dataset for markdown corpus")
    parser.add_argument("--backend", default="openai",
                       help="RLM backend (default: openai)")
    parser.add_argument("--model", default="gpt-5-mini-2025-08-07",
                       help="Model name")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--output", "-o", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RLMAgent(
        hf_dataset=args.corpus_dataset,
        backend=args.backend,
        model_name=args.model,
        verbose=args.verbose
    )
    
    # Answer question
    result = agent.answer_question(args.question)
    
    # Print result
    print("\n" + "="*80)
    print("QUESTION:", result["question"])
    print("\nANSWER:", json.dumps(result["answer"], indent=2))
    print("\nCITATIONS:", json.dumps(result["citations"], indent=2))
    print("\nMODEL:", result["model"])
    print("="*80)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
