#!/usr/bin/env python3
"""
Two-Stage HEAVEN Retrieval Pipeline.

Based on: https://github.com/juyeonnn/HEAVEN
Paper: https://arxiv.org/pdf/2510.22215

Stage 1: Fast retrieval using single-vector embeddings (with VS-Page filtering)
Stage 2: Accurate re-ranking using multi-vector MaxSim scoring (with query token filtering)

Key features:
- VS-Page filtering for efficient initial candidate retrieval
- Query token filtering to reduce Stage 2 computation
- Hybrid α/β weighting for score combination
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    index: int
    pdf_name: str
    page_num: int
    score: float
    stage: int  # Which stage produced this result


class HEAVENRetriever:
    """
    Full HEAVEN two-stage retrieval pipeline.
    
    Stage 1: 
    - First filters at VS-page level for efficiency
    - Then retrieves candidates at page level
    
    Stage 2:
    - Re-ranks candidates using multi-vector MaxSim
    - Uses query token filtering to reduce computation
    """
    
    def __init__(
        self,
        index,  # DocumentIndex
        stage1_encoder=None,
        stage2_encoder=None,
        stage1_k: int = 100,  # Candidates from stage 1
        stage2_k: int = 10,   # Final results after stage 2
        alpha: float = 0.0,   # Stage 1 hybrid weight (0 = visual only)
        beta: float = 0.0,    # Stage 2 score combination weight
        vs_page_filter_ratio: float = 0.5,  # Ratio of VS-pages to consider
        query_filter_ratio: float = 0.5,    # Ratio of query tokens to keep
        use_query_filtering: bool = True,   # Whether to filter query tokens
    ):
        """
        Initialize HEAVEN retriever.
        
        Args:
            index: DocumentIndex with embeddings
            stage1_encoder: Encoder for query embedding (stage 1)
            stage2_encoder: Encoder for query embedding (stage 2)
            stage1_k: Number of candidates to retrieve in stage 1
            stage2_k: Number of final results after stage 2 re-ranking
            alpha: Stage 1 hybrid weight for combining visual + text scores
            beta: Stage 2 weight for score combination
            vs_page_filter_ratio: Fraction of VS-pages to consider in filtering
            query_filter_ratio: Fraction of query tokens to keep in Stage 2
            use_query_filtering: Whether to apply query token filtering
        """
        self.index = index
        self.stage1_encoder = stage1_encoder
        self.stage2_encoder = stage2_encoder
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k
        self.alpha = alpha
        self.beta = beta
        self.vs_page_filter_ratio = vs_page_filter_ratio
        self.query_filter_ratio = query_filter_ratio
        self.use_query_filtering = use_query_filtering
        
        # Initialize query token filter for Stage 2 (using POS tagging like paper)
        self.token_filter = None
        if use_query_filtering:
            from query_filtering import POSTokenFilter
            self.token_filter = POSTokenFilter()
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve documents for a query using two-stage pipeline.
        
        Args:
            query: Text query
            k: Number of results to return (defaults to stage2_k)
        
        Returns:
            List of RetrievalResult objects
        """
        k = k or self.stage2_k
        
        # Stage 1: Fast retrieval (with VS-page filtering if available)
        stage1_results = self._stage1_retrieve(query, self.stage1_k)
        
        # If no stage 2 encoder or embeddings, return stage 1 results
        if self.stage2_encoder is None or self.index.stage2_embeddings is None:
            return stage1_results[:k]
        
        # Stage 2: Re-rank using multi-vector similarity (with query filtering)
        stage2_results = self._stage2_rerank(query, stage1_results, k)
        
        return stage2_results
    
    def _stage1_retrieve(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Stage 1: Fast retrieval using single-vector similarity.
        
        If VS-pages are available:
        1. First filter at VS-page level
        2. Then rank pages within selected VS-pages
        
        Otherwise: direct page-level retrieval.
        """
        if self.index.stage1_embeddings is None:
            raise ValueError("No stage 1 embeddings in index")
        
        if self.stage1_encoder is None:
            raise ValueError("No stage 1 encoder provided")
        
        # Encode query
        query_emb = self.stage1_encoder.encode_queries([query])  # (1, D)
        
        # Check if VS-pages are available
        if self.index.vs_page_embeddings is not None and self.index.vs_page_metadata is not None:
            return self._stage1_with_vs_pages(query_emb, k)
        else:
            return self._stage1_direct(query_emb, k)
    
    def _stage1_with_vs_pages(self, query_emb: np.ndarray, k: int) -> List[RetrievalResult]:
        """
        Stage 1 retrieval with VS-page filtering (paper implementation).
        
        Following the paper:
        1. Compute VS-page scores for all pages (max across VS-pages containing each page)
        2. Filter pages based on VS-page scores (keep top filter_ratio)
        3. Combine: final_score = alpha * vs_page_score + (1-alpha) * page_score
        """
        num_pages = len(self.index.doc_metadata)
        
        # Step 1: Compute page-level VS-page scores
        # For each page, get max score across all VS-pages that contain it
        vs_page_similarities = np.dot(self.index.vs_page_embeddings, query_emb.T).flatten()
        
        page_vs_scores = np.zeros(num_pages)
        for page_idx in range(num_pages):
            vs_indices = self.index.get_vs_pages_for_page(page_idx)
            if vs_indices:
                page_vs_scores[page_idx] = max(vs_page_similarities[vi] for vi in vs_indices)
        
        # Step 2: Filter pages based on VS-page scores
        num_keep = max(1, int(num_pages * self.vs_page_filter_ratio))
        top_page_indices = np.argsort(page_vs_scores)[::-1][:num_keep]
        
        # Create mask for filtered pages
        mask = np.zeros(num_pages)
        mask[top_page_indices] = 1.0
        
        # Step 3: Compute page scores
        page_similarities = np.dot(self.index.stage1_embeddings, query_emb.T).flatten()
        
        # Apply mask to page scores (zero out filtered pages)
        masked_page_similarities = page_similarities * mask
        
        # Step 4: Combine scores using alpha (paper formula)
        # final = alpha * vs_page_score + (1-alpha) * page_score
        final_scores = self.alpha * page_vs_scores + (1 - self.alpha) * masked_page_similarities
        
        # Get top-k
        top_indices = np.argsort(final_scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            pdf_name, page_num = self.index.doc_metadata[idx]
            results.append(RetrievalResult(
                index=int(idx),
                pdf_name=pdf_name,
                page_num=page_num,
                score=float(final_scores[idx]),
                stage=1
            ))
        
        return results
    
    def _stage1_direct(self, query_emb: np.ndarray, k: int) -> List[RetrievalResult]:
        """Direct Stage 1 retrieval without VS-pages."""
        # Compute similarities with all pages
        similarities = np.dot(self.index.stage1_embeddings, query_emb.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            pdf_name, page_num = self.index.doc_metadata[idx]
            results.append(RetrievalResult(
                index=int(idx),
                pdf_name=pdf_name,
                page_num=page_num,
                score=float(similarities[idx]),
                stage=1
            ))
        
        return results
    
    def _stage2_rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """
        Stage 2: Re-rank candidates using multi-vector MaxSim with query token filtering.
        
        Following paper's hierarchical filtering:
        1. Filter top-k from stage 1 (already done)
        2. Score with key tokens (nouns) -> further filter to k_refine
        3. Score with non-key tokens (others) on refined candidates
        4. Combine: beta * stage1 + (1-beta) * (key_score + non_key_score)
        """
        if self.stage2_encoder is None:
            raise ValueError("No stage 2 encoder provided")
        
        # Encode query with multi-vector encoder
        # ColQwen encoder now returns (embeddings, input_ids) tuple
        encode_result = self.stage2_encoder.encode_queries([query])
        
        # Handle both old (list) and new (tuple) return formats
        if isinstance(encode_result, tuple):
            query_embs, input_ids_list = encode_result
            query_emb = query_embs[0]  # (num_tokens, dim)
            input_ids = input_ids_list[0] if input_ids_list else None
        else:
            query_emb = encode_result[0]
            input_ids = None
        
        if self.use_query_filtering and self.token_filter is not None:
            # Split query into key (nouns) and non-key (others) token embeddings
            # Pass input_ids for proper tokenizer-aligned POS tagging
            key_emb, non_key_emb, key_indices, non_key_indices = self.token_filter.split_embeddings(
                query, query_emb, input_ids
            )
            
            # Step 1: Score all candidates with key tokens
            key_scores = {}
            for candidate in candidates:
                doc_emb = self.index.stage2_embeddings[candidate.index]
                if len(key_emb) > 0:
                    key_scores[candidate.index] = self._compute_maxsim(key_emb, doc_emb)
                else:
                    key_scores[candidate.index] = self._compute_maxsim(query_emb, doc_emb)
            
            # Step 2: Filter to k_refine using key token scores
            k_refine = max(k, int(len(candidates) * self.query_filter_ratio))
            sorted_by_key = sorted(candidates, key=lambda c: key_scores[c.index], reverse=True)
            refined_candidates = sorted_by_key[:k_refine]
            
            # Step 3: Score refined candidates with non-key tokens
            scored_candidates = []
            for candidate in refined_candidates:
                doc_emb = self.index.stage2_embeddings[candidate.index]
                
                if len(non_key_emb) > 0:
                    non_key_score = self._compute_maxsim(non_key_emb, doc_emb)
                else:
                    non_key_score = 0.0
                
                # Combine key + non-key scores
                multi_vector_score = key_scores[candidate.index] + non_key_score
                
                # Combine with stage 1 using beta
                combined_score = self.beta * candidate.score + (1 - self.beta) * multi_vector_score
                
                scored_candidates.append(RetrievalResult(
                    index=candidate.index,
                    pdf_name=candidate.pdf_name,
                    page_num=candidate.page_num,
                    score=combined_score,
                    stage=2
                ))
        else:
            # No filtering - use all query tokens
            scored_candidates = []
            for candidate in candidates:
                doc_emb = self.index.stage2_embeddings[candidate.index]
                maxsim_score = self._compute_maxsim(query_emb, doc_emb)
                
                combined_score = self.beta * candidate.score + (1 - self.beta) * maxsim_score
                
                scored_candidates.append(RetrievalResult(
                    index=candidate.index,
                    pdf_name=candidate.pdf_name,
                    page_num=candidate.page_num,
                    score=combined_score,
                    stage=2
                ))
        
        # Sort by score and return top-k
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:k]
    
    def _compute_maxsim(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """
        Compute MaxSim score between query and document embeddings.
        
        For each query token, find max similarity with any document token.
        Sum over all query tokens to get final score.
        """
        # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
        similarities = np.dot(query_emb, doc_emb.T)
        
        # For each query token, take max over all doc tokens
        max_sims = similarities.max(axis=1)
        
        # Sum over all query tokens
        return float(max_sims.sum())
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Retrieve with detailed scoring information.
        
        Returns:
            Tuple of (results, metadata dict with stage info)
        """
        k = k or self.stage2_k
        
        # Stage 1
        stage1_results = self._stage1_retrieve(query, self.stage1_k)
        stage1_top_scores = [r.score for r in stage1_results[:10]]
        
        # VS-page info
        vs_page_info = None
        if self.index.vs_page_embeddings is not None:
            query_emb = self.stage1_encoder.encode_queries([query])
            vs_similarities = np.dot(self.index.vs_page_embeddings, query_emb.T).flatten()
            top_vs_idx = np.argmax(vs_similarities)
            vs_page_info = {
                "num_vs_pages": len(vs_similarities),
                "top_vs_page": int(top_vs_idx),
                "top_vs_score": float(vs_similarities[top_vs_idx])
            }
        
        # Stage 2 (if available)
        if self.stage2_encoder is not None and self.index.stage2_embeddings is not None:
            final_results = self._stage2_rerank(query, stage1_results, k)
            stage2_top_scores = [r.score for r in final_results[:10]]
        else:
            final_results = stage1_results[:k]
            stage2_top_scores = None
        
        metadata = {
            "query": query,
            "stage1_candidates": len(stage1_results),
            "stage1_top_scores": stage1_top_scores,
            "vs_page_info": vs_page_info,
            "stage2_reranked": stage2_top_scores is not None,
            "stage2_top_scores": stage2_top_scores,
            "query_filtering_enabled": self.use_query_filtering,
            "final_results": len(final_results)
        }
        
        return final_results, metadata


class SimpleRetriever:
    """
    Simple single-stage retriever (baseline).
    
    Uses only single-vector similarity without re-ranking.
    """
    
    def __init__(self, index, encoder, k: int = 10):
        self.index = index
        self.encoder = encoder
        self.k = k
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve documents using single-vector similarity."""
        k = k or self.k
        
        # Encode query
        query_emb = self.encoder.encode_queries([query])
        
        # Compute similarities
        similarities = np.dot(self.index.stage1_embeddings, query_emb.T).flatten()
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            pdf_name, page_num = self.index.doc_metadata[idx]
            results.append(RetrievalResult(
                index=int(idx),
                pdf_name=pdf_name,
                page_num=page_num,
                score=float(similarities[idx]),
                stage=1
            ))
        
        return results
