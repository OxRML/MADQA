#!/usr/bin/env python3
"""
Query Token Filtering for HEAVEN Stage 2.

Copied from: https://github.com/juyeonnn/HEAVEN
- split_query from retrieval/heaven/utils.py
- QueryPreprocessor from retrieval/heaven/preprocess.py

Key method: POS-based filtering - keep nouns (key tokens), filter others
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import re


def _setup_nltk():
    """Download required NLTK data for POS tagging."""
    import nltk
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("Downloading NLTK POS tagger data...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('punkt', quiet=True)


def split_query(pos_tag: List[str]) -> Tuple[List[int], List[int]]:
    """
    Split query tokens into key tokens (nouns) and non-key tokens (others).
    
    Copied directly from: https://github.com/juyeonnn/HEAVEN/blob/main/retrieval/heaven/utils.py
    
    Args:
        pos_tag: List of POS tags for each token
        
    Returns:
        Tuple of (key_token_indices, non_key_token_indices)
    """
    query_1, query_2 = [], []
    for idx, item in enumerate(pos_tag):
        if not item:
            continue
        if item.startswith('N'):  # Nouns
            query_1.append(idx)
        else:  # Other tokens
            query_2.append(idx)
    return query_1, query_2


# Alias for backwards compatibility
split_query_by_pos = split_query


# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has',
    'have', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'you', 'your', 'this', 'what',
    'which', 'who', 'how', 'when', 'where', 'why', 'can', 'could', 'would',
    'should', 'do', 'does', 'did', 'been', 'being', 'had', 'has', 'have',
    'having', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'just', 'don', 'now', 'll', 've', 'd', 'm', 're', 'ain', 'aren', 'couldn',
    'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'
}


class QueryTokenFilter:
    """
    Filters query tokens by linguistic importance.
    
    Used in Stage 2 to reduce the number of query tokens for MaxSim,
    keeping only the most important tokens.
    """
    
    def __init__(
        self,
        method: str = "tfidf",
        filter_ratio: float = 0.5,  # Keep top 50% of tokens
        min_tokens: int = 3,  # Always keep at least this many
        remove_stop_words: bool = True,
        idf_corpus: Optional[List[str]] = None
    ):
        """
        Initialize query token filter.
        
        Args:
            method: Filtering method ('tfidf', 'attention', 'hybrid')
            filter_ratio: Fraction of tokens to keep (0-1)
            min_tokens: Minimum number of tokens to keep
            remove_stop_words: Whether to filter stop words
            idf_corpus: Optional corpus for IDF computation
        """
        self.method = method
        self.filter_ratio = filter_ratio
        self.min_tokens = min_tokens
        self.remove_stop_words = remove_stop_words
        
        # IDF scores (computed from corpus)
        self.idf_scores: Dict[str, float] = {}
        
        if idf_corpus:
            self._compute_idf(idf_corpus)
    
    def _compute_idf(self, corpus: List[str]):
        """Compute IDF scores from a corpus of documents."""
        # Document frequency
        df = Counter()
        n_docs = len(corpus)
        
        for doc in corpus:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
        
        # IDF = log(N / df)
        for token, freq in df.items():
            self.idf_scores[token] = np.log(n_docs / (freq + 1))
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def compute_token_importance(
        self,
        query: str,
        query_embeddings: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        """
        Compute importance scores for each token in the query.
        
        Args:
            query: Input query string
            query_embeddings: Optional multi-vector embeddings (num_tokens, dim)
            
        Returns:
            List of (token, importance_score) tuples
        """
        tokens = self._tokenize(query)
        
        if self.method == "tfidf":
            scores = self._tfidf_importance(tokens)
        elif self.method == "attention" and query_embeddings is not None:
            scores = self._attention_importance(tokens, query_embeddings)
        elif self.method == "hybrid" and query_embeddings is not None:
            tfidf_scores = self._tfidf_importance(tokens)
            attn_scores = self._attention_importance(tokens, query_embeddings)
            # Combine scores
            scores = [(t, 0.5 * s1 + 0.5 * s2) 
                     for (t, s1), (_, s2) in zip(tfidf_scores, attn_scores)]
        else:
            # Default: TF-IDF
            scores = self._tfidf_importance(tokens)
        
        return scores
    
    def _tfidf_importance(self, tokens: List[str]) -> List[Tuple[str, float]]:
        """Compute TF-IDF based importance."""
        # Term frequency in query
        tf = Counter(tokens)
        n_tokens = len(tokens)
        
        scores = []
        for token in tokens:
            # TF component
            tf_score = tf[token] / n_tokens
            
            # IDF component (default to high score if unknown)
            idf_score = self.idf_scores.get(token, 5.0)
            
            # Penalty for stop words
            if self.remove_stop_words and token in STOP_WORDS:
                tfidf = 0.0
            else:
                tfidf = tf_score * idf_score
            
            scores.append((token, tfidf))
        
        return scores
    
    def _attention_importance(
        self,
        tokens: List[str],
        embeddings: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Compute importance based on embedding norms.
        
        Higher norm often indicates more important tokens.
        """
        # Compute L2 norm of each token embedding
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Handle mismatch between tokens and embeddings
        # (embeddings may include special tokens)
        if len(norms) != len(tokens):
            # Assume first and last are special tokens
            if len(norms) > len(tokens):
                norms = norms[1:len(tokens)+1]
        
        scores = []
        for i, token in enumerate(tokens):
            if i < len(norms):
                # Penalize stop words
                if self.remove_stop_words and token in STOP_WORDS:
                    score = 0.0
                else:
                    score = float(norms[i])
            else:
                score = 0.0
            scores.append((token, score))
        
        return scores
    
    def filter_tokens(
        self,
        query: str,
        query_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Filter query embeddings to keep only important tokens.
        
        Args:
            query: Input query string
            query_embeddings: Multi-vector embeddings (num_tokens, dim)
            
        Returns:
            Tuple of (filtered_embeddings, kept_indices)
        """
        tokens = self._tokenize(query)
        importance_scores = self.compute_token_importance(query, query_embeddings)
        
        # Determine how many tokens to keep
        n_tokens = len(tokens)
        n_keep = max(self.min_tokens, int(n_tokens * self.filter_ratio))
        n_keep = min(n_keep, n_tokens)
        
        # Sort by importance and get top indices
        indexed_scores = [(i, score) for i, (_, score) in enumerate(importance_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        kept_indices = sorted([idx for idx, _ in indexed_scores[:n_keep]])
        
        # Handle embedding dimension mismatch
        if query_embeddings.shape[0] != n_tokens:
            # Assume embeddings include special tokens
            # Adjust indices accordingly
            offset = (query_embeddings.shape[0] - n_tokens) // 2
            kept_indices = [idx + offset for idx in kept_indices]
            kept_indices = [idx for idx in kept_indices if 0 <= idx < query_embeddings.shape[0]]
        
        if len(kept_indices) == 0:
            # Fallback: keep all
            return query_embeddings, list(range(query_embeddings.shape[0]))
        
        filtered_embeddings = query_embeddings[kept_indices]
        
        return filtered_embeddings, kept_indices


def compute_filtered_maxsim(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    query: str,
    token_filter: QueryTokenFilter
) -> float:
    """
    Compute MaxSim score with query token filtering.
    
    Args:
        query_emb: Query embeddings (num_query_tokens, dim)
        doc_emb: Document embeddings (num_doc_tokens, dim)
        query: Original query string (for token filtering)
        token_filter: QueryTokenFilter instance
        
    Returns:
        MaxSim score
    """
    # Filter query tokens
    filtered_query_emb, _ = token_filter.filter_tokens(query, query_emb)
    
    # Compute similarity matrix
    similarities = np.dot(filtered_query_emb, doc_emb.T)
    
    # MaxSim: for each query token, take max over doc tokens, then sum
    max_sims = similarities.max(axis=1)
    
    return float(max_sims.sum())


class AdaptiveTokenFilter(QueryTokenFilter):
    """
    Adaptive token filtering that adjusts filter ratio based on query length.
    
    Short queries keep more tokens, long queries are filtered more aggressively.
    """
    
    def __init__(
        self,
        base_filter_ratio: float = 0.5,
        min_filter_ratio: float = 0.3,
        max_filter_ratio: float = 0.8,
        **kwargs
    ):
        super().__init__(filter_ratio=base_filter_ratio, **kwargs)
        self.base_filter_ratio = base_filter_ratio
        self.min_filter_ratio = min_filter_ratio
        self.max_filter_ratio = max_filter_ratio
    
    def filter_tokens(
        self,
        query: str,
        query_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, List[int]]:
        """Filter with adaptive ratio based on query length."""
        tokens = self._tokenize(query)
        n_tokens = len(tokens)
        
        # Adapt filter ratio based on query length
        if n_tokens <= 5:
            # Short query: keep more tokens
            self.filter_ratio = self.max_filter_ratio
        elif n_tokens >= 20:
            # Long query: filter more aggressively
            self.filter_ratio = self.min_filter_ratio
        else:
            # Linear interpolation
            t = (n_tokens - 5) / 15
            self.filter_ratio = self.max_filter_ratio - t * (self.max_filter_ratio - self.min_filter_ratio)
        
        return super().filter_tokens(query, query_embeddings)


# Key POS tags are nouns (tags starting with 'N')
# Following the paper's split_query logic
KEY_POS_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}  # Noun types in Penn Treebank tagset


class POSTokenFilter:
    """
    POS-based query token filtering matching original HEAVEN implementation.
    
    Uses NLTK POS tagging to split tokens into:
    - Key tokens (nouns): Used for primary MaxSim scoring
    - Non-key tokens (others): Used for refinement scoring
    
    This matches the query preprocessing in:
    https://github.com/juyeonnn/HEAVEN/blob/main/retrieval/heaven/preprocess.py
    
    IMPORTANT: The original HEAVEN uses ColQwen's tokenizer to get tokens,
    then applies NLTK POS tagging to those decoded tokens. This ensures
    alignment between token embeddings and POS tags.
    """
    
    def __init__(self, tokenizer=None):
        """Initialize POS token filter with NLTK.
        
        Args:
            tokenizer: Optional ColQwen tokenizer for proper token decoding.
                      If None, will try to load from colpali_engine.
        """
        _setup_nltk()
        import nltk
        self.nltk = nltk
        self.tokenizer = tokenizer
        
        # Try to load ColQwen tokenizer if not provided
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
                print("Loaded Qwen2.5 tokenizer for query filtering")
            except Exception as e:
                print(f"Warning: Could not load tokenizer: {e}")
                print("Query filtering will use NLTK word_tokenize (less accurate)")
    
    def decode_input_ids(self, input_ids: 'torch.Tensor') -> List[Optional[str]]:
        """
        Decode token IDs to text, matching original HEAVEN preprocessing.
        
        Args:
            input_ids: Tensor of token IDs from ColQwen encoder
            
        Returns:
            List of decoded tokens (None for padding/special tokens)
        """
        if self.tokenizer is None:
            return None
        
        import torch
        decoded_tokens = []
        
        for token_id in input_ids.flatten():
            token_id = int(token_id)
            if token_id == -1:  # Padding
                decoded_tokens.append(None)
            else:
                decoded_tokens.append(
                    self.tokenizer.decode(token_id, skip_special_tokens=False)
                )
        return decoded_tokens
    
    def clean_tokens(self, tokens: List[Optional[str]]) -> Tuple[List[bool], List[str]]:
        """
        Clean tokenized query by removing special tokens and padding.
        Matching original HEAVEN's clean_tokens logic.
        
        Args:
            tokens: List of decoded tokens
            
        Returns:
            Tuple of (mask, cleaned_tokens) where mask indicates valid token positions
        """
        mask = []
        cleaned = []
        
        for token in tokens:
            # Skip None and special tokens
            if token is not None and token != "<|endoftext|>":
                # Normalize uppercase tokens
                if token.upper() == token:
                    token = token.lower()
                cleaned.append(token.strip())
                mask.append(True)
            else:
                mask.append(False)
        
        return mask, cleaned
    
    def get_pos_tags(self, tokens: List[str]) -> List[str]:
        """
        Get POS tags for a list of tokens using NLTK.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of POS tags
        """
        # Get POS tags from NLTK
        pos_tagged = self.nltk.pos_tag(tokens)
        return [tag for _, tag in pos_tagged]
    
    def get_pos_tags_aligned(self, tokens: List[str], mask: List[bool]) -> List[Optional[str]]:
        """
        Get POS tags aligned to original token positions (including special tokens).
        
        This matches original HEAVEN's preprocessing.
        
        Args:
            tokens: List of cleaned tokens
            mask: Boolean mask indicating valid positions
            
        Returns:
            List of POS tags aligned with original positions (None for invalid)
        """
        # Get POS tags from NLTK for cleaned tokens
        nltk_pos_tags = self.nltk.pos_tag(tokens)
        
        # Map back to original positions
        pos_tags = []
        nltk_idx = 0
        
        for i, is_valid in enumerate(mask):
            if is_valid and nltk_idx < len(nltk_pos_tags):
                pos_tag = nltk_pos_tags[nltk_idx][1]
                # Mark first valid token as QUERY (special handling in original)
                if nltk_idx == 0:
                    pos_tag = "QUERY"
                pos_tags.append(pos_tag)
                nltk_idx += 1
            else:
                pos_tags.append(None)
        
        return pos_tags
    
    def split_embeddings(
        self,
        query: str,
        query_embeddings: np.ndarray,
        input_ids: Optional['torch.Tensor'] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        """
        Split query embeddings into key (nouns) and non-key (others).
        
        This matches the original HEAVEN Stage 2 query token filtering.
        
        Args:
            query: Original query string
            query_embeddings: Multi-vector embeddings (num_tokens, dim)
            input_ids: Optional input_ids tensor from ColQwen for proper alignment
            
        Returns:
            Tuple of (key_embeddings, non_key_embeddings, key_indices, non_key_indices)
        """
        n_emb = query_embeddings.shape[0]
        
        # Method 1: Use ColQwen tokenizer for proper alignment (preferred)
        if input_ids is not None and self.tokenizer is not None:
            decoded_tokens = self.decode_input_ids(input_ids)
            if decoded_tokens is not None:
                mask, cleaned_tokens = self.clean_tokens(decoded_tokens)
                
                if cleaned_tokens:
                    # Get POS tags aligned to embedding positions
                    pos_tags = self.get_pos_tags_aligned(cleaned_tokens, mask)
                    
                    # Split into key and non-key indices
                    key_indices = []
                    non_key_indices = []
                    
                    for i, pos_tag in enumerate(pos_tags):
                        if pos_tag is None:
                            continue
                        # Match paper: nouns (tags starting with 'N') are key tokens
                        # Also skip QUERY marker (first valid token)
                        if pos_tag.startswith('N'):
                            key_indices.append(i)
                        elif pos_tag != "QUERY":
                            non_key_indices.append(i)
                    
                    # Validate indices
                    key_indices = [i for i in key_indices if i < n_emb]
                    non_key_indices = [i for i in non_key_indices if i < n_emb]
                    
                    if key_indices or non_key_indices:
                        key_emb = query_embeddings[key_indices] if key_indices else np.zeros((0, query_embeddings.shape[1]))
                        non_key_emb = query_embeddings[non_key_indices] if non_key_indices else np.zeros((0, query_embeddings.shape[1]))
                        return key_emb, non_key_emb, key_indices, non_key_indices
        
        # Method 2: Fallback to NLTK word_tokenize (less accurate)
        tokens = self.nltk.word_tokenize(query)
        pos_tags = self.get_pos_tags(tokens)
        
        # Split into key and non-key indices
        key_indices = []
        non_key_indices = []
        
        for i, pos_tag in enumerate(pos_tags):
            # Match paper: nouns (tags starting with 'N') are key tokens
            if pos_tag.startswith('N'):
                key_indices.append(i)
            else:
                non_key_indices.append(i)
        
        # Handle embedding dimension mismatch (special tokens)
        n_tok = len(tokens)
        
        if n_emb != n_tok:
            # Assume embeddings include special tokens at start/end
            # Adjust indices with offset
            offset = (n_emb - n_tok) // 2
            key_indices = [idx + offset for idx in key_indices if 0 <= idx + offset < n_emb]
            non_key_indices = [idx + offset for idx in non_key_indices if 0 <= idx + offset < n_emb]
        
        # Extract embeddings
        key_emb = query_embeddings[key_indices] if key_indices else np.zeros((0, query_embeddings.shape[1]))
        non_key_emb = query_embeddings[non_key_indices] if non_key_indices else np.zeros((0, query_embeddings.shape[1]))
        
        return key_emb, non_key_emb, key_indices, non_key_indices
    
    def filter_tokens(
        self,
        query: str,
        query_embeddings: np.ndarray,
        input_ids: Optional['torch.Tensor'] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Filter to keep only key tokens (nouns).
        
        Compatible with QueryTokenFilter interface.
        
        Args:
            query: Query string
            query_embeddings: Multi-vector embeddings
            input_ids: Optional input_ids from ColQwen for proper alignment
            
        Returns:
            Tuple of (filtered_embeddings, kept_indices)
        """
        key_emb, _, key_indices, _ = self.split_embeddings(query, query_embeddings, input_ids)
        
        # Fallback: keep all if no key tokens found
        if len(key_indices) == 0:
            return query_embeddings, list(range(query_embeddings.shape[0]))
        
        return key_emb, key_indices

