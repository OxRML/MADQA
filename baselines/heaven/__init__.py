"""
HEAVEN: Hybrid-Vector Retrieval for Visually Rich Documents

Copied from: https://github.com/juyeonnn/HEAVEN
Paper: https://arxiv.org/pdf/2510.22215

Key components:
- DSE Encoder: Single-vector embeddings (Stage 1)
- ColQwen2.5 Encoder: Multi-vector embeddings (Stage 2)
- DLA: Document Layout Analysis for VS-Page construction
- VS-Pages: Visually-Summarized Pages for efficient Stage 1 filtering
- Query Token Filtering: POS-based noun filtering for Stage 2
"""

from .encoders import (
    BaseEncoder,
    DSEEncoder,
    ColQwenEncoder,
    SigLIPEncoder,
    get_encoder,
    compute_maxsim_score,
)

from .index import (
    DocumentIndex,
    pdf_to_images,
)

from .retrieval import (
    HEAVENRetriever,
    SimpleRetriever,
    RetrievalResult,
)

from .dla import DLA

from .heaven_utils import (
    split_query,
    score_multi_vector,
    evaluate,
    prepare_evaluate,
    get_doc_mapping,
    clean_name,
    filter_files,
)

from .query_filtering import (
    QueryTokenFilter,
    AdaptiveTokenFilter,
    POSTokenFilter,
    compute_filtered_maxsim,
    split_query_by_pos,
    STOP_WORDS,
)

from .heaven import HEAVENAgent

__all__ = [
    # Encoders
    "BaseEncoder",
    "DSEEncoder",
    "ColQwenEncoder", 
    "SigLIPEncoder",
    "get_encoder",
    "compute_maxsim_score",
    # Index
    "DocumentIndex",
    "pdf_to_images",
    # Retrieval
    "HEAVENRetriever",
    "SimpleRetriever",
    "RetrievalResult",
    # DLA
    "DLA",
    # Utils (from original HEAVEN)
    "split_query",
    "score_multi_vector",
    "evaluate",
    "prepare_evaluate",
    "get_doc_mapping",
    "clean_name",
    "filter_files",
    # Query Filtering
    "QueryTokenFilter",
    "AdaptiveTokenFilter",
    "POSTokenFilter",
    "compute_filtered_maxsim",
    "split_query_by_pos",
    "STOP_WORDS",
    # Agent
    "HEAVENAgent",
]
