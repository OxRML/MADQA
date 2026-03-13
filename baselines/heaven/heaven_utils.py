#!/usr/bin/env python3
"""
HEAVEN Utility Functions.

Copied directly from: https://github.com/juyeonnn/HEAVEN/blob/main/retrieval/heaven/utils.py
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import numpy as np
from typing import Union, List, Optional, Dict
import warnings
warnings.filterwarnings("ignore")


def get_split_key(folder: str) -> str:
    """
    Get the split key for a dataset to extract document names from page names.

    Returns:
        Split key string or None
    """
    return '_'


def clean_name(fname: str) -> str:
    """
    Remove image extensions from filename.

    Returns:
        Filename without extensions
    """
    return fname.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')


def filter_files(files: List[str]) -> List[str]:
    """Filter files to only include supported image formats."""
    return [f for f in files if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg')]


def get_doc_name(page_name: str, split_key: str = '_') -> str:
    """
    Extract document name from page name using split key.
    
    Returns:
        Document name
    """
    if split_key is None:
        return page_name
    else:
        parts = page_name.split(split_key)
        return split_key.join(parts[:-1])


def split_page_name(page_name: str, split_key: str = '_') -> tuple:
    """
    Split page name into document name and page number.
    
    Returns:
        Tuple of (document_name, page_number)
    """
    if split_key is None:
        return page_name, 0
    else:
        parts = page_name.split(split_key)
        num = clean_name(parts[-1])

        return split_key.join(parts[:-1]), int(num)


def get_doc_mapping(mapping: Dict[str, int], split_key: str = '_') -> Dict[str, List[int]]:
    """
    Create document to page indices mapping.
    
    Returns:
        Dictionary mapping document names to lists of page indices
    """
    doc_mapping = {}
    page_count = 0
    
    for page_name, idx in mapping.items():
        doc_name = get_doc_name(page_name, split_key)
        
        if doc_name not in doc_mapping:
            doc_mapping[doc_name] = []
        
        doc_mapping[doc_name].append(idx)
        page_count += 1
    
    # Sort indices for each document
    for doc_name in doc_mapping:
        doc_mapping[doc_name] = sorted(doc_mapping[doc_name])
    
    print(f"Documents: {len(doc_mapping)}\tPages: {page_count}")
    return doc_mapping


def get_doc_mapping_layout(pages: List[str], split_key: str = '_') -> Dict[str, List[str]]:
    """
    Create document to page names mapping (for VS-page construction).
    
    Returns:
        Dictionary mapping document names to lists of page names
    """
    doc_mapping = {}
    
    for page_name in pages:
        doc_name = get_doc_name(clean_name(page_name), split_key)
        
        if doc_name not in doc_mapping:
            doc_mapping[doc_name] = []
        
        doc_mapping[doc_name].append(page_name)
    
    # Sort pages by page number
    for doc_name in doc_mapping:
        doc_mapping[doc_name] = sorted(
            doc_mapping[doc_name],
            key=lambda x: split_page_name(clean_name(x), split_key)[1]
        )
    
    return doc_mapping


def sort_doc_by_page(pages: List[str], page_nums: List[int]) -> List[str]:
    """Sort pages by page number."""
    sorted_pairs = sorted(zip(pages, page_nums), key=lambda x: x[1])
    return [p for p, _ in sorted_pairs]


def prepare_data(
    data: List[Dict],
    split_key: str = '_'
) -> List[List[str]]:
    """
    Prepare ground truth document names from test data.
    
    Returns:
        List of lists of ground truth document names per query
    """
    gt_doc_names = []
    for item in tqdm(data, desc='Preparing dataset'):
        gt_doc_names.append([get_doc_name(doc, split_key) for doc in item['doc_ids']])
    return gt_doc_names


def load_embedding(path: str, device: Union[str, torch.device]) -> torch.Tensor:
    """
    Load embeddings from file and normalize.
    
    Returns:
        Normalized embedding tensor
    """
    embedding = torch.load(path, map_location=device, weights_only=False)
    
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding)
    else:
        embedding = embedding.float()
    
    print(f"{path}:\t{embedding.shape}")
    
    # Normalize
    embedding = F.normalize(embedding, dim=-1)
    return embedding.to(device)


def merge_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Merge multiple embedding tensors
    
    Returns:
        Merged and normalized tensor
    """
    if tensors[0].dim() == 3:
        # Multi-vector embeddings: pad to max length
        max_len = max([t.shape[1] for t in tensors])
        padded_tensors = []
        
        for t in tensors:
            if t.shape[1] < max_len:
                # Pad sequence dimension
                padded_tensors.append(F.pad(t, (0, 0, 0, max_len - t.shape[1])))
            else:
                padded_tensors.append(t)
        
        merged = torch.cat(padded_tensors, dim=0)
    else:
        # Single-vector embeddings
        merged = torch.cat(tensors, dim=0)
    
    # Normalize
    return F.normalize(merged, dim=-1)


def score_multi_vector(
    qs: Union[torch.Tensor, List[torch.Tensor]],
    ps: Union[torch.Tensor, List[torch.Tensor]],
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Compute multi-vector similarity scores (MaxSim + Sum, ColBERT-like).
    
    Copied from: https://github.com/illuin-tech/colpali/blob/main/colpali_engine/utils/processing_utils.py
    
    Args:
        qs: Query embeddings of shape (n_queries, query_length, embed_dim)
        ps: Passage embeddings of shape (n_passages, passage_length, embed_dim)
        batch_size: Batch size for computation
        device: Device for computation
        
    Returns:
        Score tensor of shape (n_queries, n_passages)
    """
    scores_list: List[torch.Tensor] = []
    
    total_batches = 1 + int(len(qs) / batch_size)
    
    for i in tqdm(range(0, len(qs), batch_size), desc="Scoring", total=total_batches):
        scores_batch = []
        qs_batch = qs[i : i + batch_size].to(device)
        
        for j in range(0, len(ps), batch_size):
            ps_batch = ps[j : j + batch_size].to(device)
            
            # Compute: max over passage tokens, then sum over query tokens
            # einsum: b=batch_q, n=query_len, d=dim, c=batch_p, s=passage_len
            batch_scores = torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
            batch_scores = batch_scores.max(dim=3)[0].sum(dim=2)
            scores_batch.append(batch_scores)
        
        scores_batch = torch.cat(scores_batch, dim=1).cpu()
        scores_list.append(scores_batch)
    
    scores = torch.cat(scores_list, dim=0)
    return scores


def evaluate(
    doc_scores: torch.Tensor,
    gt_doc_ids: List[List[int]],
    k_values: List[int] = [1, 3, 5, 10, 20, 50, 100, 200, 500]
) -> Dict[str, float]:
    """
    Evaluate retrieval using Mean Reciprocal Rank (MRR) and Recall@K.
        
    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    recall = {k: [] for k in k_values}
    mrr = []
    
    sort_val, sort_idx = torch.sort(doc_scores, dim=1, descending=True)
    
    for i in range(len(gt_doc_ids)):
        # Find ranks of ground truth documents
        rank = [torch.where(sort_idx[i] == gt)[0].item() for gt in gt_doc_ids[i]]
        
        # MRR: reciprocal of the best (minimum) rank
        min_rank = min(rank)
        mrr.append(1.0 / (min_rank + 1))
        
        # Recall@K
        for k in k_values:
            recall[k].append(np.mean([1 if r < k else 0 for r in rank]))
    
    # Compile results
    results = {'mrr': np.mean(mrr)}
    
    print(f"MRR: {results['mrr']:.4f}", end="\t")
    for k in k_values:
        results[f'recall@{k}'] = np.mean(recall[k])
        print(f"Recall@{k}: {100*results[f'recall@{k}']:.2f}", end="\t")
    print()
    
    return results


def get_max(doc_mapping: Dict[str, List[int]]) -> int:
    """
    Get the maximum index from document mapping.
    
    Returns:
        Maximum index value
    """
    max_val = 0
    for indices in doc_mapping.values():
        if indices and max(indices) > max_val:
            max_val = max(indices)
    return max_val


def check_dict(doc_mapping: Dict[str, List[int]]) -> None:
    """
    Check for duplicate values in document mapping.
    """
    values = []
    for v in doc_mapping.values():
        values.extend(v)
    
    if len(values) != len(set(values)):
        print(f"Warning: Found duplicate values!")
        print(f"Total values: {len(values)}, Unique: {len(set(values))}")
    else:
        print(f"No duplicate indices found")


def split_query(pos_tag: List[str]) -> tuple:
    """
    Split query tokens into key tokens (nouns) and non-key tokens (others).
    
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


def prepare_evaluate(
    scores: torch.Tensor,
    doc_mapping: Dict[str, List[int]],
    gt_doc_names: List[List[str]]
) -> tuple:
    """
    Prepare scores for evaluation by aggregating to document level.
    
    Returns:
        Tuple of (doc_scores, gt_doc_ids)
    """
    doc_score_mapping = {}
    doc_scores = []
    
    for mapping_idx, (doc, idx) in enumerate(doc_mapping.items()):
        doc_score_mapping[doc] = mapping_idx
        if len(idx) == 1:
            max_val = scores[:, idx[0]]
        else:
            max_val, _ = scores[:, idx].max(dim=1)
        doc_scores.append(max_val)
    
    doc_scores = torch.stack(doc_scores, dim=0).T
    gt_doc_ids = [[doc_score_mapping[i] for i in item] for item in gt_doc_names]
    
    return doc_scores, gt_doc_ids


def filter_doc(doc_scores: torch.Tensor, k: int = 50) -> tuple:
    """
    Filter documents by keeping only top-k scores per query.
    
    Returns:
        Tuple of (mask, ratio) where mask indicates kept documents
    """
    # Get top-k indices
    _, sort_idx = torch.topk(doc_scores, k=min(k, doc_scores.shape[1]), dim=1)
    
    # Create mask using scatter
    mask = torch.zeros_like(doc_scores, dtype=torch.bool)
    mask.scatter_(1, sort_idx, True)
    
    print(f"Created mask for top-{k} documents")
    used_doc = mask.sum().item()
    total_docs = mask.shape[0] * mask.shape[1]
    print(f"Masked: {used_doc} / {total_docs} = {used_doc/total_docs * 100:.2f}%")
    
    avg_docs_per_query = used_doc / mask.shape[0]
    return mask, avg_docs_per_query / mask.shape[1]

