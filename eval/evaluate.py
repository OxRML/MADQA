#!/usr/bin/env python3
"""
Evaluation CLI for Agentic Document AI.

Evaluates model predictions against the OxRML/MADQA benchmark.

Usage:
    python evaluate.py results.jsonl [--by-category] [--by-domain]
    python evaluate.py results_*.jsonl --compare
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from metrics import (
    anls_star, 
    anls_star_llm, 
    aggregate_anls_star_llm,
    citation_f1, 
    kuiper_statistic, 
    wasted_effort_ratio
)


def derive_hop_type(evidence: list) -> str:
    """Derive hop type from evidence list.
    
    - single: Single page from a single document
    - cross_page: Multiple pages from the same document
    - cross_doc: Pages from different documents
    
    Args:
        evidence: List of dicts with 'document' and 'page' keys
    
    Returns:
        'single', 'cross_page', or 'cross_doc'
    """
    if not evidence:
        return 'single'
    
    # Get unique documents and pages
    documents = set()
    pages = set()
    
    for ev in evidence:
        doc = ev.get('document')
        page = ev.get('page')
        if doc is not None:
            documents.add(doc)
        if doc is not None and page is not None:
            pages.add((doc, page))
    
    # Determine hop type based on evidence structure
    if len(documents) > 1:
        return 'cross_doc'  # Multiple documents
    elif len(pages) > 1:
        return 'cross_page'  # Multiple pages from same document
    else:
        return 'single'  # Single page


def load_gold_standard(dataset_name: str = "OxRML/MADQA", split: str = "dev"):
    """Load gold standard from HuggingFace dataset.
    
    Returns two mappings:
    - by_text: question text -> gold data (primary)
    - by_id: question id -> gold data (fallback)
    """
    print(f"Loading {dataset_name} ({split} split)...")
    dataset = load_dataset(dataset_name, split=split)
    
    by_text = {}
    by_id = {}
    
    for ex in dataset:
        question = ex['question'].strip()
        qid = ex.get('id', '')
        
        evidence = ex.get('evidence', [])
        
        gold_data = {
            'answers': ex.get('answer_variants', []),
            'evidence': evidence,
            'category': ex.get('document_category', ''),
            'domain': ex.get('domain', ''),
            'hop_type': derive_hop_type(evidence)
        }
        
        by_text[question] = gold_data
        if qid:
            by_id[qid] = gold_data
    
    print(f"Loaded {len(by_text)} gold examples")
    return by_text, by_id


def load_results(filepath: Path) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def evaluate_single(
    result: Dict,
    gold_by_text: Dict[str, Dict],
    gold_by_id: Dict[str, Dict],
    use_semantic: bool = False
) -> Optional[Dict[str, Any]]:
    """Evaluate a single prediction.
    
    Matches by question text first, falls back to question ID if not found.
    
    Args:
        result: Prediction dict with 'question', 'answer', 'citations'
        gold_by_text: Gold data indexed by question text
        gold_by_id: Gold data indexed by question ID
        use_semantic: If True, also compute semantic accuracy with LLM judge
    """
    question = result.get('question', '').strip()
    qid = result.get('id', '')
    
    # Try matching by question text first
    if question in gold_by_text:
        gold_data = gold_by_text[question]
    elif qid and qid in gold_by_id:
        # Fallback to ID-based matching
        gold_data = gold_by_id[qid]
    else:
        return None
    answer = result.get('answer', '')
    citations = result.get('citations', [])
    
    # ANLS*
    anls = anls_star(answer, gold_data['answers'])
    
    # Semantic accuracy with LLM judge (if enabled)
    if use_semantic:
        llm_result = anls_star_llm(answer, gold_data['answers'], question)
        semantic = llm_result['score']
        correct = semantic >= 0.5
    else:
        semantic = anls
        correct = anls >= 0.5
    
    # Citation F1
    doc_f1 = citation_f1(citations, gold_data['evidence'], level='document')
    page_f1 = citation_f1(citations, gold_data['evidence'], level='page')
    
    # Steps (for Kuiper)
    search_history = result.get('search_history', [])
    steps = len(search_history) if search_history else result.get('iterations', 0)
    
    return {
        'question': question,
        'anls': anls,
        'semantic': semantic,
        'correct': correct,
        'doc_f1': doc_f1['f1'],
        'page_f1': page_f1['f1'],
        'steps': steps,
        'category': gold_data['category'],
        'domain': gold_data['domain'],
        'hop_type': gold_data.get('hop_type', 'single')
    }


def aggregate_metrics(evals: List[Dict], use_semantic: bool = False) -> Dict[str, Any]:
    """Aggregate metrics across evaluations."""
    if not evals:
        return {}
    
    n = len(evals)
    accuracy = sum(e['correct'] for e in evals) / n
    mean_anls = sum(e['anls'] for e in evals) / n
    mean_doc_f1 = sum(e['doc_f1'] for e in evals) / n
    mean_page_f1 = sum(e['page_f1'] for e in evals) / n
    
    # Semantic accuracy with bias correction
    if use_semantic and 'semantic' in evals[0]:
        semantic_scores = [e['semantic'] for e in evals]
        agg = aggregate_anls_star_llm(semantic_scores, apply_bias_correction=True)
        mean_semantic = agg['adjusted_score']
        semantic_ci = (agg['ci_lower'], agg['ci_upper'])
    else:
        mean_semantic = mean_anls
        semantic_ci = None
    
    # Kuiper
    kuiper = kuiper_statistic(evals)
    wasted = wasted_effort_ratio(evals)
    
    return {
        'n': n,
        'accuracy': accuracy,
        'mean_anls': mean_anls,
        'mean_semantic': mean_semantic,
        'semantic_ci': semantic_ci,
        'doc_f1': mean_doc_f1,
        'page_f1': mean_page_f1,
        'kuiper_stat': kuiper['kuiper_stat'],
        'kuiper_degenerate': kuiper['degenerate'],
        'wasted_effort_ratio': wasted['ratio'],
        'mean_steps_correct': wasted['mean_steps_correct'],
        'mean_steps_incorrect': wasted['mean_steps_incorrect'],
    }


def print_metrics(name: str, metrics: Dict, indent: int = 0, use_semantic: bool = False):
    """Print metrics in a formatted way."""
    prefix = "  " * indent
    
    if 'n' not in metrics:
        print(f"{prefix}{name}: No data")
        return
    
    print(f"{prefix}{name} (n={metrics['n']}):")
    
    if use_semantic and 'mean_semantic' in metrics:
        ci = metrics.get('semantic_ci')
        ci_str = f" [{ci[0]:.2%}-{ci[1]:.2%}]" if ci else ""
        print(f"{prefix}  Semantic Accuracy:    {metrics['mean_semantic']:.2%}{ci_str}")
        print(f"{prefix}  ANLS* (string):       {metrics['mean_anls']:.4f}")
    else:
        print(f"{prefix}  Accuracy (ANLS*>=0.5): {metrics['accuracy']:.1%}")
        print(f"{prefix}  Mean ANLS*:           {metrics['mean_anls']:.4f}")
    
    print(f"{prefix}  Document F1:          {metrics['doc_f1']:.4f}")
    print(f"{prefix}  Page F1:              {metrics['page_f1']:.4f}")
    
    if not metrics.get('kuiper_degenerate'):
        print(f"{prefix}  Kuiper Statistic:     {metrics['kuiper_stat']:.2f}")
    
    if metrics.get('wasted_effort_ratio', 0) < float('inf'):
        print(f"{prefix}  Wasted Effort Ratio:  {metrics['wasted_effort_ratio']:.3f}")


def evaluate_file(
    filepath: Path,
    gold_by_text: Dict[str, Dict],
    gold_by_id: Dict[str, Dict],
    by_category: bool = False,
    by_domain: bool = False,
    by_hop_type: bool = True,
    use_semantic: bool = False
) -> Dict[str, Any]:
    """Evaluate a single results file."""
    results = load_results(filepath)
    
    evals = []
    unmatched = 0
    total = len(results)
    
    for i, result in enumerate(results):
        if use_semantic and (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{total}...")
        ev = evaluate_single(result, gold_by_text, gold_by_id, use_semantic=use_semantic)
        if ev:
            evals.append(ev)
        else:
            unmatched += 1
    
    if unmatched > 0:
        print(f"  Warning: {unmatched} questions not found in gold standard")
    
    # Overall metrics
    overall = aggregate_metrics(evals, use_semantic=use_semantic)
    
    output = {'overall': overall, 'use_semantic': use_semantic}
    
    # By hop type (always included by default)
    if by_hop_type:
        by_hop = defaultdict(list)
        for e in evals:
            by_hop[e.get('hop_type', 'single')].append(e)
        output['by_hop_type'] = {hop: aggregate_metrics(items, use_semantic) for hop, items in sorted(by_hop.items())}
    
    # By category
    if by_category:
        by_cat = defaultdict(list)
        for e in evals:
            by_cat[e['category'] or 'Unknown'].append(e)
        output['by_category'] = {cat: aggregate_metrics(items, use_semantic) for cat, items in sorted(by_cat.items())}
    
    # By domain
    if by_domain:
        by_dom = defaultdict(list)
        for e in evals:
            by_dom[e['domain'] or 'Other'].append(e)
        output['by_domain'] = {dom: aggregate_metrics(items, use_semantic) for dom, items in sorted(by_dom.items())}
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions on Agentic Document AI benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py results.jsonl
  python evaluate.py results.jsonl --by-category --by-domain
  python evaluate.py model1.jsonl model2.jsonl --compare
        """
    )
    parser.add_argument('files', nargs='+', type=Path, help='Result JSONL file(s)')
    parser.add_argument('--dataset', default='OxRML/MADQA',
                        help='HuggingFace dataset name')
    parser.add_argument('--split', default='dev', help='Dataset split to evaluate on')
    parser.add_argument('--by-category', action='store_true', help='Show metrics by document category')
    parser.add_argument('--by-domain', action='store_true', help='Show metrics by domain')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models side-by-side')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--semantic', action='store_true', 
                        help='Use semantic accuracy (ANLS* + LLM judge) instead of pure ANLS*. Requires GOOGLE_API_KEY.')
    
    args = parser.parse_args()
    
    # Load gold standard
    gold_by_text, gold_by_id = load_gold_standard(args.dataset, args.split)
    
    if not gold_by_text:
        print("Error: No gold standard data loaded", file=sys.stderr)
        sys.exit(1)
    
    all_results = {}
    
    for filepath in args.files:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            continue
        
        # Extract model name
        name = filepath.stem
        if name.startswith("results_"):
            name = name[8:]
        if name.endswith("_results"):
            name = name[:-8]
        
        print(f"\nEvaluating: {filepath.name}")
        if args.semantic:
            print("  Using semantic accuracy (ANLS* + LLM judge)...")
        result = evaluate_file(
            filepath, gold_by_text, gold_by_id, 
            args.by_category, args.by_domain, 
            use_semantic=args.semantic
        )
        all_results[name] = result
    
    # Output
    if args.json:
        # Convert for JSON serialization
        def sanitize(obj):
            if isinstance(obj, float) and (obj != obj or obj == float('inf')):  # NaN or inf
                return None
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj
        
        print(json.dumps(sanitize(all_results), indent=2))
    else:
        # Print formatted output
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        if args.compare and len(all_results) > 1:
            # Comparison table
            models = list(all_results.keys())
            
            if args.semantic:
                print(f"\n{'Model':<35} {'Semantic':<10} {'ANLS*':<8} {'Doc F1':<8} {'Page F1':<8} {'Kuiper':<8}")
                print("-" * 85)
                
                for model in sorted(models, key=lambda m: -all_results[m]['overall'].get('mean_semantic', 0)):
                    m = all_results[model]['overall']
                    kuiper_str = f"{m['kuiper_stat']:.2f}" if not m.get('kuiper_degenerate') else "N/A"
                    print(f"{model:<35} {m.get('mean_semantic', 0):.1%}      {m.get('mean_anls', 0):.4f}  "
                          f"{m.get('doc_f1', 0):.4f}  {m.get('page_f1', 0):.4f}  {kuiper_str}")
            else:
                print(f"\n{'Model':<35} {'Acc':<8} {'ANLS*':<8} {'Doc F1':<8} {'Page F1':<8} {'Kuiper':<8}")
                print("-" * 75)
                
                for model in sorted(models, key=lambda m: -all_results[m]['overall'].get('accuracy', 0)):
                    m = all_results[model]['overall']
                    kuiper_str = f"{m['kuiper_stat']:.2f}" if not m.get('kuiper_degenerate') else "N/A"
                    print(f"{model:<35} {m.get('accuracy', 0):.1%}    {m.get('mean_anls', 0):.4f}  "
                          f"{m.get('doc_f1', 0):.4f}  {m.get('page_f1', 0):.4f}  {kuiper_str}")
        else:
            # Detailed per-model output
            for model, result in all_results.items():
                print(f"\n{'─' * 40}")
                use_sem = result.get('use_semantic', False)
                print_metrics(model, result['overall'], use_semantic=use_sem)
                
                if 'by_hop_type' in result:
                    print(f"\n  By Hop Type:")
                    for hop, metrics in sorted(result['by_hop_type'].items(),
                                               key=lambda x: -x[1].get('n', 0)):
                        print_metrics(hop, metrics, indent=2, use_semantic=use_sem)
                
                if 'by_category' in result:
                    print(f"\n  By Category:")
                    for cat, metrics in sorted(result['by_category'].items(), 
                                              key=lambda x: -x[1].get('n', 0)):
                        print_metrics(cat, metrics, indent=2, use_semantic=use_sem)
                
                if 'by_domain' in result:
                    print(f"\n  By Domain:")
                    for dom, metrics in sorted(result['by_domain'].items(),
                                              key=lambda x: -x[1].get('n', 0)):
                        print_metrics(dom, metrics, indent=2, use_semantic=use_sem)
    
    print()


if __name__ == "__main__":
    main()
