"""
Core evaluation metrics for document QA.

Metrics:
- ANLS*: Answer-level Normalized Levenshtein Similarity
- ANLS*+LLM: ANLS* with LLM fallback for semantic equivalence
- Citation F1: Document-level and Page-level F1 scores
- Kuiper Statistic: Effort-accuracy calibration measure

Bias Correction:
Based on "How to Correctly Report LLM-as-a-Judge Evaluations" (2511.21140v2)
"""

import json
import os
import time
from math import sqrt
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as np
except ImportError:  # Optional dependency for Kuiper statistic
    np = None

try:
    from scipy.stats import norm
except ImportError:  # Optional dependency for confidence interval
    norm = None

from anls_star import anls_score


# ============================================================================
# LLM Judge Calibration (from human evaluation)
# ============================================================================

# Calibration values from 200-sample human evaluation
# Sensitivity: P(LLM=correct | Human=correct)
LLM_JUDGE_SENSITIVITY = 0.980  # q1
# Specificity: P(LLM=incorrect | Human=incorrect)  
LLM_JUDGE_SPECIFICITY = 1.000  # q0
# Calibration sample sizes (for confidence intervals)
LLM_JUDGE_CALIBRATION_M1 = 152  # samples where human=correct
LLM_JUDGE_CALIBRATION_M0 = 48   # samples where human=incorrect


def bias_adjusted_score(
    raw_score: float,
    q0: float = LLM_JUDGE_SPECIFICITY,
    q1: float = LLM_JUDGE_SENSITIVITY
) -> float:
    """
    Compute bias-adjusted score using Rogan-Gladen correction.
    
    From "How to Correctly Report LLM-as-a-Judge Evaluations":
    θ̂ = (p̂ + q₀ - 1) / (q₀ + q₁ - 1)
    
    Args:
        raw_score: Raw LLM judgment score (p̂)
        q0: Specificity - P(LLM=incorrect | true=incorrect)
        q1: Sensitivity - P(LLM=correct | true=correct)
    
    Returns:
        Bias-adjusted score, clipped to [0, 1]
    """
    if q0 + q1 <= 1:
        # Degenerate case - judge is no better than random
        return raw_score
    
    adjusted = (raw_score + q0 - 1) / (q0 + q1 - 1)
    return max(0.0, min(1.0, adjusted))


def standard_error(
    raw_score: float,
    n_samples: int,
    q0: float = LLM_JUDGE_SPECIFICITY,
    q1: float = LLM_JUDGE_SENSITIVITY
) -> float:
    """
    Compute bias-adjusted standard error.
    
    SE is scaled by the bias adjustment factor to account for
    the transformation from raw to adjusted score.
    
    Args:
        raw_score: Raw LLM judgment score (p̂)
        n_samples: Number of test samples
        q0: Specificity
        q1: Sensitivity
    
    Returns:
        Bias-adjusted standard error
    """
    if n_samples <= 0 or q0 + q1 <= 1:
        return 0.0
    
    # Raw binomial SE
    p = raw_score
    se_raw = sqrt(p * (1 - p) / n_samples) if 0 < p < 1 else 0
    
    # Scale by bias adjustment factor
    se_adjusted = se_raw / (q0 + q1 - 1)
    
    return se_adjusted


def confidence_interval(
    raw_score: float,
    n_samples: int,
    q0: float = LLM_JUDGE_SPECIFICITY,
    q1: float = LLM_JUDGE_SENSITIVITY,
    m0: int = LLM_JUDGE_CALIBRATION_M0,
    m1: int = LLM_JUDGE_CALIBRATION_M1,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute confidence interval for bias-adjusted score.
    
    Simplified version that uses observed q0, q1 directly when calibration
    is high quality (q0 + q1 > 1.9). Falls back to full formula otherwise.
    
    Args:
        raw_score: Raw LLM judgment score (p̂)
        n_samples: Number of test samples
        q0: Specificity
        q1: Sensitivity
        m0: Calibration samples where human=incorrect
        m1: Calibration samples where human=correct
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if norm is not None:
        z = norm.ppf(1 - alpha / 2)
    else:
        # Fallback for common alpha values if scipy is unavailable
        if abs(alpha - 0.10) < 1e-8:
            z = 1.645
        elif abs(alpha - 0.05) < 1e-8:
            z = 1.96
        else:
            z = 1.96
    
    # For high-quality calibration (q0 + q1 > 1.9), use simplified CI
    # that trusts the observed sensitivity/specificity
    if q0 + q1 > 1.9:
        # Bias-adjusted point estimate
        theta = bias_adjusted_score(raw_score, q0, q1)
        
        # Simple binomial SE for the test dataset only
        # (calibration is trusted to be accurate)
        p = raw_score
        se_raw = sqrt(p * (1 - p) / n_samples) if n_samples > 0 else 0
        
        # Scale SE by the bias adjustment factor
        se_adjusted = se_raw / (q0 + q1 - 1)
        
        lower = max(0.0, theta - z * se_adjusted)
        upper = min(1.0, theta + z * se_adjusted)
        return (lower, upper)
    
    # Full formula with regularization for lower-quality calibration
    p = (n_samples * raw_score + z**2 / 2) / (n_samples + z**2)
    q0_adj = (m0 * q0 + 1) / (m0 + 2)
    q1_adj = (m1 * q1 + 1) / (m1 + 2)
    
    n_adj = n_samples + z**2
    m0_adj = m0 + 2
    m1_adj = m1 + 2
    
    # Point estimate
    if q0_adj + q1_adj <= 1:
        return (0.0, 1.0)
    
    theta = (p + q0_adj - 1) / (q0_adj + q1_adj - 1)
    
    # Bias correction term
    dth = 2 * z**2 * (
        -(1 - theta) * q0_adj * (1 - q0_adj) / m0_adj 
        + theta * q1_adj * (1 - q1_adj) / m1_adj
    )
    
    # Standard error
    se = sqrt(
        p * (1 - p) / n_adj 
        + (1 - theta)**2 * q0_adj * (1 - q0_adj) / m0_adj 
        + theta**2 * q1_adj * (1 - q1_adj) / m1_adj
    ) / (q0_adj + q1_adj - 1)
    
    lower = max(0.0, theta + dth - z * se)
    upper = min(1.0, theta + dth + z * se)
    
    return (lower, upper)


def anls_star(predicted: Any, ground_truths: List[List[str]]) -> float:
    """
    Calculate ANLS* score (case-insensitive).
    
    Args:
        predicted: Predicted answer (string or list)
        ground_truths: List of answer variants, each variant is a list of strings
    
    Returns:
        Maximum ANLS* score across all variants (0.0 to 1.0)
    """
    if not ground_truths:
        return 0.0

    if predicted is None:
        predicted = []
    
    if isinstance(predicted, str):
        predicted = [predicted]
    
    if not predicted:
        return 0.0
    
    # Convert all elements to lowercase strings
    pred_lower = [str(p).lower() for p in predicted]
    
    max_score = 0.0
    for gold_variant in ground_truths:
        if isinstance(gold_variant, str):
            gold_variant = [gold_variant]
        gold_lower = [g.lower() if isinstance(g, str) else str(g).lower() for g in gold_variant]
        score = anls_score(pred_lower, gold_lower)
        max_score = max(max_score, score)
    
    return max_score


# ============================================================================
# ANLS* + LLM Judge Metric
# ============================================================================

_GEVAL_PROMPT_TEMPLATE = """You are evaluating answer correctness for a Document QA benchmark.

## Input
Question: {question}
Predicted Answer: {predicted}
Gold Answer Variants: {gold_variants}

## Evaluation Criteria

**correct**: Predicted answer is semantically equivalent to at least one gold variant. Minor format differences are acceptable.

**partial**: Predicted answer contains correct core information but has a significant format issue (e.g., list presented as comma-separated string when items are short/atomic) OR includes irrelevant additions.

**incorrect**: Predicted answer is factually wrong, missing, contains different information, or fails to answer the question type (e.g., no Yes/No for binary questions). Missing unit qualifiers that change magnitude (thousands, millions) are incorrect.

## Evaluation Steps

Follow these steps in order:

Step 1 - Check for refusal: Does the answer refuse or claim inability to answer? If yes → incorrect.

Step 2 - Compare content: Does the predicted answer match the core meaning of any gold variant? If content is wrong or different → incorrect.

Step 3 - Check critical errors (any of these → incorrect):
- Missing scale qualifiers that change magnitude: "50" vs "$50 million" → incorrect
- Binary questions without explicit Yes/No: Q: "Is X true?" A: "X is observed" → incorrect (must say Yes or No)
- Wrong entity/value: different person, company, number than gold → incorrect
- Partial list with wrong items mixed in: some correct + some wrong items → incorrect

Step 4 - Check format (only if content is correct):
- If gold expects multiple items AND predicted is a comma-separated string (not a list) → partial
- If gold expects single item → no format issue possible

Step 5 - Check verbosity (only if content is correct):
- CORRECT (acceptable verbosity):
  * Extra qualifiers: "three security questions" when gold is "3" → correct
  * Relevant context: "No — Massachusetts; Washington" for "same state?" question → correct
  * Clarifying phrases: "in his personal capacity", "per annum" → correct
- PARTIAL (medium verbosity) - ONLY when additions are truly irrelevant:
  * Adding unrequested details to list items
  * Over-specific precision: date+time when only date asked → partial
- INCORRECT (high verbosity):
  * Multi-sentence responses when a word/phrase suffices
  * Full paragraphs of explanation
  * Conversational preambles: "Based on the document...", "The answer is..."

Based on your step-by-step analysis, provide your final judgment.

After your reasoning, you MUST call submit_judgment with your final decision."""


_LLM_JUDGE_TOOL = {
    "function_declarations": [{
        "name": "submit_judgment",
        "description": "Submit your final judgment after reasoning through the evaluation steps",
        "parameters": {
            "type": "object",
            "properties": {
                "judgment": {
                    "type": "string",
                    "enum": ["correct", "partial", "incorrect"],
                    "description": "Final judgment: correct, partial, or incorrect"
                },
                "main_issue": {
                    "type": "string",
                    "enum": ["none", "refusal", "wrong_content", "missing_unit", "no_yes_no", "list_format", "verbosity_medium", "verbosity_high"],
                    "description": "The primary issue found, if any"
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of your judgment"
                }
            },
            "required": ["judgment", "main_issue", "explanation"]
        }
    }]
}


def _get_gemini_model():
    """Initialize Gemini model (lazy loading)."""
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')


def _call_gemini_with_timeout(model, prompt, timeout=30):
    """Call Gemini with a timeout using threading."""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    try:
        temperature = float(os.environ.get("GEMINI_TEMPERATURE", "0"))
    except ValueError:
        temperature = 0.0
    
    def _call():
        return model.generate_content(
            prompt,
            tools=[_LLM_JUDGE_TOOL],
            tool_config={"function_calling_config": {"mode": "ANY"}},
            generation_config={"temperature": temperature},
            request_options={"timeout": timeout},
        )
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            raise TimeoutError(f"Gemini API call timed out after {timeout}s")


def _call_llm_judge(
    question: str,
    predicted: Any,
    gold_variants: List[List[str]],
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Call Gemini LLM judge with retries and timeout.
    
    Returns:
        Dict with 'judgment', 'main_issue', 'explanation', 'score'
    """
    prompt = _GEVAL_PROMPT_TEMPLATE.format(
        question=question,
        predicted=json.dumps(predicted),
        gold_variants=json.dumps(gold_variants)
    )
    
    model = _get_gemini_model()
    
    for attempt in range(max_retries):
        try:
            response = _call_gemini_with_timeout(model, prompt, timeout=timeout)
            
            # Extract function call result
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call.name == "submit_judgment":
                        args = dict(part.function_call.args)
                        judgment = args.get('judgment', 'incorrect')
                        
                        # Map judgment to score
                        score_map = {'correct': 1.0, 'partial': 0.5, 'incorrect': 0.0}
                        args['score'] = score_map.get(judgment, 0.0)
                        return args
            
            # No function call found - retry
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
                
        except TimeoutError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return {
                'judgment': 'error',
                'main_issue': 'timeout',
                'explanation': str(e),
                'score': 0.0
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            return {
                'judgment': 'error',
                'main_issue': 'error',
                'explanation': str(e),
                'score': 0.0
            }
    
    return {
        'judgment': 'error',
        'main_issue': 'parse_error',
        'explanation': 'Failed to get valid response after retries',
        'score': 0.0
    }


def anls_star_llm(
    predicted: Any,
    ground_truths: List[List[str]],
    question: str = "",
    threshold: float = 1.0
) -> Dict[str, Any]:
    """
    ANLS* with LLM fallback for semantic equivalence checking.
    
    If ANLS* >= threshold (default 1.0), returns ANLS* score.
    Otherwise, calls Gemini LLM judge to evaluate semantic correctness.
    
    Args:
        predicted: Predicted answer (string or list)
        ground_truths: List of answer variants
        question: The question text (needed for LLM judge)
        threshold: ANLS* threshold above which to skip LLM (default 1.0)
    
    Returns:
        Dict with:
        - 'score': Final score (0.0, 0.5, or 1.0)
        - 'anls_score': Raw ANLS* score
        - 'used_llm': Whether LLM judge was called
        - 'llm_judgment': LLM judgment details (if used)
    """
    # Check for empty prediction (optimization: skip LLM, return 0)
    is_empty = (
        predicted is None 
        or predicted == "" 
        or predicted == [] 
        or (isinstance(predicted, list) and all(not p for p in predicted))
    )
    
    if is_empty:
        return {
            'score': 0.0,
            'anls_score': 0.0,
            'used_llm': False,
            'llm_judgment': {'judgment': 'incorrect', 'main_issue': 'empty', 'explanation': 'Empty prediction'}
        }
    
    # Check for overly long answers (optimization: skip LLM, return 0)
    MAX_ANSWER_LENGTH = 2000
    try:
        answer_length = len(json.dumps(predicted))
    except (TypeError, ValueError):
        answer_length = len(str(predicted))
    
    if answer_length > MAX_ANSWER_LENGTH:
        return {
            'score': 0.0,
            'anls_score': 0.0,
            'used_llm': False,
            'llm_judgment': {
                'judgment': 'incorrect', 
                'main_issue': 'too_long', 
                'explanation': f'Answer too long ({answer_length} chars > {MAX_ANSWER_LENGTH})'
            }
        }
    
    # Check ANLS*
    anls = anls_star(predicted, ground_truths)
    
    result = {
        'score': anls,
        'anls_score': anls,
        'used_llm': False,
        'llm_judgment': None
    }
    
    # If ANLS* is perfect, no need for LLM
    if anls >= threshold:
        result['score'] = 1.0
        return result
    
    # Call LLM judge for cases where ANLS* < threshold
    if question:
        llm_result = _call_llm_judge(question, predicted, ground_truths)
        result['used_llm'] = True
        result['llm_judgment'] = llm_result
        result['score'] = llm_result.get('score', 0.0)
    
    return result


def aggregate_anls_star_llm(
    scores: List[float],
    apply_bias_correction: bool = True
) -> Dict[str, Any]:
    """
    Compute aggregate ANLS*+LLM score with optional bias correction.
    
    Based on "How to Correctly Report LLM-as-a-Judge Evaluations" (2511.21140v2).
    
    Args:
        scores: List of individual ANLS*+LLM scores (0.0, 0.5, or 1.0)
        apply_bias_correction: Whether to apply Rogan-Gladen correction
    
    Returns:
        Dict with:
        - 'raw_score': Mean of raw scores
        - 'adjusted_score': Bias-adjusted score (if correction applied)
        - 'se': Bias-adjusted standard error
        - 'ci_lower': 95% CI lower bound
        - 'ci_upper': 95% CI upper bound
        - 'n_samples': Number of samples
        - 'q0': Specificity used
        - 'q1': Sensitivity used
    """
    if not scores:
        return {
            'raw_score': 0.0,
            'adjusted_score': 0.0,
            'se': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'n_samples': 0,
            'q0': LLM_JUDGE_SPECIFICITY,
            'q1': LLM_JUDGE_SENSITIVITY
        }
    
    n = len(scores)
    raw = sum(scores) / n
    
    result = {
        'raw_score': raw,
        'n_samples': n,
        'q0': LLM_JUDGE_SPECIFICITY,
        'q1': LLM_JUDGE_SENSITIVITY
    }
    
    if apply_bias_correction:
        result['adjusted_score'] = bias_adjusted_score(raw)
        result['se'] = standard_error(raw, n)
        ci = confidence_interval(raw, n)
        result['ci_lower'] = ci[0]
        result['ci_upper'] = ci[1]
    else:
        result['adjusted_score'] = raw
        result['se'] = sqrt(raw * (1 - raw) / n) if n > 0 and 0 < raw < 1 else 0.0
        # Simple binomial CI without calibration correction
        se = sqrt(raw * (1 - raw) / n) if n > 0 else 0
        z = 1.96
        result['ci_lower'] = max(0.0, raw - z * se)
        result['ci_upper'] = min(1.0, raw + z * se)
    
    return result


def citation_f1(
    predicted_citations: List[Dict[str, Any]],
    gold_locations: List[Dict[str, Any]],
    level: str = 'page'
) -> Dict[str, float]:
    """
    Calculate Citation F1 at document or page level.
    
    Args:
        predicted_citations: List of dicts with 'file'/'document' and 'page' keys
        gold_locations: List of dicts with 'document' and 'page' keys
        level: 'document' or 'page'
    
    Returns:
        Dict with 'precision', 'recall', 'f1', 'support'
    """
    if not gold_locations:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
    
    # Extract gold citations
    if level == 'document':
        gt_set: Set = {loc.get('document') for loc in gold_locations if loc.get('document')}
    else:
        gt_set = {
            (loc.get('document'), loc.get('page')) 
            for loc in gold_locations 
            if loc.get('document') is not None
        }
    
    # Extract predicted citations
    if not predicted_citations:
        pred_set: Set = set()
    else:
        if level == 'document':
            pred_set = {
                cite.get('file') or cite.get('document') 
                for cite in predicted_citations 
                if (cite.get('file') or cite.get('document'))
            }
        else:
            pred_set = {
                (cite.get('file') or cite.get('document'), cite.get('page')) 
                for cite in predicted_citations 
                if (cite.get('file') or cite.get('document')) is not None
            }
    
    # Clean None values
    gt_set = {c for c in gt_set if c is not None and (not isinstance(c, tuple) or None not in c)}
    pred_set = {c for c in pred_set if c is not None and (not isinstance(c, tuple) or None not in c)}
    
    if not gt_set:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0}
    
    tp = len(gt_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'support': len(gt_set)}


def _get_effort_value(result: Dict) -> float:
    """Extract effort value with fallbacks.
    
    Priority: steps -> llm_calls -> effort
    """
    for key in ("steps", "llm_calls", "effort"):
        value = result.get(key)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if value_float > 0:
            return value_float
    return 0.0


def kuiper_statistic(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute Kuiper calibration statistic for effort-accuracy analysis.
    
    Measures dependency between effort (steps) and accuracy. Lower values
    indicate more uniform error distribution across effort levels.
    
    Args:
        results: List of dicts with effort fields and 'correct' (bool).
                 Effort falls back from 'steps' -> 'llm_calls' -> 'effort'.
    
    Returns:
        Dict with:
        - kuiper_stat: The Kuiper statistic (lower = better calibration)
        - y_bar: Global mean accuracy
        - max_positive: Maximum positive deviation
        - max_negative: Maximum negative deviation
        - n_samples: Number of valid samples
        - degenerate: True if all samples have same correctness
    """
    if np is None:
        raise ImportError("numpy is required for kuiper_statistic; please install numpy")
    valid = [r for r in results if _get_effort_value(r) > 0]
    
    if not valid:
        return {
            'kuiper_stat': float('nan'),
            'y_bar': 0.0,
            'max_positive': 0.0,
            'max_negative': 0.0,
            'n_samples': 0,
            'degenerate': True
        }
    
    # Sort by effort (steps -> llm_calls -> effort)
    sorted_results = sorted(valid, key=_get_effort_value)
    correctness = [1 if r['correct'] else 0 for r in sorted_results]
    
    y_bar = np.mean(correctness)
    
    # Degenerate case: all same (0% or 100% accuracy)
    if y_bar == 0.0 or y_bar == 1.0:
        return {
            'kuiper_stat': float('nan'),
            'y_bar': float(y_bar),
            'max_positive': 0.0,
            'max_negative': 0.0,
            'n_samples': len(valid),
            'degenerate': True
        }
    
    # Cumulative difference: D_k = Σ(y_i - ȳ)
    residuals = np.array(correctness) - y_bar
    cumulative_diff = np.cumsum(residuals)
    
    max_positive = float(np.max(cumulative_diff))
    max_negative = float(np.min(cumulative_diff))
    kuiper_stat = max_positive - max_negative
    
    return {
        'kuiper_stat': kuiper_stat,
        'y_bar': float(y_bar),
        'max_positive': max_positive,
        'max_negative': max_negative,
        'n_samples': len(valid),
        'degenerate': False
    }


def wasted_effort_ratio(results: List[Dict]) -> Dict[str, float]:
    """
    Compute Wasted Effort Ratio: μ_steps(Incorrect) / μ_steps(Correct).
    
    - ρ > 1: Model grinds on unsolved problems (poor calibration)
    - ρ ≈ 1: Model spends similar effort regardless of outcome
    - ρ < 1: Model fails fast (good calibration)
    
    Args:
        results: List of dicts with effort fields and 'correct'.
                 Effort falls back from 'steps' -> 'llm_calls' -> 'effort'.
    
    Returns:
        Dict with 'ratio', 'mean_steps_correct', 'mean_steps_incorrect'
    """
    correct_steps = [_get_effort_value(r) for r in results if r.get('correct') and _get_effort_value(r) > 0]
    incorrect_steps = [_get_effort_value(r) for r in results if not r.get('correct') and _get_effort_value(r) > 0]
    
    mean_correct = float(np.mean(correct_steps)) if correct_steps else 0.0
    mean_incorrect = float(np.mean(incorrect_steps)) if incorrect_steps else 0.0
    
    ratio = mean_incorrect / mean_correct if mean_correct > 0 else float('inf')
    
    return {
        'ratio': ratio,
        'mean_steps_correct': mean_correct,
        'mean_steps_incorrect': mean_incorrect,
        'n_correct': len(correct_steps),
        'n_incorrect': len(incorrect_steps)
    }
