"""Master function for computing metrics."""

from typing import Dict, List, Any

from metrics.clip_score import clip_metrics
from metrics.kid_score import KIDScore
from metrics.ngram_levensthein import ngram_levenshtein
from metrics.ocr_metrics import evaluate_ocr


def compute_metrics(
        generated_images,
        original_images,
        prompts : List[str],
        generated_words : List[str],
        gt_words : List[str],
        device,
        batch_size : int,
        clip_model,
        num_samples : int
) -> Dict :
    metrics = {}
    metrics['clip-score'] = clip_metrics(
        clip_model=clip_model,
        images=generated_images,
        device=device,
        batch_size=batch_size,
        prompts_A=prompts
    )
    
    metrics['kid_score'] = KIDScore(
        generated_images=generated_images,
        original_images=original_images,
        device=device,
        num_samples_per_bucket=num_samples
    )

    _, metrics['ngram_levenshtein'] = ngram_levenshtein(
        generated_list=generated_words,
        ground_truth_list=gt_words
    )

    metrics.update(
        evaluate_ocr(
            generated_words=generated_words,
            ground_truth=gt_words
        )
    )

    return metrics
