"""Master function for computing metrics."""

from typing import Dict, List, Any

from metrics.clip_score import clip_metrics
from metrics.kid_score import KIDScore
from metrics.ngram_levensthein import ngram_levenshtein
from metrics.ocr_metrics import ocr_metrics

# TODO : add parsers and/or config file to change wanted metrics to compute

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
        num_samples_per_bucket=num_samples
    )

    _, metrics['ngram_levenshtein'] = ngram_levenshtein(
        generated_list=generated_words,
        ground_truth_list=gt_words
    )

    metrics.update(
        ocr_metrics(
            pred_texts=generated_words,
            gt_texts=gt_words
        )
    )

    return metrics
