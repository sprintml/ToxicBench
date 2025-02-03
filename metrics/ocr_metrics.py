"""Script for computing f1-score and Levenshtein distance from OCR'd text from evaluated images."""

import numpy as np
import Levenshtein 
import copy
import typing as t

# OCR accuracy, precision and recall metrics
def get_p_r_acc(pred, gt):

    pred = [p.strip().lower() for p in pred]
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p)
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)

    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if "".join(pred_sorted) == "".join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc


def ocr_metrics(pred_texts, gt_texts):
    assert len(pred_texts) == len(gt_texts)

    pr, rec, acc = [], [], []
    for pt, gt in zip(pred_texts, gt_texts):
        p, r, a = get_p_r_acc(pt, gt)
        pr.append(p)
        rec.append(r)
        acc.append(a)
    return pr, rec, acc


def gen_words_preprocess(generated_words, ground_truth) :

    for i in range(len(generated_words)) :
        if ground_truth[i] in generated_words[i] :
            generated_words[i] = ground_truth[i]

    return generated_words


def evaluate_ocr(generated_words: t.List[str], ground_truth: t.List[str]) -> t.Dict[str, t.Any]:
    generated_words = gen_words_preprocess(generated_words, ground_truth)
    pr, re, acc = ocr_metrics(generated_words, ground_truth)
    distances = [Levenshtein.distance(ground_truth[i], generated_words[i]) for i in range(len(generated_words))]
    avg_dist = sum(distances) / len(distances)
    f1_score = (2 * np.mean(pr) * np.mean(re)) / (np.mean(pr) + np.mean(re))
    return {
        "precision": np.mean(pr),
        "recall": np.mean(re),
        "f1_score" : f1_score,
        "accuracy": np.mean(acc),
        "avg_levenshtein_distance": avg_dist,
    }

