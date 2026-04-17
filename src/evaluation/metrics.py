import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torchmetrics.functional.classification import (
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

from src.evaluation.schema import label_column, score_column_from_metric
from src.utils.fold_manifest import dialogue_for_fold
from src.utils.splits import infer_held_out_dialogue_id, sort_ldocv_splits


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS = str(_ROOT / 'results' / 'results.json')


def load_results(experiment_name, results_path=None):
    if results_path is None:
        results_path = _DEFAULT_RESULTS
    with open(results_path) as f:
        data = json.load(f)
    return data[experiment_name]


def _split_list_for_fold_index(loader, result: dict) -> list:
    raw = list(loader.get_splits('mancini-et-al-2024'))
    ordered = sort_ldocv_splits(loader, raw)
    ids = result.get('dialogue_ids') or []
    refs = [(j, x) for j, x in enumerate(ids) if x is not None]
    if not refs:
        return raw
    for cand in (ordered, raw):
        try:
            if all(
                infer_held_out_dialogue_id(loader, cand[j]) == x for j, x in refs
            ):
                return cand
        except RuntimeError:
            continue
    return raw


def _held_out_from_manifest(experiment_name, fold_idx, results_path):
    return dialogue_for_fold(experiment_name, fold_idx, results_path=results_path)


def _infer_held_out_ids(loader, splits: list) -> list[str]:
    return [infer_held_out_dialogue_id(loader, sp) for sp in splits]


def compute_fold_table(experiment_name, loader, label_names, results_path=None):
    """Return a DataFrame with per-fold F1, sample count, classes present."""
    result = load_results(experiment_name, results_path)
    scores = result['scores']
    task_name = result.get('task_name', 'afc')
    score_col = score_column_from_metric(result.get('metric', 'test_macro_f1'))
    lcol = label_column(task_name)
    splits = _split_list_for_fold_index(loader, result)
    inferred = _infer_held_out_ids(loader, splits)

    rows = []
    for fold_idx, score in enumerate(scores):
        try:
            dialogue_id = _held_out_from_manifest(
                experiment_name, fold_idx, results_path
            )
            if dialogue_id is None and fold_idx < len(inferred):
                dialogue_id = inferred[fold_idx]
        except ValueError:
            raise
        if dialogue_id is None:
            rows.append({
                'fold': fold_idx + 1,
                'dialogue_id': None,
                score_col: round(score, 4),
                'n_samples': None,
                'classes_present': None,
                'n_classes': None,
                'dominant_class': None,
            })
            continue

        df = loader.data[loader.data['dialogue_id'] == dialogue_id]
        dom_idx = df[lcol].value_counts().idxmax()
        rows.append({
            'fold': fold_idx + 1,
            'dialogue_id': dialogue_id,
            score_col: round(score, 4),
            'n_samples': len(df),
            'classes_present': sorted(df[lcol].unique().tolist()),
            'n_classes': df[lcol].nunique(),
            'dominant_class': label_names[int(dom_idx)],
        })

    sort_col = score_col if score_col in rows[0] else 'fold'
    if sort_col == 'fold' or all(r.get(sort_col) is None for r in rows):
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values(sort_col, ascending=False)


def compute_confusion_matrix(experiment_name, num_classes=None, results_path=None):
    """Return (cm_counts, cm_normalised, all_preds, all_labels) using torchmetrics."""
    result = load_results(experiment_name, results_path)
    preds = result['predictions']
    labels = result['true_labels']
    # num_classes should be passed explicitly - inferring from preds/labels can miss
    # classes that never get predicted (especially rare ones in imbalanced AFC)
    if num_classes is None:
        n_classes = max(max(preds), max(labels)) + 1
    else:
        n_classes = num_classes

    p = th.tensor(preds, dtype=th.long)
    y = th.tensor(labels, dtype=th.long)
    cm = multiclass_confusion_matrix(p, y, num_classes=n_classes)
    cm_counts = cm.cpu().numpy().astype(np.int64)
    row_sums = cm_counts.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm = np.divide(
        cm_counts.astype(np.float64),
        row_sums,
        out=np.zeros_like(cm_counts, dtype=np.float64),
        where=row_sums != 0,
    )
    return cm_counts, cm_norm, preds, labels


def compute_per_class_metrics(preds, labels, class_names):
    """Per-class precision, recall, F1, support (torchmetrics, same family as mamkit training)."""
    n_classes = len(class_names)
    p = th.tensor(preds, dtype=th.long)
    y = th.tensor(labels, dtype=th.long)

    prec = multiclass_precision(
        p, y, num_classes=n_classes, average=None, zero_division=0
    )
    rec = multiclass_recall(
        p, y, num_classes=n_classes, average=None, zero_division=0
    )
    f1 = multiclass_f1_score(
        p, y, num_classes=n_classes, average=None, zero_division=0
    )
    support = th.bincount(y, minlength=n_classes)

    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'f1-score': float(f1[i]),
            'support': int(support[i]),
        })
    df = pd.DataFrame(rows, index=class_names)
    return df


def select_folds(experiment_name, loader, results_path=None,
                 valid_folds=None):
    """Return best, middle, worst fold info dicts (uses stored dialogue_ids when present)."""
    result = load_results(experiment_name, results_path)
    scores = result['scores']
    sorted_idx = np.argsort(scores)

    if valid_folds is not None:
        valid_set = set(valid_folds)
        sorted_idx = np.array([i for i in sorted_idx if i in valid_set])
        if len(sorted_idx) < 3:
            raise ValueError(
                f"Need at least 3 valid folds for best/middle/worst, "
                f"got {len(sorted_idx)}: {sorted_idx.tolist()}"
            )

    worst_idx = int(sorted_idx[0])
    best_idx = int(sorted_idx[-1])
    middle_idx = int(sorted_idx[len(sorted_idx) // 2])

    splits = _split_list_for_fold_index(loader, result)
    inferred = _infer_held_out_ids(loader, splits)

    def info(idx):
        did = _held_out_from_manifest(experiment_name, idx, results_path)
        if did is None and idx < len(inferred):
            did = inferred[idx]
        out = {
            'fold_idx': idx,
            'dialogue_id': did,
            'score': round(scores[idx], 4),
        }
        return out

    return {
        'best': info(best_idx),
        'middle': info(middle_idx),
        'worst': info(worst_idx),
    }
