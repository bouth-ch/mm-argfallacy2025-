"""Per-fold binary prediction stats from stored results.json (no re-training needed)."""

from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import load_results
from src.evaluation.schema import label_column, text_column
from src.utils.splits import sort_ldocv_splits


def _splits_in_order(loader, *, sort_splits: bool, split_key: str):
    splits = list(loader.get_splits(split_key))
    if sort_splits:
        splits = sort_ldocv_splits(loader, splits)
    return splits


def per_fold_binary_prediction_stats(experiment_name, loader, *, results_path=None,
                                      sort_splits=False, split_key="mancini-et-al-2024",
                                      positive_label=1):
    """One row per CV fold: test size, fallacy rate, P/R/F1 recomputed from stored predictions."""
    if results_path is None:
        from pathlib import Path as _Path
        results_path = str(_Path(__file__).resolve().parents[2] / 'results' / 'results.json')
    res: dict[str, Any] = load_results(experiment_name, results_path)
    preds = list(res["predictions"])
    y_flat = list(res["true_labels"])
    scores = list(res["scores"])
    splits = _splits_in_order(loader, sort_splits=sort_splits, split_key=split_key)

    if len(preds) != len(y_flat):
        raise ValueError(
            f"predictions ({len(preds)}) and true_labels ({len(y_flat)}) length differ."
        )
    if len(scores) != len(splits):
        raise ValueError(
            f"scores ({len(scores)}) vs n_splits ({len(splits)}): mismatch."
        )

    cursor = 0
    rows: list[dict[str, Any]] = []
    for fold_idx, sp in enumerate(splits):
        n_exp = len(sp.test)
        end = min(cursor + n_exp, len(preds))
        p_chunk = np.array(preds[cursor:end], dtype=int)
        y_chunk = np.array(y_flat[cursor:end], dtype=int)
        cursor = end
        chunk_ok = len(p_chunk) == n_exp

        if len(p_chunk) == 0:
            pred_pos = 0
            tp = fp = tn = fn = 0
            prec = rec = f1 = float("nan")
            n_true_pos = 0
        else:
            pred_pos = int((p_chunk == positive_label).sum())
            y_pos = (y_chunk == positive_label).astype(int)
            p_pos = (p_chunk == positive_label).astype(int)
            tp = int(((y_pos == 1) & (p_pos == 1)).sum())
            fp = int(((y_pos == 0) & (p_pos == 1)).sum())
            fn = int(((y_pos == 1) & (p_pos == 0)).sum())
            tn = int(((y_pos == 0) & (p_pos == 0)).sum())
            n_true_pos = int(y_pos.sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * prec * rec / (prec + rec)
                if (prec + rec) > 0
                else 0.0
            )

        n_eff = len(p_chunk)
        rows.append(
            {
                "fold": fold_idx + 1,
                "expected_test_n": n_exp,
                "chunk_n": n_eff,
                "chunk_complete": chunk_ok and (n_eff == n_exp),
                "n_true_fallacy": n_true_pos,
                "fallacy_rate": n_true_pos / n_eff if n_eff else float("nan"),
                "n_pred_fallacy": pred_pos,
                "pred_fallacy_rate": pred_pos / n_eff if n_eff else float("nan"),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": prec,
                "recall": rec,
                "f1_recomputed": f1,
                "f1_stored": float(scores[fold_idx]),
            }
        )

    expected_total = sum(len(sp.test) for sp in splits)
    notes = []
    if cursor != len(preds):
        notes.append(
            f"Consumed cursor={cursor} but len(predictions)={len(preds)} (unexpected)."
        )
    if expected_total > len(preds):
        notes.append(
            f"Stored predictions shorter than full OOF by {expected_total - len(preds)} "
            f"({len(preds)}/{expected_total}); last fold(s) have partial or empty chunks."
        )

    out = pd.DataFrame(rows)
    out.attrs["chunk_notes"] = " ".join(notes) if notes else ""
    out.attrs["sort_splits"] = sort_splits
    return out


def gold_fallacy_sentences_for_dialogue(
    loader,
    dialogue_id: str,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Ground-truth Fallacy sentences (``label==1``) for one ``dialogue_id``."""
    df = loader.data
    lcol = label_column(loader.task_name)
    tcol = text_column(loader.task_name)
    m = (df["dialogue_id"].astype(str) == str(dialogue_id)) & (df[lcol] == 1)
    sub = df.loc[m, [tcol, lcol, "dialogue_id"]].copy()
    sub = sub.rename(columns={tcol: "text", lcol: "label"})
    if max_rows is not None:
        sub = sub.head(max_rows)
    return sub.reset_index(drop=True)


def plot_f1_vs_fallacy_rate(
    fold_df: pd.DataFrame,
    *,
    title: str = "Per-fold binary F1 vs fallacy rate (test fold)",
):
    """Scatter: ``f1_stored`` vs ``fallacy_rate`` (uses only rows with valid fallacy_rate)."""
    import matplotlib.pyplot as plt

    d = fold_df.dropna(subset=["fallacy_rate", "f1_stored"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["fallacy_rate"], d["f1_stored"], alpha=0.75, edgecolors="k", linewidths=0.4)
    for _, r in d.iterrows():
        ax.annotate(
            str(int(r["fold"])),
            (r["fallacy_rate"], r["f1_stored"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=7,
            alpha=0.8,
        )
    ax.set_xlabel("Fallacy rate (n_true_fallacy / chunk_n)")
    ax.set_ylabel("Binary F1 (stored per fold)")
    ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_precision_recall_per_fold(fold_df: pd.DataFrame, *, title: str = "Per-fold precision vs recall"):
    import matplotlib.pyplot as plt

    d = fold_df[np.isfinite(fold_df["precision"]) & np.isfinite(fold_df["recall"])].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(d["recall"], d["precision"], alpha=0.75, edgecolors="k", linewidths=0.4)
    for _, r in d.iterrows():
        ax.annotate(
            str(int(r["fold"])),
            (r["recall"], r["precision"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=7,
            alpha=0.8,
        )
    lim = max(1.05, float(d["recall"].max()), float(d["precision"].max()))
    ax.plot([0, lim], [0, lim], "k--", alpha=0.25, label="recall = precision")
    ax.set_xlabel("Recall (Fallacy)")
    ax.set_ylabel("Precision (Fallacy)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax
