"""
Analysis of dialogue context impact on model performance (AFC).
Compares roberta_afc baseline vs wavlm_roberta_afc_context across the 5 multimodal folds.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS = _ROOT / "results" / "results.json"


def load_context_scores(results_path=None, test_dialogues=None):
    """
    Returns a dict mapping experiment name → list of per-fold F1 scores
    (only for test_dialogues folds, in the same order).
    Also returns roberta_afc scores remapped to the same 5 folds.
    """
    from src.experiments.mmused_text import make_mmused_fallacy_loader
    from src.utils.splits import infer_held_out_dialogue_id, sort_ldocv_splits
    from src.configs.fold_selection import MULTIMODAL_TEST_DIALOGUES

    if test_dialogues is None:
        test_dialogues = MULTIMODAL_TEST_DIALOGUES

    rpath = Path(results_path) if results_path else _DEFAULT_RESULTS
    d = json.load(open(rpath))

    # Remap roberta_afc (35 folds) to the 5 target dialogues
    loader = make_mmused_fallacy_loader("afc")
    splits = list(loader.get_splits("mancini-et-al-2024"))
    splits = sort_ldocv_splits(loader, splits)
    fold_dialogues = [infer_held_out_dialogue_id(loader, sp) for sp in splits]

    roberta_scores_5 = {
        did: score
        for did, score in zip(fold_dialogues, d["roberta_afc"]["scores"])
        if did in test_dialogues
    }

    return {
        "roberta_afc (35 folds)":    (d["roberta_afc"]["mean"],           None),
        "roberta_afc (5 folds)":     (np.mean(list(roberta_scores_5.values())),
                                      [roberta_scores_5[did] for did in test_dialogues]),
        "wavlm_roberta_afc":         (d["wavlm_roberta_afc"]["mean"],
                                      d["wavlm_roberta_afc"]["scores"]),
        "wavlm_roberta_afc_context": (d["wavlm_roberta_afc_context"]["mean"],
                                      d["wavlm_roberta_afc_context"]["scores"]),
    }, test_dialogues


def plot_context_impact(results_path=None, test_dialogues=None, save_path=None) -> None:
    """2-panel figure: mean F1 bar chart + per-fold comparison."""
    exps, dialogues = load_context_scores(results_path, test_dialogues)

    roberta_5  = exps["roberta_afc (5 folds)"][1]
    ctx_scores = exps["wavlm_roberta_afc_context"][1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names  = list(exps.keys())
    means  = [v[0] for v in exps.values()]
    colors = ["#4C72B0", "#4C72B0", "#55A868", "#C44E52"]
    bars   = axes[0].bar(names, means, color=colors, alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylim(0, 0.65)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Mean F1 — context impact on multimodal")
    axes[0].axhline(exps["wavlm_roberta_afc"][0], color="#55A868",
                    linestyle="--", alpha=0.5, label="wavlm baseline")
    axes[0].legend()

    x = np.arange(len(dialogues))
    w = 0.35
    axes[1].bar(x - w / 2, roberta_5,  w, label="roberta_afc (text only)", color="#4C72B0", alpha=0.85)
    axes[1].bar(x + w / 2, ctx_scores, w, label="wavlm+context",           color="#C44E52", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.split("_")[1] for d in dialogues])
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Per-fold: text only vs multimodal+context")
    axes[1].legend()
    axes[1].set_ylim(0, 0.65)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n=== Summary ===")
    for name, (mean, _) in exps.items():
        print(f"  {name:<35} : {mean:.4f}")
    delta = exps["wavlm_roberta_afc"][0] - exps["wavlm_roberta_afc_context"][0]
    print(f"\n  Delta (wavlm vs +context) : {delta:+.4f}")
    print("\n  Per-fold breakdown:")
    for did, r, c in zip(dialogues, roberta_5, ctx_scores):
        print(f"    {did}: roberta={r:.3f}  wavlm+ctx={c:.3f}  Δ={c-r:+.3f}")
