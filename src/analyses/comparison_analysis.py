"""
Text vs Multimodal comparison analysis for MM-USED-Fallacy (AFC).
Reconstructs per-dialogue predictions and provides plots + diagnostics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS = _ROOT / "results" / "results.json"
_DEFAULT_DATASET = _ROOT / "data" / "MMUSED-fallacy" / "dataset.pkl"
_DEFAULT_AUDIT   = _ROOT / "results" / "whisper_audit.csv"

FALLACY_MAP = {
    "AppealtoEmotion": 0, "AppealtoAuthority": 1, "AdHominem": 2,
    "FalseCause": 3, "Slipperyslope": 4, "Slogans": 5,
}
FALLACY_NAMES = {v: k for k, v in FALLACY_MAP.items()}

TARGET = ["13_1988", "22_1996", "25_2000", "31_2004", "46_2020"]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _find_dialogue_offset(flat_true, dialogue_labels):
    n = len(dialogue_labels)
    for start in range(len(flat_true) - n + 1):
        if np.array_equal(flat_true[start:start + n], dialogue_labels):
            return start
    return None


def load_results(results_path=None, dataset_path=None, target=None):
    """
    Load and reconstruct per-dialogue predictions for text and multimodal models.
    Returns (text_results, mm_results) — dicts mapping dialogue_id → {true, pred}.
    """
    rpath = Path(results_path) if results_path else _DEFAULT_RESULTS
    dpath = Path(dataset_path) if dataset_path else _DEFAULT_DATASET
    dialogues = target or TARGET

    r  = json.load(open(rpath))
    df = pd.read_pickle(dpath)
    df["label"] = df["fallacy"].map(FALLACY_MAP)

    # Text — sliding window to find each dialogue's position in flat predictions
    text_exp       = r["roberta_afc"]
    flat_preds_txt = np.array(text_exp["predictions"])
    flat_true_txt  = np.array(text_exp["true_labels"])

    text_results = {}
    for did in dialogues:
        expected = df[df["dialogue_id"] == did]["label"].values
        offset   = _find_dialogue_offset(flat_true_txt, expected)
        if offset is None:
            raise ValueError(f"Could not locate dialogue {did} in roberta_afc flat predictions")
        text_results[did] = {
            "true": expected,
            "pred": flat_preds_txt[offset:offset + len(expected)],
        }

    # Multimodal — stored per dialogue_id in results.json
    mm_exp    = r["wavlm_roberta_afc"]
    mm_dids   = mm_exp["dialogue_ids"]
    mm_preds  = np.array(mm_exp["predictions"])
    mm_true   = np.array(mm_exp["true_labels"])

    mm_results = {}
    offset = 0
    for did in mm_dids:
        size = len(df[df["dialogue_id"] == did])
        mm_results[did] = {
            "true": mm_true[offset:offset + size],
            "pred": mm_preds[offset:offset + size],
        }
        offset += size

    return text_results, mm_results


def build_comparison_df(text_results, mm_results, target=None):
    """Return a DataFrame with per-dialogue macro F1 for text and multimodal."""
    dialogues = target or TARGET
    rows = []
    for did in dialogues:
        f1_t  = f1_score(text_results[did]["true"], text_results[did]["pred"],
                         average="macro", zero_division=0)
        f1_mm = f1_score(mm_results[did]["true"],   mm_results[did]["pred"],
                         average="macro", zero_division=0)
        rows.append({"dialogue": did, "text": f1_t, "multimodal": f1_mm, "delta": f1_mm - f1_t})
    return pd.DataFrame(rows)


def load_all_experiment_means(results_path=None):
    """
    Return a dict of {exp_name: mean_f1} for all AFC experiments in results.json.
    Includes remapped roberta_afc mean on the same 5 folds.
    """
    from src.experiments.mmused_text import make_mmused_fallacy_loader
    from src.utils.splits import infer_held_out_dialogue_id, sort_ldocv_splits

    rpath = Path(results_path) if results_path else _DEFAULT_RESULTS
    r = json.load(open(rpath))

    # Remap roberta_afc (35 folds) → 5 folds
    loader = make_mmused_fallacy_loader("afc")
    splits = list(loader.get_splits("mancini-et-al-2024"))
    splits = sort_ldocv_splits(loader, splits)
    fold_dialogues = [infer_held_out_dialogue_id(loader, sp) for sp in splits]
    roberta_5 = np.mean([s for did, s in zip(fold_dialogues, r["roberta_afc"]["scores"])
                         if did in TARGET])

    return {
        "roberta_afc":              roberta_5,
        "wavlm_roberta_afc":        r["wavlm_roberta_afc"]["mean"],
        "wavlm_roberta_afc_whisper":r.get("wavlm_roberta_afc_whisper", {}).get("mean"),
        "wavlm_roberta_afc_context":r.get("wavlm_roberta_afc_context", {}).get("mean"),
        "longformer_afc_context":   r.get("longformer_afc_context",    {}).get("mean"),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all_experiments(results_path=None, save_path=None):
    """Bar chart comparing all AFC experiments on the same 5 folds."""
    means = load_all_experiment_means(results_path)
    means = {k: v for k, v in means.items() if v is not None}

    LABELS = {
        "roberta_afc":               "RoBERTa\n(text)",
        "wavlm_roberta_afc":         "WavLM+RoBERTa\n(baseline)",
        "wavlm_roberta_afc_whisper": "WavLM+RoBERTa\n(whisper text)",
        "wavlm_roberta_afc_context": "WavLM+RoBERTa\n(+context)",
        "longformer_afc_context":    "Longformer\n(+context)",
    }
    COLORS = {
        "roberta_afc":               "#4C72B0",
        "wavlm_roberta_afc":         "#55A868",
        "wavlm_roberta_afc_whisper": "#DD8452",
        "wavlm_roberta_afc_context": "#C44E52",
        "longformer_afc_context":    "#9467BD",
    }

    keys   = [k for k in LABELS if k in means]
    labels = [LABELS[k] for k in keys]
    values = [means[k] for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(means["wavlm_roberta_afc"], color="#55A868", linestyle="--",
               alpha=0.6, label=f"baseline {means['wavlm_roberta_afc']:.3f}")
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("Macro F1 (5 folds)")
    ax.set_title("AFC — comparaison de toutes les expériences (mêmes 5 folds)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print("\n=== Résumé ===")
    baseline = means["wavlm_roberta_afc"]
    for k in keys:
        delta = means[k] - baseline
        marker = "✓" if delta >= 0 else "✗"
        print(f"  {LABELS[k].replace(chr(10),' '):<35} {means[k]:.4f}  Δ={delta:+.4f} {marker}")


def plot_text_vs_multimodal(comp_df, save_path=None):
    """Bar chart (text vs multimodal) + audio gain per fold."""
    TARGET_FOLDS = comp_df["dialogue"].tolist()
    x = np.arange(len(TARGET_FOLDS))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x - w/2, comp_df["text"],       w, label="RoBERTa (text)", color="#3498db", alpha=0.85)
    axes[0].bar(x + w/2, comp_df["multimodal"], w, label="WavLM+RoBERTa",  color="#2ecc71", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TARGET_FOLDS, rotation=20)
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Text vs Multimodal — same 5 folds")
    axes[0].legend()
    axes[0].set_ylim(0, 0.75)

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in comp_df["delta"]]
    axes[1].bar(x, comp_df["delta"], color=colors, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TARGET_FOLDS, rotation=20)
    axes[1].set_ylabel("Δ F1 (multimodal − text)")
    axes[1].set_title("Audio gain per fold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(comp_df.to_string(index=False))
    print(f"\nMean text      : {comp_df['text'].mean():.3f}")
    print(f"Mean multimodal: {comp_df['multimodal'].mean():.3f}")
    print(f"Mean delta     : {comp_df['delta'].mean():+.3f}")


def plot_per_class_f1(text_results, mm_results, target=None, save_path=None):
    """Per-class F1 subplot for each dialogue."""
    dialogues = target or TARGET
    fig, axes = plt.subplots(1, len(dialogues), figsize=(20, 5), sharey=True)

    for ax, did in zip(axes, dialogues):
        t_true, t_pred   = text_results[did]["true"],  text_results[did]["pred"]
        mm_true, mm_pred = mm_results[did]["true"],     mm_results[did]["pred"]

        present = sorted(set(t_true) | set(mm_true))
        names   = [FALLACY_NAMES[c] for c in present]
        f1_t    = f1_score(t_true,  t_pred,  labels=present, average=None, zero_division=0)
        f1_mm   = f1_score(mm_true, mm_pred, labels=present, average=None, zero_division=0)

        x = np.arange(len(present))
        ax.bar(x - 0.2, f1_t,  0.4, label="text", color="#3498db", alpha=0.85)
        ax.bar(x + 0.2, f1_mm, 0.4, label="mm",   color="#2ecc71", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(did, fontsize=9)
        ax.set_ylim(0, 1.05)
        if ax == axes[0]:
            ax.set_ylabel("F1")
            ax.legend(fontsize=7)

    plt.suptitle("Per-class F1 — Text vs Multimodal", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_fold_errors(did, text_results, mm_results, audit_path=None):
    """Print class distribution, errors for both models, and whisper audit stats."""
    t_true,  t_pred  = text_results[did]["true"],  text_results[did]["pred"]
    mm_true, mm_pred = mm_results[did]["true"],     mm_results[did]["pred"]

    print(f"=== {did} — class distribution ===")
    for c, name in FALLACY_NAMES.items():
        n = (t_true == c).sum()
        if n > 0:
            print(f"  {name:<25} n={n}")

    print(f"\n=== Text errors ({(t_true != t_pred).sum()}) ===")
    for tr, pr in zip(t_true, t_pred):
        if tr != pr:
            print(f"  true={FALLACY_NAMES[tr]:<25} pred={FALLACY_NAMES[pr]}")

    print(f"\n=== Multimodal errors ({(mm_true != mm_pred).sum()}) ===")
    for tr, pr in zip(mm_true, mm_pred):
        if tr != pr:
            print(f"  true={FALLACY_NAMES[tr]:<25} pred={FALLACY_NAMES[pr]}")

    apath = Path(audit_path) if audit_path else _DEFAULT_AUDIT
    fold_audit = pd.read_csv(apath).fillna("").query("dialogue_id == @did")
    print(f"\n=== Whisper audit {did} ===")
    print(f"  Total clips : {len(fold_audit)}")
    print(f"  WER moyen   : {fold_audit['wer'].mean():.3f}")
    print(f"  WER médian  : {fold_audit['wer'].median():.3f}")


def plot_token_length_distribution(did, base_data_path=None, save_path=None):
    """
    Histogram of RoBERTa token lengths for context+snippet vs snippet alone.
    Highlights the 512-token truncation limit.
    """
    from transformers import AutoTokenizer
    from mamkit.data.datasets import MMUSEDFallacy, InputMode
    from src.data.context_dataset import ContextMMUSEDFallacy

    dpath = Path(base_data_path) if base_data_path else _ROOT / "data"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    ctx_df = ContextMMUSEDFallacy(
        task_name="afc", input_mode=InputMode.TEXT_ONLY, base_data_path=dpath
    ).data.query("dialogue_id == @did").copy()
    ctx_df["n_tokens"] = ctx_df["snippet"].apply(
        lambda t: len(tokenizer(t, truncation=False)["input_ids"])
    )

    plain_df = MMUSEDFallacy(
        task_name="afc", input_mode=InputMode.TEXT_ONLY, base_data_path=dpath
    ).data.query("dialogue_id == @did").copy()
    plain_df["n_tokens"] = plain_df["snippet"].apply(
        lambda t: len(tokenizer(t, truncation=False)["input_ids"])
    )

    print(f"=== {did} — token lengths (context+snippet) ===")
    print(f"  Min    : {ctx_df['n_tokens'].min()}")
    print(f"  Max    : {ctx_df['n_tokens'].max()}")
    print(f"  Mean   : {ctx_df['n_tokens'].mean():.1f}")
    print(f"  > 512  : {(ctx_df['n_tokens'] > 512).sum()} / {len(ctx_df)}")
    print(f"  > 400  : {(ctx_df['n_tokens'] > 400).sum()} / {len(ctx_df)}")
    print(f"\n=== {did} — token lengths (snippet only) ===")
    print(f"  Max    : {plain_df['n_tokens'].max()}")
    print(f"  Mean   : {plain_df['n_tokens'].mean():.1f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ctx_df["n_tokens"],   bins=20, color="#C44E52", alpha=0.7,
            label="context+snippet", edgecolor="white")
    ax.hist(plain_df["n_tokens"], bins=20, color="#4C72B0", alpha=0.7,
            label="snippet seul", edgecolor="white")
    ax.axvline(512, color="red", linestyle="--", linewidth=1.5, label="limite RoBERTa (512)")
    ax.set_xlabel("Nombre de tokens")
    ax.set_ylabel("Fréquence")
    ax.set_title(f"{did} — longueur des inputs tokenisés")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
