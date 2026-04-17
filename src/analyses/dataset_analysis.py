"""
Dataset exploration for MM-USED-Fallacy: AFC and AFD tasks.
Distribution plots, class weights, dialogue statistics.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight

_ROOT = Path(__file__).resolve().parents[2]

LABEL_NAMES = {
    0: "Appeal to Emotion",
    1: "Appeal to Authority",
    2: "Ad Hominem",
    3: "False Cause",
    4: "Slippery Slope",
    5: "Slogans",
}


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def plot_class_distributions(loader_afc, loader_afd, save_dir=None):
    """Side-by-side bar charts for AFC (6-class) and AFD (binary)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    afc_counts = loader_afc.data["fallacy"].value_counts().sort_index()
    axes[0].bar([LABEL_NAMES[i] for i in afc_counts.index], afc_counts.values,
                color=sns.color_palette("viridis", len(afc_counts)))
    axes[0].set_title("AFC — Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=40)

    afd_counts = loader_afd.data["label"].value_counts().sort_index()
    axes[1].bar(["Non-fallacious (0)", "Fallacious (1)"], afd_counts.values,
                color=["#3498db", "#e74c3c"])
    axes[1].set_title("AFD — Class Distribution", fontweight="bold")
    axes[1].set_ylabel("Count")
    for ax, vc in zip(axes, [afc_counts, afd_counts]):
        for i, v in enumerate(vc.values):
            ax.text(i, v + 1, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "class_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_class_weights(loader_afc, loader_afd, save_path=None):
    """Compute and print balanced class weights for AFC and AFD."""
    afc_labels  = loader_afc.data["fallacy"].values
    afc_classes = np.unique(afc_labels)
    afc_weights = compute_class_weight("balanced", classes=afc_classes, y=afc_labels)

    afd_labels  = loader_afd.data["label"].values
    afd_classes = np.unique(afd_labels)
    afd_weights = compute_class_weight("balanced", classes=afd_classes, y=afd_labels)

    total_afc = len(afc_labels)
    print("=== AFC class weights ===")
    print(f"{'Class':<25} {'N':>6} {'%':>7} {'Weight':>8}")
    for cls, w in zip(afc_classes, afc_weights):
        n = (afc_labels == cls).sum()
        print(f"  {LABEL_NAMES[cls]:<23} {n:>6} {100*n/total_afc:>6.1f}% {w:>8.4f}")

    total_afd = len(afd_labels)
    print("\n=== AFD class weights ===")
    for cls, w in zip(afd_classes, afd_weights):
        n = (afd_labels == cls).sum()
        name = "Fallacious" if cls == 1 else "Non-fallacious"
        print(f"  {name:<23} {n:>6} {100*n/total_afd:>6.1f}% {w:>8.4f}")

    if save_path:
        weights_dict = {
            "afc": {int(c): float(w) for c, w in zip(afc_classes, afc_weights)},
            "afd": {int(c): float(w) for c, w in zip(afd_classes, afd_weights)},
        }
        with open(save_path, "w") as f:
            json.dump(weights_dict, f, indent=2)
        print(f"\nSaved to {save_path}")


# ---------------------------------------------------------------------------
# Text length
# ---------------------------------------------------------------------------

def plot_snippet_lengths(loader_afc, loader_afd, save_dir=None):
    """Histogram + boxplot of snippet/sentence lengths by class."""
    df_afc = loader_afc.data.copy()
    df_afc["n_words"] = df_afc["snippet"].str.split().str.len()

    df_afd = loader_afd.data.copy()
    df_afd["n_words"] = df_afd["sentence"].astype(str).str.split().str.len()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sns.histplot(df_afc["n_words"], bins=30, ax=axes[0, 0], color="#3498db")
    axes[0, 0].set_title("AFC — Snippet length (words)", fontweight="bold")
    axes[0, 0].set_xlabel("Number of words")

    sns.boxplot(data=df_afc, x="fallacy", y="n_words", palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_xticks(range(6))
    axes[0, 1].set_xticklabels(list(LABEL_NAMES.values()), rotation=40, ha="right")
    axes[0, 1].set_title("AFC — Snippet length per class", fontweight="bold")

    sns.histplot(df_afd["n_words"], bins=30, ax=axes[1, 0], color="#e74c3c")
    axes[1, 0].set_title("AFD — Sentence length (words)", fontweight="bold")
    axes[1, 0].set_xlabel("Number of words")

    df_afd["label_name"] = df_afd["label"].map({0: "Non-fallacious", 1: "Fallacious"})
    sns.boxplot(data=df_afd, x="label_name", y="n_words", ax=axes[1, 1])
    axes[1, 1].set_title("AFD — Sentence length per class", fontweight="bold")

    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "text_lengths.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Dialogue statistics
# ---------------------------------------------------------------------------

def plot_dialogue_stats(loader_afd, save_dir=None):
    """Sentences per dialogue, fallacy rate per dialogue, stacked bar."""
    df = loader_afd.data.copy()

    sent_per_dlg   = df.groupby("dialogue_id").size()
    fallacy_counts = df[df["label"] == 1].groupby("dialogue_id").size()
    total_per_dlg  = df.groupby("dialogue_id").size()
    fallacy_rate   = (fallacy_counts / total_per_dlg * 100).fillna(0).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(sent_per_dlg, bins=20, ax=axes[0], color="steelblue")
    axes[0].set_title("Sentences per Dialogue", fontweight="bold")
    axes[0].set_xlabel("Number of sentences")

    colors = ["#e74c3c" if v > fallacy_rate.mean() else "#3498db" for v in fallacy_rate.values]
    axes[1].bar(range(len(fallacy_rate)), fallacy_rate.values, color=colors, edgecolor="white")
    axes[1].axhline(fallacy_rate.mean(), color="#2c3e50", linestyle="--",
                    label=f"Mean: {fallacy_rate.mean():.1f}%")
    axes[1].set_title("Fallacy Rate per Dialogue", fontweight="bold")
    axes[1].set_xlabel("Dialogue")
    axes[1].set_ylabel("% Fallacious sentences")
    axes[1].set_xticks(range(len(fallacy_rate)))
    axes[1].set_xticklabels(fallacy_rate.index, rotation=90, fontsize=6)
    axes[1].legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "dialogue_stats.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Sentences/dialogue — mean: {sent_per_dlg.mean():.1f}  "
          f"median: {sent_per_dlg.median():.1f}  max: {sent_per_dlg.max()}")
    print(f"Fallacy rate       — mean: {fallacy_rate.mean():.1f}%  "
          f"max: {fallacy_rate.max():.1f}%")


def plot_fallacy_heatmap(loader_afc, save_dir=None):
    """Heatmap: fallacy type × dialogue."""
    df = loader_afc.data.copy()
    hm = df.groupby(["dialogue_id", "fallacy"]).size().unstack(fill_value=0)
    hm.columns = [LABEL_NAMES[c] for c in hm.columns]

    plt.figure(figsize=(14, 10))
    sns.heatmap(hm, cmap="YlOrRd", linewidths=0.5, linecolor="white",
                annot=True, fmt="d", cbar_kws={"label": "Count"})
    plt.title("Fallacy Type per Dialogue", fontsize=14, fontweight="bold")
    plt.xticks(rotation=40, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "fallacy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Examples & diagnostics
# ---------------------------------------------------------------------------

def print_examples(loader_afc, loader_afd, n=2):
    """Print n examples per AFC class and AFD binary label."""
    print("=== AFC — examples per class ===\n")
    for label_id, label_name in LABEL_NAMES.items():
        examples = loader_afc.data[loader_afc.data["fallacy"] == label_id]["snippet"].head(n)
        print(f"--- {label_name} ---")
        for i, ex in enumerate(examples):
            print(f"  {i+1}. {ex[:200]}")
        print()

    print("=== AFD — fallacious vs non-fallacious ===\n")
    for label, name in [(0, "Non-fallacious"), (1, "Fallacious")]:
        print(f"--- {name} ---")
        for i, ex in enumerate(loader_afd.data[loader_afd.data["label"] == label]["sentence"].head(n)):
            print(f"  {i+1}. {ex[:200]}")
        print()


def print_afd_diagnostics(loader_afd):
    """Duplicate sentences across dialogues (AFD-specific issue)."""
    df = loader_afd.data.copy()
    dup = (df.groupby(df["sentence"].astype(str), as_index=False)
             .agg(n_dialogues=("dialogue_id", "nunique"),
                  dialogues=("dialogue_id", lambda s: sorted(set(s)))))
    dup = dup.sort_values("n_dialogues", ascending=False)
    print(f"Sentences appearing in >1 dialogue: {(dup['n_dialogues'] > 1).sum()}")
    print(dup.head(10).to_string(index=False))
