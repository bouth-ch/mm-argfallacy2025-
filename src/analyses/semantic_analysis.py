"""
Semantic similarity analysis between reference text and Whisper transcriptions.
Requires sentence-transformers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TYPE_COLORS = {
    "Good":         "#2ecc71",
    "TooLong":      "#e74c3c",
    "TooShort":     "#3498db",
    "BadAlignment": "#f39c12",
}


def compute_semantic_similarity(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """Add 'semantic_sim' column using cosine similarity of sentence embeddings."""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"Loading SentenceTransformer ({model_name})...")
    model = SentenceTransformer(model_name)

    ref_embs     = model.encode(df["ref_text"].tolist(),     batch_size=64, show_progress_bar=True)
    whisper_embs = model.encode(df["whisper_text"].tolist(), batch_size=64, show_progress_bar=True)

    df = df.copy()
    df["semantic_sim"] = [
        cosine_similarity([ref_embs[i]], [whisper_embs[i]])[0][0]
        for i in range(len(df))
    ]
    return df


def print_semantic_stats(df: pd.DataFrame) -> None:
    print(f"Semantic similarity global :")
    print(f"  Mean   : {df['semantic_sim'].mean():.3f}")
    print(f"  Median : {df['semantic_sim'].median():.3f}")
    print(f"  < 0.5  : {(df['semantic_sim'] < 0.5).sum()} ({100*(df['semantic_sim'] < 0.5).mean():.1f}%)")
    print(f"  > 0.8  : {(df['semantic_sim'] > 0.8).sum()} ({100*(df['semantic_sim'] > 0.8).mean():.1f}%)")
    print()
    print(f"{'Fallacy':<25} {'Mean':>6} {'Median':>7}")
    for cls, grp in df.groupby("fallacy"):
        print(f"  {cls:<23} {grp['semantic_sim'].mean():>6.3f} {grp['semantic_sim'].median():>7.3f}")


def plot_semantic_per_class(df: pd.DataFrame, save_path=None) -> None:
    """Histogram of semantic similarity split by alignment type, one subplot per class."""
    classes = sorted(df["fallacy"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(20, 5), sharey=True)

    for ax, cls in zip(axes, classes):
        grp = df[df["fallacy"] == cls]
        for t, color in TYPE_COLORS.items():
            sub = grp[grp["alignment_type"] == t]["semantic_sim"]
            if len(sub) > 0:
                ax.hist(sub, bins=20, alpha=0.7, color=color,
                        label=f"{t} (n={len(sub)})", edgecolor="white")
        ax.axvline(grp["semantic_sim"].median(), color="black", linestyle="--",
                   linewidth=1.5, label=f"Median: {grp['semantic_sim'].median():.3f}")
        ax.set_title(f"{cls}\nmean={grp['semantic_sim'].mean():.3f}", fontweight="bold")
        ax.set_xlabel("Semantic Similarity")
        ax.set_xlim(0, 1.1)
        ax.legend(fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Count")

    fig.suptitle("Semantic Similarity Distribution per Class and Alignment Type",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_semantic_misaligned(df: pd.DataFrame, save_path=None) -> None:
    """Histograms for misaligned clips only (TooLong / TooShort / BadAlignment)."""
    classes = sorted(df["fallacy"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        grp = df[df["fallacy"] == cls]
        for t in ["BadAlignment", "TooShort", "TooLong"]:
            sub = grp[grp["alignment_type"] == t]["semantic_sim"]
            if len(sub) > 0:
                axes[i].hist(sub, bins=15, alpha=0.7, color=TYPE_COLORS[t],
                             label=f"{t} (n={len(sub)})", edgecolor="white")
        axes[i].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold 0.5")
        problematic = grp[grp["alignment_type"] != "Good"]["semantic_sim"]
        if len(problematic) > 0:
            axes[i].axvline(problematic.mean(), color="purple", linestyle=":",
                            linewidth=1.5, label=f"Mean: {problematic.mean():.3f}")
        n_dangerous = (problematic < 0.5).sum() if len(problematic) > 0 else 0
        axes[i].set_title(f"{cls}\nn_problematic={len(problematic)}  semantic<0.5: {n_dangerous}",
                          fontweight="bold")
        axes[i].set_xlabel("Semantic Similarity")
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=8)
        axes[i].set_xlim(0, 1.1)

    fig.suptitle("Semantic Similarity — Misaligned Clips Only", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_semantic_boxplot(df: pd.DataFrame, save_path=None) -> None:
    """Boxplot of semantic similarity by alignment type, one subplot per class."""
    import seaborn as sns

    df_prob = df[df["alignment_type"] != "Good"].copy()
    classes = sorted(df_prob["fallacy"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        grp = df_prob[df_prob["fallacy"] == cls]
        sns.boxplot(data=grp, x="alignment_type", y="semantic_sim",
                    palette={k: v for k, v in TYPE_COLORS.items() if k != "Good"},
                    order=["BadAlignment", "TooShort", "TooLong"], ax=axes[i])
        axes[i].axhline(0.5, color="black", linestyle="--", linewidth=1.5,
                        label="Danger threshold (0.5)")
        for j, t in enumerate(["BadAlignment", "TooShort", "TooLong"]):
            n = len(grp[grp["alignment_type"] == t])
            axes[i].text(j, -0.08, f"n={n}", ha="center", fontsize=9, color="#555")
        axes[i].set_title(cls, fontweight="bold", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Semantic Similarity")
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].legend(fontsize=8)
        axes[i].tick_params(axis="x", rotation=20)

    fig.suptitle("Semantic Similarity per Alignment Type — Misaligned Clips Only",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
