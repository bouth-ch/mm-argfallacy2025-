"""
Audio-text alignment analysis for MM-USED-Fallacy.
Computes match scores, classifies clips into 4 types, and provides plots.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_AUDIT = _ROOT / "results" / "whisper_audit.csv"

TYPE_COLORS = {
    "Good":         "#2ecc71",
    "TooLong":      "#e74c3c",
    "TooShort":     "#3498db",
    "BadAlignment": "#f39c12",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _best_match_score(ref: str, hyp: str) -> float:
    ref_words = _normalize(ref).split()
    hyp_words = _normalize(hyp).split()
    if not ref_words or not hyp_words:
        return 0.0
    n = len(ref_words)
    best = 0.0
    for start in range(max(1, len(hyp_words) - n + 1)):
        matches = sum(r == h for r, h in zip(ref_words, hyp_words[start:start + n]))
        best = max(best, matches / n)
    return best


def _classify_clip(row) -> str:
    match, wer = row["match_score"], row["wer"]
    if match >= 0.80 and wer <= 0.20:
        return "Good"
    elif match >= 0.80 and wer > 0.20:
        return "TooLong"
    elif match < 0.50 and wer <= 0.20:
        return "TooShort"
    else:
        return "BadAlignment"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_audit_with_alignment(audit_path=None) -> pd.DataFrame:
    """Load whisper_audit.csv and add match_score + alignment_type columns."""
    path = Path(audit_path) if audit_path else _DEFAULT_AUDIT
    df = pd.read_csv(path).fillna("")
    df["match_score"] = df.apply(
        lambda r: _best_match_score(r["ref_text"], r["whisper_text"]), axis=1
    )
    df["alignment_type"] = df.apply(_classify_clip, axis=1)
    return df


def print_alignment_stats(df: pd.DataFrame) -> None:
    print(f"Total clips : {len(df)}")
    print()
    print(df["alignment_type"].value_counts().to_string())
    print()
    print(df["alignment_type"].value_counts(normalize=True).mul(100).round(1).to_string())
    print()
    print(f"{'Fallacy':<25} {'Good':>6} {'TooLong':>8} {'TooShort':>9} {'BadAlign':>9} {'WER mean':>9}")
    for cls, grp in df.groupby("fallacy"):
        counts = grp["alignment_type"].value_counts()
        wer_mean = grp["wer"].mean()
        print(f"  {cls:<23} {counts.get('Good',0):>6} {counts.get('TooLong',0):>8} "
              f"{counts.get('TooShort',0):>9} {counts.get('BadAlignment',0):>9} {wer_mean:>9.3f}")


def plot_alignment_overview(df: pd.DataFrame, save_path=None) -> None:
    """4-panel: stacked bar, match score distribution, mean score per class, WER boxplot."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    class_stats = []
    for cls, grp in df.groupby("fallacy"):
        n_al = (grp["alignment_type"] == "Good").sum()
        class_stats.append({
            "class": cls, "n_total": len(grp),
            "n_aligned": n_al, "n_misaligned": len(grp) - n_al,
            "pct_aligned": 100 * n_al / len(grp),
        })
    stats_df = pd.DataFrame(class_stats).sort_values("pct_aligned", ascending=False)
    x = np.arange(len(stats_df))

    axes[0, 0].bar(x, stats_df["n_aligned"],    color="#2ecc71", label="Good",      edgecolor="white")
    axes[0, 0].bar(x, stats_df["n_misaligned"], bottom=stats_df["n_aligned"],
                   color="#e74c3c", label="Misaligned", edgecolor="white")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(stats_df["class"], rotation=45, ha="right")
    axes[0, 0].set_ylabel("Number of clips")
    axes[0, 0].set_title("Aligned vs Misaligned per Class\n(Good = WER≤0.2 + match≥0.8)", fontweight="bold")
    axes[0, 0].legend()
    for i, row in stats_df.reset_index().iterrows():
        axes[0, 0].text(i, row["n_total"] + 1, f"{row['pct_aligned']:.0f}%",
                        ha="center", fontsize=9, color="#2ecc71", fontweight="bold")

    axes[0, 1].hist(df["match_score"], bins=20, color="#3498db", edgecolor="white")
    axes[0, 1].axvline(1.0, color="#2ecc71", linestyle="--", linewidth=2, label="Perfect (1.0)")
    axes[0, 1].axvline(df["match_score"].mean(), color="#e74c3c", linestyle="--", linewidth=2,
                       label=f"Mean: {df['match_score'].mean():.2f}")
    axes[0, 1].set_xlabel("Match Score")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Match Score Distribution", fontweight="bold")
    axes[0, 1].legend()

    mean_scores = df.groupby("fallacy")["match_score"].mean().sort_values(ascending=False)
    colors = ["#2ecc71" if s > 0.7 else "#f39c12" if s > 0.5 else "#e74c3c"
              for s in mean_scores.values]
    axes[1, 0].barh(mean_scores.index, mean_scores.values, color=colors, edgecolor="white")
    axes[1, 0].axvline(df["match_score"].mean(), color="#2c3e50", linestyle="--", linewidth=1.5,
                       label=f"Global mean: {df['match_score'].mean():.2f}")
    axes[1, 0].set_xlabel("Mean Match Score")
    axes[1, 0].set_title("Mean Match Score per Class", fontweight="bold")
    axes[1, 0].legend()

    classes = sorted(df["fallacy"].unique())
    wer_data = [df[df["fallacy"] == c]["wer"].dropna().values for c in classes]
    bp = axes[1, 1].boxplot(wer_data, labels=classes, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#3498db")
        patch.set_alpha(0.7)
    axes[1, 1].set_ylabel("WER")
    axes[1, 1].set_title("WER Distribution per Class", fontweight="bold")
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Audio-Text Alignment Audit — Match Score & WER", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_alignment_4types(df: pd.DataFrame, save_path=None) -> None:
    """Pie chart + stacked bar by fallacy class."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    counts = df["alignment_type"].value_counts()
    colors = [TYPE_COLORS[t] for t in counts.index]
    axes[0].pie(counts.values, labels=counts.index, colors=colors,
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    axes[0].set_title("Global Alignment Type Distribution", fontweight="bold", fontsize=13)

    classes = df["fallacy"].unique()
    types   = ["Good", "TooLong", "TooShort", "BadAlignment"]
    bottom  = np.zeros(len(classes))
    x       = np.arange(len(classes))
    for t in types:
        vals = [(df[(df["fallacy"] == cls) & (df["alignment_type"] == t)]).shape[0]
                for cls in classes]
        axes[1].bar(x, vals, bottom=bottom, color=TYPE_COLORS[t], label=t, edgecolor="white")
        bottom += np.array(vals)
    for i, cls in enumerate(classes):
        grp = df[df["fallacy"] == cls]
        pct = 100 * (grp["alignment_type"] == "Good").mean()
        axes[1].text(i, len(grp) + 2, f"{pct:.0f}%", ha="center",
                     fontsize=9, color="#2ecc71", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes, rotation=45, ha="right")
    axes[1].set_ylabel("Number of clips")
    axes[1].set_title("Alignment Type per Class", fontweight="bold", fontsize=13)
    axes[1].legend(loc="upper right")

    fig.suptitle(
        "Audio Alignment Audit — 4-Type Classification\n"
        "(Good: WER≤0.2+match≥0.8 | TooLong: match≥0.8+WER>0.2 | "
        "TooShort: match<0.5+WER≤0.2 | BadAlignment: other)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_strict_filter(df: pd.DataFrame, save_path=None) -> None:
    """3-panel analysis for WER≤0.2 + match≥0.80 strict filter."""
    mask = (df["match_score"] >= 0.80) & (df["wer"] <= 0.20)
    df_good = df[mask]
    df_bad  = df[~mask]

    class_stats = []
    for cls, grp in df.groupby("fallacy"):
        n_good = mask[df["fallacy"] == cls].sum()
        class_stats.append({
            "class": cls, "n_good": n_good,
            "n_bad": len(grp) - n_good,
            "pct_good": 100 * n_good / len(grp),
        })
    stats_df = pd.DataFrame(class_stats).sort_values("pct_good", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(stats_df))

    axes[0].bar(x, stats_df["n_good"], color="#2ecc71", label="Well aligned", edgecolor="white")
    axes[0].bar(x, stats_df["n_bad"], bottom=stats_df["n_good"],
                color="#e74c3c", label="Misaligned", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats_df["class"], rotation=45, ha="right")
    axes[0].set_ylabel("Number of clips")
    axes[0].set_title("Aligned vs Misaligned\n(WER ≤ 0.2 + match ≥ 0.80)", fontweight="bold")
    axes[0].legend()

    axes[1].hist(df_good["wer"], bins=20, color="#2ecc71", alpha=0.7, label="Well aligned", edgecolor="white")
    axes[1].hist(df_bad["wer"],  bins=20, color="#e74c3c", alpha=0.7, label="Misaligned",   edgecolor="white")
    axes[1].axvline(0.2, color="black", linestyle="--", linewidth=2, label="Threshold 0.2")
    axes[1].set_xlabel("WER")
    axes[1].set_ylabel("Count")
    axes[1].set_title("WER Distribution — Good vs Misaligned", fontweight="bold")
    axes[1].legend()

    colors = ["#2ecc71" if p > 50 else "#f39c12" if p > 35 else "#e74c3c"
              for p in stats_df["pct_good"]]
    axes[2].barh(stats_df["class"], stats_df["pct_good"], color=colors, edgecolor="white")
    axes[2].axvline(100 * len(df_good) / len(df), color="#2c3e50", linestyle="--", linewidth=1.5,
                    label=f"Global: {100*len(df_good)/len(df):.1f}%")
    axes[2].set_xlabel("% Well aligned clips")
    axes[2].set_title("Alignment Rate per Class (WER ≤ 0.2)", fontweight="bold")
    axes[2].legend()

    fig.suptitle("Audio Quality — Strict Filter (WER ≤ 0.2 + match ≥ 0.80)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def show_case_examples(df: pd.DataFrame, n: int = 3) -> None:
    """Print N examples per alignment type."""
    for t in ["Good", "TooLong", "TooShort", "BadAlignment"]:
        subset = df[df["alignment_type"] == t]
        print(f"\n{'='*70}")
        print(f"TYPE: {t}  ({len(subset)} clips)")
        print(f"{'='*70}")
        for _, row in subset.sample(min(n, len(subset)), random_state=42).iterrows():
            print(f"\n  [{row['dialogue_id']} | {row['fallacy']} | "
                  f"match={row['match_score']:.2f} | WER={row['wer']:.2f} | "
                  f"ref={row['ref_len']}w whisper={row['whisper_len']}w]")
            print(f"  REF    : {row['ref_text']}")
            print(f"  WHISPER: {row['whisper_text']}")
