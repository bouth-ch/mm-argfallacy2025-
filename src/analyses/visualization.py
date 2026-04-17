import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from src.evaluation.schema import context_column, label_column, text_column


_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = _ROOT / 'notebooks' / 'figures'


def _require_dialogue(info, role):
    did = info.get('dialogue_id')
    if not did:
        warnings.warn(
            f"Skipping {role}: missing dialogue_id (legacy results?). "
            "Re-run training to store held-out dialogue_ids.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return did


def plot_fold_f1_bar(scores, experiment_name, paper_baseline=None, save=True):
    """Bar chart of per-fold F1 scores with mean ± std band."""
    mean_f1 = np.mean(scores)
    std_f1  = np.std(scores)
    n       = len(scores)
    colors  = ['#e74c3c' if s < 0.25 else '#f39c12' if s < 0.45 else '#2ecc71'
               for s in scores]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(1, n + 1), scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(mean_f1, color='#2c3e50', linestyle='--', linewidth=2)
    ax.fill_between(range(1, n + 1), mean_f1 - std_f1, mean_f1 + std_f1,
                    alpha=0.15, color='#2c3e50')

    handles = [
        mpatches.Patch(color='#e74c3c', label='F1 < 0.25 (poor)'),
        mpatches.Patch(color='#f39c12', label='0.25 ≤ F1 < 0.45 (average)'),
        mpatches.Patch(color='#2ecc71', label='F1 ≥ 0.45 (good)'),
        plt.Line2D([0], [0], color='#2c3e50', linestyle='--',
                   label=f'Mean = {mean_f1:.4f} ± {std_f1:.4f}'),
    ]
    if paper_baseline is not None:
        ax.axhline(paper_baseline, color='#8e44ad', linestyle=':', linewidth=1.5)
        handles.append(plt.Line2D([0], [0], color='#8e44ad', linestyle=':',
                                  label=f'Paper baseline = {paper_baseline}'))

    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.set_xlabel('Fold (left-out dialogue)', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'Per-Fold F1 — {experiment_name} ({n}-fold Leave-One-Dialogue-Out CV)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, n + 1))
    ax.set_ylim(0, 1.05)
    ax.set_facecolor('#f8f9fa')
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / f'{experiment_name}_fold_f1_bar.png',
                    dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Mean: {mean_f1:.4f} ± {std_f1:.4f}" +
          (f"  |  Paper: {paper_baseline}" if paper_baseline else ""))


def plot_confusion_matrix(cm, cm_norm, class_names, experiment_name, save=True):
    """Side-by-side confusion matrix: raw counts and row-normalised."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2f'],
        ['Counts', 'Row-normalised (recall per class)']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f'Confusion Matrix — {title}', fontsize=11, fontweight='bold')
        ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=9)
        ax.set_yticklabels(class_names, rotation=0, fontsize=9)

    plt.suptitle(f'Global Confusion Matrix — {experiment_name} (all folds)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / f'{experiment_name}_confusion_matrix.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_per_class_metrics(class_metrics_df, experiment_name, save=True):
    """Grouped bar chart: precision / recall / F1 per class."""
    class_names = list(class_metrics_df.index)
    x = np.arange(len(class_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, class_metrics_df['precision'], w, label='Precision',
           color='#3498db', alpha=0.85)
    ax.bar(x,     class_metrics_df['recall'],    w, label='Recall',
           color='#2ecc71', alpha=0.85)
    ax.bar(x + w, class_metrics_df['f1-score'],  w, label='F1-score',
           color='#e74c3c', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=40, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Per-Class Precision / Recall / F1 — {experiment_name}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_facecolor('#f8f9fa')

    for i, name in enumerate(class_names):
        ax.text(i, -0.08, f"n={class_metrics_df.loc[name, 'support']}",
                ha='center', fontsize=8, color='#555')

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / f'{experiment_name}_per_class_f1.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_fold_snippet_length(loader, selected_folds, experiment_name, save=True):
    """Boxplot of text-unit length (snippet or sentence) for best / middle / worst folds."""
    import pandas as pd

    tcol = text_column(loader.task_name)
    all_data = []
    for role, info in selected_folds.items():
        did = _require_dialogue(info, role)
        if did is None:
            continue
        df = loader.data[loader.data['dialogue_id'] == did].copy()
        df['text_unit_len'] = df[tcol].str.split().str.len()
        df['role'] = f"{role}\n{did}\nF1={info['score']}"
        all_data.append(df[['text_unit_len', 'role']])

    if not all_data:
        return
    combined = pd.concat(all_data)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=combined, x='role', y='text_unit_len', palette='viridis')
    ylab = 'Snippet length (words)' if loader.task_name == 'afc' else 'Sentence length (words)'
    plt.title('Text length distribution per fold', fontweight='bold')
    plt.ylabel(ylab)
    plt.xlabel('')
    plt.gca().set_facecolor('#f8f9fa')
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / f'{experiment_name}_fold_snippet_length.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_fold_scatter(loader, selected_folds, experiment_name, save=True):
    """Scatter of text-unit length vs context length per fold (AFC or AFD)."""
    colors = {'best': '#2ecc71', 'middle': '#e67e22', 'worst': '#e74c3c'}
    plt.figure(figsize=(10, 6))

    tcol = text_column(loader.task_name)
    ccol = context_column(loader.task_name)
    for role, info in selected_folds.items():
        did = _require_dialogue(info, role)
        if did is None:
            continue
        df = loader.data[loader.data['dialogue_id'] == did].copy()
        df['text_len'] = df[tcol].str.split().str.len()
        df['ctx_len'] = df[ccol].str.split().str.len()
        plt.scatter(df['ctx_len'], df['text_len'],
                    label=f"{role} ({did}, F1={info['score']})",
                    color=colors[role], alpha=0.7, s=60)

    ctx_label = 'Context length (words)'
    text_label = (
        'Snippet length (words)' if loader.task_name == 'afc' else 'Sentence length (words)'
    )
    plt.xlabel(ctx_label, fontsize=12)
    plt.ylabel(text_label, fontsize=12)
    plt.title('Text vs context length per fold', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / f'{experiment_name}_fold_scatter.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_fold_length_stats(loader, selected_folds):
    """Print text-unit and context length statistics for selected folds."""
    tcol = text_column(loader.task_name)
    ccol = context_column(loader.task_name)
    for role, info in selected_folds.items():
        did = _require_dialogue(info, role)
        if did is None:
            continue
        df = loader.data[loader.data['dialogue_id'] == did].copy()
        df['text_len'] = df[tcol].str.split().str.len()
        df['ctx_len'] = df[ccol].str.split().str.len()
        print(f"\n{role.upper()} — {did}")
        print(f"  Text length:   mean={df['text_len'].mean():.1f}, "
              f"min={df['text_len'].min()}, max={df['text_len'].max()}")
        print(f"  Context length: mean={df['ctx_len'].mean():.1f}, "
              f"min={df['ctx_len'].min()}, max={df['ctx_len'].max()}")


def plot_fold_class_distribution(loader, selected_folds, label_names, experiment_name, save=True):
    """Bar charts of class distribution for best / middle / worst folds."""
    lcol = label_column(loader.task_name)
    order = ['best', 'middle', 'worst']
    triple = []
    for role in order:
        info = selected_folds[role]
        did = _require_dialogue(info, role)
        if did is None:
            return
        triple.append((role, info, did))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (role, info, did) in zip(axes, triple):
        df = loader.data[loader.data['dialogue_id'] == did]
        counts = df[lcol].value_counts().sort_index()
        names = [label_names[int(i)] for i in counts.index]
        ax.bar(names, counts.values, color=sns.color_palette('viridis', len(counts)))
        ax.set_title(f"{role.upper()} — {did}\nF1={info['score']}",
                     fontweight='bold')
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_facecolor('#f8f9fa')

    plt.suptitle(f'Class Distribution — 3 Selected Folds ({experiment_name})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / f'{experiment_name}_fold_class_dist.png',
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_snippets_for_top_confusion_pairs(
    cm_counts,
    loader,
    label_names: dict[int, str],
    experiment_name: str,
    *,
    top_pairs: int = 8,
    snippets_per_pair: int = 3,
    save: bool = True,
):
    """
    Qualitative view for **off-diagonal** confusion: for each frequent (true → pred)
    pair, show a few **gold-true** snippets from the dataset (not necessarily the exact
    OOF misclassified rows unless OOF order matches ``results.json``).

    Useful to eyeball **linguistic resemblance** between snippets of a gold class that
    the model often maps to another label (see confusion matrix counts in titles).
    """
    import textwrap

    n = cm_counts.shape[0]
    lcol = label_column(loader.task_name)
    pairs: list[tuple[int, int, int]] = []
    for t in range(n):
        for p in range(n):
            if t != p and cm_counts[t, p] > 0:
                pairs.append((int(cm_counts[t, p]), t, p))
    pairs.sort(reverse=True)
    pairs = pairs[:top_pairs]
    if not pairs:
        print('No off-diagonal confusion mass to plot.')
        return

    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, (cnt, t, p) in zip(axes, pairs):
        pool = loader.data[loader.data[lcol] == t]
        k = min(snippets_per_pair, len(pool))
        if k == 0:
            ax.text(0.5, 0.5, f'No samples for gold class {t}', ha='center', va='center')
            ax.axis('off')
            continue
        sample = pool.sample(k, random_state=42)
        tcol = 'snippet' if loader.task_name == 'afc' else 'sentence'
        blocks = []
        for _, row in sample.iterrows():
            raw = str(row[tcol])
            short = raw if len(raw) <= 420 else raw[:417] + '...'
            blocks.append(textwrap.fill(short, width=100))
        body = '\n\n—\n\n'.join(blocks)
        ax.text(
            0.02, 0.98, body, transform=ax.transAxes, va='top', ha='left',
            fontsize=8, family='sans-serif', linespacing=1.35,
        )
        ax.axis('off')
        ax.set_title(
            f'{cnt}×  true={label_names[t]}  →  pred={label_names[p]}  '
            f'(showing {k} random gold-{label_names[t]} snippets)',
            fontsize=10, fontweight='bold', loc='left',
        )

    plt.suptitle(
        f'Top confusion pairs — {experiment_name}\n'
        f'(snippets are gold-true class only; counts from confusion matrix)',
        fontsize=11, fontweight='bold', y=1.01,
    )
    plt.tight_layout()
    if save:
        plt.savefig(
            FIGURES_DIR / f'{experiment_name}_confusion_pair_snippets.png',
            dpi=150, bbox_inches='tight',
        )
    plt.show()


def plot_oof_wrong_snippet_table(
    wrong_df,
    experiment_name: str,
    *,
    max_rows: int = 24,
    max_chars: int = 200,
    save: bool = True,
):
    """
    Table-style figure of **actual** OOF misclassifications (needs aligned
    ``build_oof_misclassification_dataframe``).
    """
    import textwrap

    if wrong_df.empty:
        print('No wrong predictions in dataframe.')
        return
    show = wrong_df.head(max_rows).copy()
    show['snippet_short'] = show['snippet'].str.slice(0, max_chars).apply(
        lambda s: textwrap.fill(s + ('…' if len(str(s)) >= max_chars else ''), width=70)
    )

    fig_h = min(0.45 * len(show) + 1.5, 30)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis('off')
    rows = []
    for _, r in show.iterrows():
        rows.append(
            f"True: {r['true_name']}  |  Pred: {r['pred_name']}\n{r['snippet_short']}"
        )
    ax.text(
        0.02, 0.98, '\n\n'.join(rows), transform=ax.transAxes, va='top', ha='left',
        fontsize=8, family='sans-serif',
    )
    ax.set_title(
        f'OOF misclassified snippets (first {len(show)} of {len(wrong_df)}) — {experiment_name}',
        fontsize=11, fontweight='bold', loc='left', pad=12,
    )
    plt.tight_layout()
    if save:
        plt.savefig(
            FIGURES_DIR / f'{experiment_name}_oof_wrong_snippets.png',
            dpi=150, bbox_inches='tight',
        )
    plt.show()
