import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS = str(_ROOT / 'results' / 'results.json')
_DEFAULT_DATA = _ROOT / 'data' / 'MMUSED-fallacy' / 'dataset.pkl'

FALLACY_NAMES = {
    0: 'AdHominem',
    1: 'AppealToAuthority',
    2: 'AppealToEmotion',
    3: 'FalseCause',
    4: 'Slipperyslope',
    5: 'Slogans'
}


def _rebuild_text_scores(results):
    """Reconstruct per-dialogue text scores using sort_ldocv_splits."""
    from utils.splits import infer_held_out_dialogue_id, sort_ldocv_splits
    from mamkit.data.datasets import MMUSEDFallacy, InputMode

    txt = results.get('roberta_afc', {})
    scores = txt.get('scores', [])

    lookup = {}
    if txt.get('dialogue_ids') and any(d is not None for d in txt['dialogue_ids']):
        for did, score in zip(txt['dialogue_ids'], scores):
            if did is not None:
                lookup[did] = score
        return lookup

    # fallback: reconstruct via sorted splits
    loader = MMUSEDFallacy(task_name='afc', input_mode=InputMode.TEXT_ONLY,
                           base_data_path=_ROOT / 'data')
    splits = list(loader.get_splits('mancini-et-al-2024'))
    splits = sort_ldocv_splits(loader, splits)
    for sp, score in zip(splits, scores):
        did = infer_held_out_dialogue_id(loader, sp)
        lookup[did] = score
    return lookup


def _get_filtered_preds(results, exp_name, selected):
    """Extract predictions/true_labels for selected dialogues only."""
    exp = results[exp_name]
    dids = exp.get('dialogue_ids', [])
    preds = exp.get('predictions', [])
    trues = exp.get('true_labels', [])

    df_pkl = pd.read_pickle(_DEFAULT_DATA)
    sizes = {did: len(df_pkl[df_pkl['dialogue_id'] == did]) for did in dids}

    f_preds, f_trues = [], []
    offset = 0
    for did in dids:
        size = sizes.get(did, 0)
        if did in selected:
            f_preds.extend(preds[offset:offset + size])
            f_trues.extend(trues[offset:offset + size])
        offset += size

    return np.array(f_preds), np.array(f_trues)


def load_comparison(results_path=None, selected_dialogues=None):
    """Load results.json and return a per-fold comparison dataframe."""
    path = results_path or _DEFAULT_RESULTS
    with open(path) as f:
        results = json.load(f)

    df = pd.read_pickle(_DEFAULT_DATA)
    n_classes = df.groupby('dialogue_id')['fallacy'].nunique()
    n_samples = df.groupby('dialogue_id').size()

    txt_lookup = _rebuild_text_scores(results)

    mm = results.get('wavlm_roberta_afc', {})
    focal = results.get('wavlm_roberta_afc_focal', {})
    focal_lookup = dict(zip(focal.get('dialogue_ids', []), focal.get('scores', [])))

    rows = []
    for did, mm_score in zip(mm.get('dialogue_ids', []), mm.get('scores', [])):
        year = int(did.split('_')[-1]) if did else None
        rows.append({
            'dialogue_id': did,
            'year': year,
            'n_classes': n_classes.get(did),
            'n_samples': n_samples.get(did),
            'roberta_afc': txt_lookup.get(did),
            'wavlm_roberta_afc': mm_score,
            'wavlm_roberta_afc_focal': focal_lookup.get(did),
        })

    result_df = pd.DataFrame(rows)
    if selected_dialogues:
        result_df = result_df[result_df['dialogue_id'].isin(selected_dialogues)]

    return result_df.reset_index(drop=True)


def plot_f1_comparison(df):
    """Bar plot comparing text vs multimodal F1 per fold."""
    models = ['roberta_afc', 'wavlm_roberta_afc']
    if df['wavlm_roberta_afc_focal'].notna().any():
        models.append('wavlm_roberta_afc_focal')

    x = np.arange(len(df))
    width = 0.25 if len(models) == 3 else 0.35
    labels = ['RoBERTa (text)', 'WavLM+RoBERTa', 'WavLM+RoBERTa (focal)']
    colors = ['steelblue', 'darkorange', 'green']

    _, ax = plt.subplots(figsize=(11, 5))
    for i, (model, label, color) in enumerate(zip(models, labels, colors)):
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, df[model], width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['dialogue_id']}\n({row['year']})" for _, row in df.iterrows()])
    ax.set_ylabel("Macro F1")
    ax.set_title("Text vs Multimodal — F1 per Fold")
    ax.legend()
    ax.axhline(df['roberta_afc'].mean(), color='steelblue', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.axhline(df['wavlm_roberta_afc'].mean(), color='darkorange', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.show()


def print_summary_table(df):
    """Print mean/std comparison table."""
    models = {
        'RoBERTa (text)': 'roberta_afc',
        'WavLM+RoBERTa': 'wavlm_roberta_afc',
        'WavLM+RoBERTa (focal)': 'wavlm_roberta_afc_focal',
    }
    print(f"\n{'='*55}")
    print(f"{'Model':<25} {'Mean F1':>8} {'Std':>8} {'Folds':>6}")
    print(f"{'='*55}")
    for name, col in models.items():
        vals = df[col].dropna()
        if len(vals) > 0:
            print(f"{name:<25} {vals.mean():>8.4f} {vals.std():>8.4f} {len(vals):>6}")
    print(f"{'='*55}")


def plot_per_class_accuracy(df, results_path=None):
    """Per-class accuracy for multimodal model."""
    path = results_path or _DEFAULT_RESULTS
    with open(path) as f:
        results = json.load(f)

    selected = set(df['dialogue_id'].tolist())
    f_preds, f_trues = _get_filtered_preds(results, 'wavlm_roberta_afc', selected)

    per_class = {}
    counts = {}
    for cls, name in FALLACY_NAMES.items():
        mask = f_trues == cls
        if mask.sum() > 0:
            per_class[name] = (f_preds[mask] == cls).mean()
            counts[name] = mask.sum()

    _, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(per_class.keys(), per_class.values(), color='darkorange', alpha=0.85)
    for bar, (name, count) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("WavLM+RoBERTa — Per-class Accuracy")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(df, results_path=None):
    """Confusion matrix for multimodal model."""
    path = results_path or _DEFAULT_RESULTS
    with open(path) as f:
        results = json.load(f)

    selected = set(df['dialogue_id'].tolist())
    f_preds, f_trues = _get_filtered_preds(results, 'wavlm_roberta_afc', selected)

    n = len(FALLACY_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(f_trues, f_preds):
        cm[t][p] += 1

    _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = list(FALLACY_NAMES.values())
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("WavLM+RoBERTa — Confusion Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=9,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.show()
def plot_f1_vs_fold_properties(df):
    """F1 vs fold properties: n_samples, n_classes, snippet length."""
    df2 = df.copy()
    
    # add mean snippet length per dialogue
    dataset = pd.read_pickle(_DEFAULT_DATA)
    dataset['snippet_len'] = dataset['snippet_sentences'].apply(len)
    mean_len = dataset.groupby('dialogue_id')['snippet_len'].mean()
    df2['mean_snippet_len'] = df2['dialogue_id'].map(mean_len)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # F1 vs nb samples
    axes[0].scatter(df2['n_samples'], df2['wavlm_roberta_afc'], color='darkorange', s=80)
    for _, row in df2.iterrows():
        axes[0].annotate(row['dialogue_id'], (row['n_samples'], row['wavlm_roberta_afc']),
                         fontsize=7, ha='left')
    axes[0].set_xlabel("Nb samples in fold")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("F1 vs Fold Size")

    # F1 vs nb classes
    axes[1].scatter(df2['n_classes'], df2['wavlm_roberta_afc'], color='darkorange', s=80)
    for _, row in df2.iterrows():
        axes[1].annotate(row['dialogue_id'], (row['n_classes'], row['wavlm_roberta_afc']),
                         fontsize=7, ha='left')
    axes[1].set_xlabel("Nb unique classes")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("F1 vs Class Diversity")

    # F1 vs mean snippet length
    axes[2].scatter(df2['mean_snippet_len'], df2['wavlm_roberta_afc'], color='darkorange', s=80)
    for _, row in df2.iterrows():
        axes[2].annotate(row['dialogue_id'], (row['mean_snippet_len'], row['wavlm_roberta_afc']),
                         fontsize=7, ha='left')
    axes[2].set_xlabel("Mean snippet length (sentences)")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("F1 vs Snippet Length")

    plt.suptitle("WavLM+RoBERTa — F1 vs Fold Properties")
    plt.tight_layout()
    plt.show()

def plot_error_distribution(df, results_path=None):
    """Which classes are most confused."""
    path = results_path or _DEFAULT_RESULTS
    with open(path) as f:
        results = json.load(f)

    selected = set(df['dialogue_id'].tolist())
    f_preds, f_trues = _get_filtered_preds(results, 'wavlm_roberta_afc', selected)

    errors = f_preds[f_preds != f_trues]
    true_errors = f_trues[f_preds != f_trues]

    error_counts = {FALLACY_NAMES[c]: (true_errors == c).sum() for c in FALLACY_NAMES}
    total_counts = {FALLACY_NAMES[c]: (f_trues == c).sum() for c in FALLACY_NAMES}
    error_rate = {k: error_counts[k] / total_counts[k] if total_counts[k] > 0 else 0
                  for k in error_counts}

    _, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(error_rate.keys(), error_rate.values(), color='crimson', alpha=0.8)
    for bar, (k, v) in zip(bars, error_counts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{v} err', ha='center', fontsize=8)
    ax.set_ylabel("Error Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("WavLM+RoBERTa — Error Rate per Class")
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()


def plot_f1_per_class_per_fold(df, results_path=None):
    """F1 per class per fold heatmap."""
    path = results_path or _DEFAULT_RESULTS
    with open(path) as f:
        results = json.load(f)

    exp = results['wavlm_roberta_afc']
    dids = exp.get('dialogue_ids', [])
    preds = exp.get('predictions', [])
    trues = exp.get('true_labels', [])
    selected = df['dialogue_id'].tolist()

    dataset = pd.read_pickle(_DEFAULT_DATA)
    sizes = {did: len(dataset[dataset['dialogue_id'] == did]) for did in dids}

    matrix = np.full((len(selected), len(FALLACY_NAMES)), np.nan)
    offset = 0
    for did in dids:
        size = sizes.get(did, 0)
        if did in selected:
            fold_idx = selected.index(did)
            fp = np.array(preds[offset:offset + size])
            ft = np.array(trues[offset:offset + size])
            for cls, name in FALLACY_NAMES.items():
                mask = ft == cls
                if mask.sum() > 0:
                    matrix[fold_idx, cls] = (fp[mask] == cls).mean()
        offset += size

    _, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(FALLACY_NAMES)))
    ax.set_xticklabels(FALLACY_NAMES.values(), rotation=20, ha='right', fontsize=8)
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels(selected, fontsize=8)
    ax.set_title("WavLM+RoBERTa — Accuracy per Class per Fold")
    plt.colorbar(im, ax=ax)
    for i in range(len(selected)):
        for j in range(len(FALLACY_NAMES)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center', fontsize=7)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=7, color='gray')
    plt.tight_layout()
    plt.show()


def plot_audio_duration_vs_f1(df, results_path=None):
    """Audio clip duration vs F1 per fold."""
    path = results_path or _DEFAULT_RESULTS
    with open(path) as f:
        results = json.load(f)

    dataset = pd.read_pickle(_DEFAULT_DATA)
    dataset['duration'] = (
        dataset['snippet_end_time'].apply(lambda x: x[-1])
        - dataset['snippet_start_time'].apply(lambda x: x[0])
    )
    mean_dur = dataset.groupby('dialogue_id')['duration'].mean().clip(upper=60)

    df2 = df.copy()
    df2['mean_duration'] = df2['dialogue_id'].map(mean_dur)

    _, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df2['mean_duration'], df2['wavlm_roberta_afc'], color='darkorange', s=80)
    for _, row in df2.iterrows():
        ax.annotate(row['dialogue_id'], (row['mean_duration'], row['wavlm_roberta_afc']),
                    fontsize=7, ha='left')
    ax.set_xlabel("Mean audio duration per snippet (s, capped at 60s)")
    ax.set_ylabel("Macro F1")
    ax.set_title("WavLM+RoBERTa — Audio Duration vs F1")
    plt.tight_layout()
    plt.show()
