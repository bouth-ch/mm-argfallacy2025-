"""Aligns stored OOF predictions with their original text snippets for error analysis."""

import warnings
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.evaluation.metrics import load_results
from src.evaluation.schema import label_column
from src.training.trainer import TextTrainer
from src.utils.results import ResultsManager
from src.utils.splits import sort_ldocv_splits


def collect_oof_decoded_snippets_and_labels(loader, config):
    """
    Decode OOF text from token ids in the same order as training stored predictions.
    Replicates the exact seed + split order from TextTrainer so everything lines up.
    """
    seed = int(config.get('seed', 42))
    # Same as BaseTrainer._seed_before_split_draws() + list(get_splits(...))
    L.seed_everything(seed, workers=True)

    _project_root = Path(__file__).resolve().parents[2]
    cache = config.get('hf_cache', str(_project_root / 'data' / 'hf_cache'))
    tokenizer = AutoTokenizer.from_pretrained(config['model_card'], cache_dir=str(cache))
    dummy_rm = ResultsManager(str(Path('/tmp/_oof_snippets_unused_results.json')))
    trainer = TextTrainer(config, dummy_rm)

    snippets: list[str] = []
    labels: list[int] = []
    splits = list(loader.get_splits('mancini-et-al-2024'))
    splits = sort_ldocv_splits(loader, splits)
    for split_info in splits:
        # Must match TextTrainer.run_fold: seed every fold before building loaders.
        L.seed_everything(seed, workers=True)
        dl = trainer.build_dataloader(split_info.test, shuffle=False)
        for batch in dl:
            inp, lbs = batch
            texts = tokenizer.batch_decode(inp['inputs'], skip_special_tokens=True)
            for t, lab in zip(texts, lbs.cpu().numpy().tolist()):
                snippets.append(t)
                labels.append(int(lab))
    return snippets, labels


def build_oof_misclassification_dataframe(experiment_name, loader, config, label_names, results_path=None):
    """
    One row per OOF test example: snippet, true label, predicted label, and correct flag.
    Raises if decoded order doesn't match results.json (predictions can't be aligned safely).
    """
    if results_path is None:
        results_path = Path(__file__).resolve().parents[2] / 'results' / 'results.json'
    snippets, y_dl = collect_oof_decoded_snippets_and_labels(loader, config)
    res = load_results(experiment_name, str(results_path))
    preds = res['predictions']
    y_js = res['true_labels']
    if len(snippets) != len(preds) or len(preds) != len(y_js):
        raise ValueError(
            f"Length mismatch: snippets={len(snippets)}, preds={len(preds)}, "
            f"true_labels={len(y_js)}. "
            "Full OOF should equal the dataset size (e.g. AFD 17118). "
            "Re-run CV with the current code (sorted LDOCV splits + per-fold seed in OOF decode); "
            "partial results cannot be aligned snippet-by-snippet."
        )
    if y_dl != y_js:
        raise ValueError(
            "OOF alignment failed: gold labels from the current DataLoader order do not "
            "match results.json 'true_labels'. Re-run full CV on this environment, or "
            "use plot_snippets_for_top_confusion_pairs() for qualitative gold-class text."
        )

    df = pd.DataFrame(
        {
            'snippet': snippets,
            'true_label': y_js,
            'pred_label': preds,
            'true_name': [label_names[int(i)] for i in y_js],
            'pred_name': [label_names[int(i)] for i in preds],
        }
    )
    df['correct'] = df['true_label'] == df['pred_label']
    return df


def mean_pairwise_tfidf_cosine(texts: list[str]) -> float:
    """Mean cosine similarity over unique pairs (diagnostic for snippet resemblance)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(texts) < 2:
        return float('nan')
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return float('nan')
    sim = cosine_similarity(X)
    iu = np.triu_indices(len(texts), k=1)
    vals = sim[iu]
    return float(np.mean(vals)) if len(vals) else float('nan')
