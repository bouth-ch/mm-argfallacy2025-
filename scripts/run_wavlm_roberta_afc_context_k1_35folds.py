"""
WavLM + RoBERTa AFC — contexte ciblé k=1 (phrase N-1 uniquement), 35 folds LODO.

Seule la phrase précédente immédiate est prépendée au snippet.
Tous les autres hyperparamètres sont identiques à la baseline multimodale
(wavlm_roberta_afc) : lr=1e-5, batch_size=8, patience=5, dropout=0.1,
weighted cross-entropy loss.

Background:
    nohup python scripts/run_wavlm_roberta_afc_context_k1_35folds.py \
        > logs/wavlm_roberta_afc_context_k1_35folds.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.multimodal_configs import get_wavlm_roberta_afc_config
from src.data.context_dataset import ContextMMUSEDFallacy
from src.experiments.mmused_multimodal import prepare_multimodal_reproducibility, run_mmused_multimodal_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_multimodal_reproducibility(seed=42)

config = get_wavlm_roberta_afc_config()
config["name"] = "wavlm_roberta_afc_context_k1_35folds"

loader = ContextMMUSEDFallacy(
    task_name='afc',
    base_data_path=ROOT / 'data',
    k=1,
)

# Sanity check
sample = loader.data.iloc[0]
print(f"Starting experiment: {config['name']}")
print(f"Context k=1 (N-1 sentence only)")
print(f"Text  : RoBERTa-base (N-1 context + snippet)")
print(f"Audio : WavLM-base (original clips)")
print(f"Folds : 35 (LODO complet)")
print(f"Example text input[:150]: {sample['snippet'][:150]!r}")
print()

summary = run_mmused_multimodal_cv(
    config,
    loader=loader,
    save_bm3_checkpoints_after=False,
    test_dialogues=None,
)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
