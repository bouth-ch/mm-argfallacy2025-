"""
WavLM + RoBERTa AFC — 35 folds (LODO complet).

Background:
    nohup python scripts/run_wavlm_roberta_afc_35folds.py > logs/wavlm_roberta_afc_35folds.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.multimodal_configs import get_wavlm_roberta_afc_config
from src.experiments.mmused_multimodal import prepare_multimodal_reproducibility, run_mmused_multimodal_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_multimodal_reproducibility(seed=42)
config = get_wavlm_roberta_afc_config()
config["name"] = "wavlm_roberta_afc_35folds"

print(f"Starting experiment: {config['name']}")
print("Text  : RoBERTa-base (trainable)")
print("Audio : WavLM-base → BiLSTM(128)")
print("Loss  : CrossEntropy pondérée")
print("Folds : 35 (LODO complet)\n")

summary = run_mmused_multimodal_cv(config, save_bm3_checkpoints_after=False,
                                   test_dialogues=None)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
