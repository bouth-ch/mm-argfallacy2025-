"""
CLI: roberta_afc_focal (Focal + weighted sampler + BM3 checkpoints).

From project root:
    python scripts/run_balanced_afc.py

Background:
    nohup python scripts/run_balanced_afc.py > logs/balanced_afc.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.text_configs import get_roberta_afc_focal_config
from src.experiments.mmused_text import prepare_text_reproducibility, run_mmused_text_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_text_reproducibility(seed=42)
config = get_roberta_afc_focal_config()

print(f"Starting experiment: {config['name']}")
print("Loss: FocalLoss(gamma=2) | WeightedRandomSampler: ON")
print("Checkpoints: BM3 after full CV (3 folds on disk).\n")

summary = run_mmused_text_cv(config, save_bm3_checkpoints_after=True)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
