"""
CLI: Longformer-base-4096 on AFD (full 35-fold LDOCV).

From project root:
    python scripts/run_longformer_afd.py

Background:
    nohup python scripts/run_longformer_afd.py > logs/longformer_afd.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.text_configs import get_longformer_afd_config
from src.configs.fold_selection import MULTIMODAL_TEST_DIALOGUES
from src.experiments.mmused_text import (
    make_mmused_fallacy_loader,
    prepare_text_reproducibility,
    run_mmused_text_cv,
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_text_reproducibility(seed=42)

config = get_longformer_afd_config()
loader = make_mmused_fallacy_loader('afd')

print(f"Starting experiment: {config['name']}")
print(f"Model : {config['model_card']}")
print("Task  : AFD | 5 folds\n")

summary = run_mmused_text_cv(config, loader=loader, save_bm3_checkpoints_after=False,
                             test_dialogues=MULTIMODAL_TEST_DIALOGUES)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
