"""
CLI: RoBERTa AFD (binary sentence-level) — same run as the AFD section in
``notebooks/02_text_baseline.ipynb`` (35-fold CV + BM3).

AFD has ~17k samples; expect **much** longer runtime than AFC.

From project root::

    python scripts/run_roberta_afd.py

Background::

    mkdir -p logs
    nohup ./mmarg_env/bin/python scripts/run_roberta_afd.py > logs/roberta_afd.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.text_configs import get_roberta_afd_config
from src.experiments.mmused_text import (
    make_mmused_fallacy_loader,
    prepare_text_reproducibility,
    run_mmused_text_cv,
)

Path(ROOT / "logs").mkdir(exist_ok=True)

prepare_text_reproducibility(seed=42)
loader_afd = make_mmused_fallacy_loader("afd")
config = get_roberta_afd_config()

print(f"Starting experiment: {config['name']}")
print("Task: AFD (binary) | CV: mancini-et-al-2024 | BM3 after full CV.\n")

summary = run_mmused_text_cv(
    config,
    loader=loader_afd,
    save_bm3_checkpoints_after=True,
)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean    : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
