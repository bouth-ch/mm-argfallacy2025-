"""
CLI: RoBERTa AFC — same run as ``notebooks/02_text_baseline.ipynb`` (CV + BM3).

Use this instead of the notebook when you need the job to survive SSH disconnects.
The notebook kernel can stop when your editor session ends; this process does not.

From project root::

    python scripts/run_roberta_afc.py

Background with log file::

    mkdir -p logs
    nohup python scripts/run_roberta_afc.py > logs/roberta_afc.log 2>&1 &

tmux (reattach after reconnect: ``tmux attach -t roberta_afc``)::

    tmux new -s roberta_afc
    python scripts/run_roberta_afc.py
    # Ctrl+B then D to detach
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.text_configs import get_roberta_afc_config
from src.experiments.mmused_text import (
    make_mmused_fallacy_loader,
    prepare_text_reproducibility,
    run_mmused_text_cv,
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_text_reproducibility(seed=42)
loader_afc = make_mmused_fallacy_loader("afc")
config = get_roberta_afc_config()

print(f"Starting experiment: {config['name']}")
print("Task: AFC | CV: mancini-et-al-2024 | BM3 checkpoints after full CV.\n")

summary = run_mmused_text_cv(
    config,
    loader=loader_afc,
    save_bm3_checkpoints_after=True,
)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
