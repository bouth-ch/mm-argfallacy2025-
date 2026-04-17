"""
CLI: wavlm_roberta_afc (CrossEntropy pondérée).

From project root:
    python scripts/run_wavlm_roberta_afc.py

Background:
    nohup python scripts/run_wavlm_roberta_afc.py > logs/wavlm_roberta_afc.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.multimodal_configs import get_wavlm_roberta_afc_config
from src.configs.fold_selection import MULTIMODAL_TEST_DIALOGUES
from src.experiments.mmused_multimodal import prepare_multimodal_reproducibility, run_mmused_multimodal_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_multimodal_reproducibility(seed=42)
config = get_wavlm_roberta_afc_config()

print(f"Starting experiment: {config['name']}")
print("Text  : RoBERTa-base (trainable)")
print("Audio : WavLM-base → BiLSTM(128)")
print("Loss  : CrossEntropy pondérée\n")

summary = run_mmused_multimodal_cv(config, save_bm3_checkpoints_after=False,
                                   test_dialogues=MULTIMODAL_TEST_DIALOGUES)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
