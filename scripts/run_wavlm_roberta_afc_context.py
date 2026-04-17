"""
CLI: wavlm_roberta_afc with dialogue context prepended to snippet text.

Preceding 3 sentences are concatenated with the snippet so RoBERTa
attends over the full local context in one pass. Audio is unchanged.

From project root:
    python scripts/run_wavlm_roberta_afc_context.py

Background:
    nohup python scripts/run_wavlm_roberta_afc_context.py > logs/wavlm_roberta_afc_context.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.multimodal_configs import get_wavlm_roberta_afc_context_config
from src.configs.fold_selection import MULTIMODAL_TEST_DIALOGUES
from src.data.context_dataset import ContextMMUSEDFallacy
from src.experiments.mmused_multimodal import prepare_multimodal_reproducibility, run_mmused_multimodal_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_multimodal_reproducibility(seed=42)

config = get_wavlm_roberta_afc_context_config()

loader = ContextMMUSEDFallacy(
    task_name='afc',
    base_data_path=ROOT / 'data',
)

# Sanity check — show one example
sample = loader.data.iloc[0]
print(f"Starting experiment: {config['name']}")
print(f"Text  : RoBERTa-base (dialogue context + snippet)")
print(f"Audio : WavLM-base (original clips)")
print(f"Example text input[:120]: {sample['snippet'][:120]!r}")
print()

summary = run_mmused_multimodal_cv(
    config,
    loader=loader,
    save_bm3_checkpoints_after=False,
    test_dialogues=MULTIMODAL_TEST_DIALOGUES,
)

print(f"\n{'='*50}")
print(f"DONE — {config['name']}")
print(f"Mean F1 : {summary['mean']:.4f}")
print(f"Std     : {summary['std']:.4f}")
print(f"Folds   : {len(summary['scores'])}")
print(f"{'='*50}")
