"""
CLI: Longformer-base-4096 + dialogue context on AFC, same 5-fold setup.

Goal: verify whether RoBERTa's 512-token limit explains the context degradation
observed (roberta_afc_context=0.352 vs roberta_afc=0.400).
Longformer supports up to 4096 tokens — context prepended to snippet should
not be truncated.

Text input: 3 preceding dialogue sentences + snippet (ContextMMUSEDFallacy).
Audio: none (text-only experiment).

From project root:
    python scripts/run_longformer_afc_context.py

Background:
    nohup python scripts/run_longformer_afc_context.py > logs/longformer_afc_context.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamkit.data.datasets import InputMode

from src.configs.text_configs import get_longformer_afc_context_config
from src.configs.fold_selection import MULTIMODAL_TEST_DIALOGUES
from src.data.context_dataset import ContextMMUSEDFallacy
from src.experiments.mmused_text import prepare_text_reproducibility, run_mmused_text_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_text_reproducibility(seed=42)

config = get_longformer_afc_context_config()

loader = ContextMMUSEDFallacy(
    task_name='afc',
    input_mode=InputMode.TEXT_ONLY,
    base_data_path=ROOT / 'data',
)

sample = loader.data.iloc[0]
print(f"Starting experiment: {config['name']}")
print(f"Model : {config['model_card']} (max 4096 tokens)")
print(f"Text  : dialogue context (3 sentences) + snippet")
print(f"Audio : none")
print(f"Example text input[:120]: {sample['snippet'][:120]!r}")
print()

summary = run_mmused_text_cv(
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
