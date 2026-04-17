"""
CLI: wavlm_roberta_afc with trimmed audio clips.

Audio clips are trimmed to match the annotated snippet exactly.
Text input is unchanged (original MAMKit snippet).

From project root:
    python scripts/run_wavlm_roberta_afc_trimmed.py

Background:
    nohup python scripts/run_wavlm_roberta_afc_trimmed.py > logs/wavlm_roberta_afc_trimmed.log 2>&1 &
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.configs.multimodal_configs import get_wavlm_roberta_afc_config
from src.configs.fold_selection import MULTIMODAL_TEST_DIALOGUES
from src.data.trimmed_dataset import TrimmedMMUSEDFallacy
from src.experiments.mmused_multimodal import prepare_multimodal_reproducibility, run_mmused_multimodal_cv

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

prepare_multimodal_reproducibility(seed=42)

config = get_wavlm_roberta_afc_config()
config['name'] = 'wavlm_roberta_afc_trimmed'

loader = TrimmedMMUSEDFallacy(
    task_name='afc',
    base_data_path=ROOT / 'data',
)

n_trimmed = sum(
    1 for paths in loader.data['snippet_paths']
    for p in paths
    if 'audio_clips_trimmed' in str(p)
)
print(f"Starting experiment: {config['name']}")
print(f"Text  : RoBERTa-base (original snippet)")
print(f"Audio : WavLM-base — trimmed clips ({n_trimmed} trimmed paths)")
print(f"Loss  : CrossEntropy pondérée\n")

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
