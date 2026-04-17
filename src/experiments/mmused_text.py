"""Text-only CV experiments for MM-USED-Fallacy using MAMKit."""

import os
from pathlib import Path
from typing import Any

import lightning as L
import torch

from mamkit.data.datasets import MMUSEDFallacy, InputMode

from src.training.trainer import TextTrainer
from src.utils.results import ResultsManager

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = _ROOT / "data"
DEFAULT_RESULTS_PATH = _ROOT / "results" / "results.json"
DEFAULT_HF_CACHE = _ROOT / "data" / "hf_cache"


def prepare_text_reproducibility(*, seed=42, hf_cache=DEFAULT_HF_CACHE, matmul_precision="medium"):
    """Call once before anything else to set seeds, HF cache, and matmul precision."""
    if hf_cache is not None:
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    torch.set_float32_matmul_precision(matmul_precision)
    L.seed_everything(int(seed), workers=True)


def make_mmused_fallacy_loader(task_name, *, base_data_path=None, input_mode=InputMode.TEXT_ONLY):
    """Build the MAMKit loader for MM-USED-Fallacy. task_name is 'afc' or 'afd'."""
    return MMUSEDFallacy(
        task_name=task_name,
        input_mode=input_mode,
        base_data_path=Path(base_data_path) if base_data_path else DEFAULT_DATA_PATH,
    )


def run_mmused_text_cv(config, *, loader=None, results_manager=None, results_path=None,
                       task_name=None, save_bm3_checkpoints_after=False,
                       max_folds=None, save_checkpoint_folds=None, test_dialogues=None):
    """Run leave-one-dialogue-out CV for a given text config dict."""
    tn = task_name or config.get("task_name")
    if tn not in ("afc", "afd"):
        raise ValueError(f"task_name must be 'afc' or 'afd', got {tn!r}")

    if loader is None:
        loader = make_mmused_fallacy_loader(tn)

    rpath = Path(results_path) if results_path else DEFAULT_RESULTS_PATH
    rm = results_manager or ResultsManager(str(rpath))
    trainer = TextTrainer(config, rm)

    return trainer.run_experiment(
        loader,
        max_folds=max_folds,
        save_checkpoint_folds=save_checkpoint_folds,
        save_bm3_checkpoints_after=save_bm3_checkpoints_after,
        test_dialogues=test_dialogues,
    )
