"""Reproducible experiment entry points (MAMKit-backed)."""

from src.experiments.mmused_text import (
    DEFAULT_DATA_PATH,
    DEFAULT_RESULTS_PATH,
    make_mmused_fallacy_loader,
    prepare_text_reproducibility,
    run_mmused_text_cv,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_RESULTS_PATH",
    "make_mmused_fallacy_loader",
    "prepare_text_reproducibility",
    "run_mmused_text_cv",
]
