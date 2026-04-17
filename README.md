# MM-ArgFallacy2025 — Reproduction & Extension

Reproduction of baselines and methodological extensions for the MM-ArgFallacy2025 shared task (ACL 2025). The dataset is MM-USED-Fallacy: 1278 clips from US presidential debates. Two tasks: AFC (6-class fallacy classification) and AFD (binary fallacy detection).

## Project Structure

```
mm_argfallacy/
├── notebooks/              # Exploration and analysis
├── src/
│   ├── configs/            # Hyperparameters and model configs
│   ├── data/               # Custom dataset classes
│   ├── experiments/        # Entry points called by scripts and notebooks
│   ├── training/           # Trainer and CV loop
│   ├── evaluation/         # Metrics and output schema
│   └── analyses/           # All plotting and analysis code
├── scripts/                # One script per experiment
├── results/                # results.json + saved figures
└── requirements.txt
```

## Installation

```bash
git clone ...
cd mm_argfallacy
python -m venv mmarg_env
source mmarg_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you have a GPU, install PyTorch manually first to match your CUDA version: https://pytorch.org/get-started/locally/

## Data

The dataset is loaded through MAMKit and downloaded automatically on first run:

```bash
python scripts/download_mmused_data.py
```

## Reproducing Results

```bash
# Text baseline AFC (35-fold)
python scripts/run_roberta_afc.py

# Text baseline AFD (35-fold)
python scripts/run_roberta_afd.py

# Multimodal baseline AFC (5-fold)
python scripts/run_wavlm_roberta_afc.py

# Multimodal AFD (5-fold)
python scripts/run_wavlm_roberta_afd.py

# RoBERTa with dialogue context AFC (5-fold)
python scripts/run_wavlm_roberta_afc_context.py

# Longformer with dialogue context AFC (5-fold)
python scripts/run_longformer_afc_context.py

# WavLM + Whisper transcriptions AFC (5-fold)
python scripts/run_wavlm_roberta_afc_whisper.py
```

Text-only models use full 35-fold leave-one-dialogue-out CV. Multimodal models use 5 folds (the 5 dialogues that have aligned audio clips) because running all 35 folds would be too slow.

## Main Results

The 5-fold results use the same test dialogues across all models for direct comparison: 13_1988, 31_2004, 25_2000, 22_1996, 46_2020.

| Model                         | Task | Folds | Macro F1 |
|-------------------------------|------|-------|----------|
| Paper baseline                | AFC  | —     | 0.393    |
| RoBERTa text-only             | AFC  | 35    | 0.476    |
| WavLM + RoBERTa (multimodal)  | AFC  | 5     | 0.502    |
| RoBERTa + dialogue context    | AFC  | 5     | 0.352    |
| Longformer + dialogue context | AFC  | 5     | 0.379    |
| WavLM + Whisper transcription | AFC  | 5     | 0.362    |
| Paper baseline                | AFD  | —     | 0.277    |
| RoBERTa text-only             | AFD  | 35    | 0.308    |

## Notebooks

```
01v2_data_exploration.ipynb      — class distributions, snippet lengths, dialogue stats
04v2_audio_exploration.ipynb     — clip durations, waveforms, spectrograms
05v2_analysis_multimodal.ipynb   — multimodal results, audio alignment, Whisper analysis, semantic similarity
06v2_comparaison_txt_multimodal.ipynb — text vs multimodal comparison, per-class F1, error analysis
```

All analysis logic is in `src/analyses/`, the notebooks just call those functions.

## Reproducibility

All experiments use a fixed seed (42) set via `L.seed_everything(seed, workers=True)` from Lightning. The seed is reset at the start of each fold so every fold is independent of fold order.

MAMKit generates LDOCV splits by iterating over a Python `set()`, which is not deterministic across runs. To fix this, `src/utils/splits.py` sorts the folds by held-out dialogue ID after generation, so the fold order is always the same.

Each script calls `prepare_text_reproducibility()` or `prepare_multimodal_reproducibility()` before anything else, which sets the seed, the HF cache path, and the matmul precision.

The full CV results are saved to `results/results.json` after each fold so a run can be inspected or resumed without retraining.

## Requirements

- Python 3.11
- CUDA 12.x
