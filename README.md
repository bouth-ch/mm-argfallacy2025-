# MM-ArgFallacy — Classification Multimodale de Sophismes

Reproduction et extension de la tâche partagée MM-ArgFallacy2025 (ACL 2025) sur le dataset **MM-USED-Fallacy** : 1278 extraits audio issus de débats présidentiels américains (1960–2020), annotés pour deux tâches :

- **AFC** — Classification de sophismes (6 classes)
- **AFD** — Détection de sophismes (binaire)

---

## Structure du projet

```
mm_argfallacy/
├── notebooks/               # Exploration et analyse
├── src/
│   ├── configs/             # Hyperparamètres et sélection des folds
│   ├── data/                # Classes de dataset, prétraitement audio, audit Whisper
│   ├── experiments/         # Points d'entrée CV appelés par les scripts
│   ├── training/            # Trainer, boucle CV, fonctions de perte
│   ├── evaluation/          # Métriques et schéma de sortie
│   ├── analyses/            # Fonctions de visualisation et d'analyse
│   └── utils/               # Splits, I/O résultats, manifest des folds
├── scripts/                 # Un script par expérience
├── results/                 # results.json, whisper_audit.csv, figures/
└── requirements.txt
```

---

## Installation

```bash
git clone <repo>
cd mm_argfallacy
python -m venv mmarg_env
source mmarg_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Installer PyTorch pour votre version CUDA avant de lancer : https://pytorch.org/get-started/locally/

---

## Données

```bash
python scripts/download_mmused_data.py
```

Les données sont placées dans `data/MMUSED-fallacy/` :
- `dataset.pkl` — 1278 échantillons sur 35 dialogues, avec texte, timestamps audio et labels de sophismes
- `audio_clips/` — clips WAV pré-extraits, un par snippet

---

## Expériences

Tous les résultats sont sauvegardés dans `results/results.json`. Chaque script ajoute son entrée sans écraser les autres.

### Baselines texte seul (35-fold LDOCV)

```bash
python scripts/run_roberta_afc.py       # RoBERTa AFC
python scripts/run_roberta_afd.py       # RoBERTa AFD
```

### Multimodal — Fusion tardive WavLM + RoBERTa

Architecture : RoBERTa-base (texte) + WavLM-base → BiLSTM(128, bidirectionnel) → concat(768+256) → tête MLP.

```bash
# Évaluation 5 folds (folds sélectionnés : 13_1988, 22_1996, 25_2000, 31_2004, 46_2020)
python scripts/run_wavlm_roberta_afc.py
python scripts/run_wavlm_roberta_afd.py
python scripts/run_wavlm_roberta_afc_focal.py         # Focal Loss + WeightedRandomSampler

# Évaluation 35 folds
python scripts/run_wavlm_roberta_afc_35folds.py

# Extensions
python scripts/run_wavlm_roberta_afc_context.py               # + contexte dialogue (k=3)
python scripts/run_wavlm_roberta_afc_context_k1_35folds.py    # + contexte dialogue (k=1, 35 folds)
python scripts/run_wavlm_roberta_afc_whisper.py               # Transcription Whisper comme entrée texte
python scripts/run_wavlm_roberta_afc_trimmed.py               # Clips audio rognés
python scripts/run_longformer_afc_context.py                  # Longformer + contexte dialogue
```

---

## Résultats

L'évaluation 5 folds utilise les mêmes 5 dialogues de test pour tous les modèles.

### AFC — Classification de sophismes (6 classes, Macro F1)

| Modèle | Folds | Macro F1 | Écart-type |
|---|---|---|---|
| Baseline papier (Mancini et al. 2024) | — | 0.393 | — |
| RoBERTa texte seul | 35 | 0.476 | 0.180 |
| **WavLM + RoBERTa (multimodal)** | **5** | **0.502** | **0.057** |
| WavLM + RoBERTa (35 folds) | 35 | 0.445 | 0.183 |
| WavLM + RoBERTa (Focal Loss) | 5 | 0.238 | 0.056 |
| WavLM + RoBERTa + contexte (k=1, 35 folds) | 35 | 0.426 | 0.190 |
| WavLM + RoBERTa + contexte (k=3) | 5 | 0.352 | 0.104 |
| WavLM + transcription Whisper | 5 | 0.362 | 0.127 |
| WavLM + clips rognés | 5 | 0.399 | 0.094 |
| Longformer + contexte dialogue | 5 | 0.379 | 0.075 |

### AFD — Détection de sophismes (binaire, Macro F1)

| Modèle | Folds | Macro F1 | Écart-type |
|---|---|---|---|
| Baseline papier | — | 0.277 | — |
| RoBERTa texte seul | 35 | 0.308 | 0.133 |
| WavLM + RoBERTa | 5 | 0.319 | 0.084 |

---

## Notebooks

| Notebook | Contenu |
|---|---|
| `01v2_data_exploration.ipynb` | Distributions des classes, longueurs des snippets, statistiques des dialogues |
| `02_text_baseline.ipynb` | Entraînement et évaluation des baselines texte RoBERTa (AFC et AFD) |
| `03_analysis_afc_afd.ipynb` | Analyse des résultats texte : F1 par fold, matrices de confusion, XAI (SHAP, saliency, attention) |
| `04v2_audio_exploration.ipynb` | Durées des clips, formes d'onde, spectrogrammes, EDA audio par classe |
| `05v2_analysis_multimodal.ipynb` | Résultats multimodaux, audit Whisper, analyse d'alignement audio-texte |
| `06_comparaison_txt_multimodal.ipynb` | Comparaison texte vs multimodal, F1 par classe, analyse des erreurs |
| `07_statistical_tests.ipynb` | Tests de Wilcoxon, significativité au niveau des folds |

Toutes les fonctions d'analyse sont dans `src/analyses/` — les notebooks ne font qu'appeler ces fonctions.

---

## Audit de qualité audio

Les clips audio du dataset ne sont pas rognés aux frontières exactes du snippet  ils incluent le contexte du dialogue environnant. Un audit basé sur Whisper (`src/data/whisper_audit.py`) a re-transcrit les 1278 clips et comparé avec le texte des snippets étiquetés :

- WER moyen : **~0.24** sur l'ensemble du dataset
- Parmi les clips avec `ref_len ≥ 10` mots (1094 clips) :
  - **53.8%** — audio plus long que le snippet (contexte environnant présent)
  - **46.2%** — désalignement réel (l'audio ne correspond pas au texte étiqueté)

Ce désalignement est une limite structurelle du benchmark MM-USED-Fallacy et affecte le signal audio reçu par WavLM durant l'entraînement.

---

## Reproductibilité

- Graine fixe `42` via `L.seed_everything(42, workers=True)`, réinitialisée au début de chaque fold.
- Le LDOCV de MAMKit itère sur un `set()` (non déterministe). `src/utils/splits.py` trie les folds par dialogue_id retenu après génération.
- Tous les résultats CV sont ajoutés à `results/results.json` après chaque fold, permettant d'inspecter ou reprendre un entraînement.
- Chaque script appelle `prepare_text_reproducibility()` ou `prepare_multimodal_reproducibility()` avant l'entraînement.

---

## Prérequis

- Python 3.11
- CUDA 12.x
- Voir `requirements.txt` pour les dépendances Python
