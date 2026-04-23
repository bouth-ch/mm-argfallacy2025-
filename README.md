# MM-ArgFallacy 2025 : Détection Multimodale de Sophismes Argumentatifs

Reproduction et extension méthodologique de la tâche partagée MM-ArgFallacy 2025 (ACL 2025) sur la détection et la classification de sophismes argumentatifs dans des débats politiques américains.

---

## Résumé

Ce projet reproduit et étend les systèmes de référence de la tâche partagée MM-ArgFallacy 2025 à partir du dataset MM-USED-Fallacy — 1 278 extraits audio-texte issus de débats présidentiels américains (1960–2020), annotés pour deux sous-tâches : la **classification de sophismes (AFC)**, problème à 6 classes, et la **détection de sophismes (AFD)**, problème de détection binaire. Nous évaluons des architectures texte seul et multimodales sous validation croisée Leave-One-Dialogue-Out (LODO-CV) sur l'ensemble des 35 folds de dialogue, et examinons plusieurs extensions méthodologiques : injection de contexte dialogique, transcription Whisper comme signal texte alternatif, rognage des clips audio et stratégies de fusion alternatives. Un audit systématique d'alignement audio-texte met en évidence une limite structurelle du benchmark. Des analyses d'explicabilité (SHAP, saliency par gradient, attribution d'attention) sont également incluses. L'ensemble des résultats, de la méthodologie et des analyses est détaillé dans le rapport et les diapositives joints.

> **Rapport :** `report/mm_argfallacy2025_rapport.pdf`  
> **Diapositives :** `report/mm_argfallacy2025_slides.pdf`

---

## Installation

```bash
git clone https://github.com/username/mm-argfallacy2025
cd mm-argfallacy2025
python -m venv mmarg_env
source mmarg_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Installer PyTorch pour votre version CUDA avant de lancer les expériences : https://pytorch.org/get-started/locally/

Testé avec Python 3.11 et CUDA 12.x. Les expériences ont été conduites sur une instance RunPod équipée d'un RTX 4090.

---

## Données

Le dataset MM-USED-Fallacy est distribué via la bibliothèque [MAMKit](https://github.com/lt-nlp-lab-unibo/mamkit) (v0.1.1). Pour télécharger et préparer les données :

```bash
python scripts/download_mmused_data.py
```

Les fichiers sont placés dans `data/MMUSED-fallacy/` :

- `dataset.pkl` — 1 278 échantillons répartis sur 35 dialogues, avec texte de transcription, timestamps audio et labels de sophismes
- `audio_clips/` — clips WAV pré-extraits, un par snippet

**Note sur l'alignement audio-texte.** Un audit mené avec Whisper (`src/data/whisper_audit.py`) révèle qu'une part significative des clips audio contient du contexte dialogique environnant plutôt que la frontière exacte du snippet annoté. Il s'agit d'une propriété structurelle du benchmark, discutée en détail dans le rapport.

---

## Reproduction des expériences

Tous les résultats sont écrits de manière incrémentale dans `results/results.json`. Chaque script ajoute sa propre entrée sans écraser les résultats existants ; les exécutions peuvent être interrompues et reprises.

### Baselines texte seul (LODO-CV 35 folds)

```bash
python scripts/run_roberta_afc.py    # RoBERTa-base, AFC
python scripts/run_roberta_afd.py    # RoBERTa-base, AFD

```

### Fusion multimodale : WavLM + RoBERTa

Architecture : RoBERTa-base (texte) et WavLM-base (audio) → BiLSTM (128 unités, bidirectionnel) → concaténation (768 + 256) → tête de classification MLP
```bash
python scripts/run_wavlm_roberta_afc_35folds.py ( # Sur 35 fold )
python scripts/run_wavlm_roberta_afd.py (# Sur 5 fold en raison de limite computationnel et de temps )

```
### Extension 
```bash

python scripts/run_wavlm_roberta_afc_context_k1_35folds.py
```

### Experience Exploratoires 
```bash
# Évaluation 5 folds (dialogues retenus : 13_1988, 22_1996, 25_2000, 31_2004, 46_2020)
python scripts/run_wavlm_roberta_afc_focal.py       # Focal loss + WeightedRandomSampler
python scripts/run_wavlm_roberta_afc_context.py              # Injection de contexte dialogique
python scripts/run_wavlm_roberta_afc_whisper.py              # Transcription Whisper comme entrée texte
python scripts/run_wavlm_roberta_afc_trimmed.py              # Clips audio rognés
python scripts/run_longformer_afc_context.py                 # Encodeur Longformer avec contexte dialogique
```


### Reproductibilité

- Graine aléatoire fixée à `42` via `L.seed_everything(42, workers=True)`, réinitialisée au début de chaque fold.
- Le LODO-CV de MAMKit itère sur un `set()` (ordre non déterministe). `src/utils/splits.py` trie les folds par `dialogue_id` après génération pour garantir une séquence de folds stable.
- Chaque script appelle `prepare_text_reproducibility()` ou `prepare_multimodal_reproducibility()` avant l'entraînement.

---

## Notebooks d'analyse

Les fonctions analytiques sont définies dans `src/analyses/` ; les notebooks appellent ces fonctions directement.

| Notebook | Contenu |
|---|---|
| `01v2_data_exploration.ipynb` | Distributions des classes, longueurs des snippets, statistiques des dialogues |
| `02_text_baseline.ipynb` | Entraînement et évaluation de la baseline texte RoBERTa |
| `03_analysis_afc_afd.ipynb` | Résultats texte : F1 par fold, matrices de confusion, XAI (SHAP, saliency par gradient, attention) |
| `04v2_audio_exploration.ipynb` | Durées des clips, formes d'onde, spectrogrammes, EDA audio par classe |
| `05v2_analysis_multimodal.ipynb` | Résultats multimodaux, audit d'alignement Whisper, analyse audio-texte |
| `06_comparaison_txt_multimodal.ipynb` | Comparaison texte vs. multimodal, F1 par classe, analyse des erreurs |
| `07_statistical_tests.ipynb` | Tests de Wilcoxon, significativité au niveau des folds |

---

## Structure du projet

```
mm_argfallacy/
├── notebooks/               # Notebooks d'exploration et d'analyse
├── src/
│   ├── configs/             # Hyperparamètres et sélection des folds
│   ├── data/                # Classes de dataset, prétraitement audio, audit Whisper
│   ├── experiments/         # Points d'entrée CV appelés par les scripts
│   ├── training/            # Trainer, boucle CV, fonctions de perte
│   ├── evaluation/          # Métriques et schéma de sortie
│   └── analyses/            # Fonctions de visualisation et d'analyse
├── scripts/                 # Un script par expérience
├── results/                 # results.json, whisper_audit.csv, figures/
├── report/                  # Rapport PDF et diapositives
└── requirements.txt
```

---

## Citation

Si vous utilisez ce code ou ces résultats dans vos travaux, merci de citer l'article de la tâche partagée originale :

```bibtex
@inproceedings{mancini-etal-2024-mmargfallacy,
  title     = {{MM-ArgFallacy}: A Shared Task on Multimodal Argumentative Fallacy Classification},
  author    = {Mancini, Eleonora and others},
  booktitle = {Proceedings of the 11th Workshop on Argument Mining (ArgMining 2024)},
  year      = {2024}
}
```
---

## Matériel et ressources de calcul

Les expériences ont été conduites sur une instance **RunPod** équipée du matériel suivant :

| Composant | Détail |
|---|---|
| GPU | NVIDIA RTX 4090 (24 Go VRAM) |
| CUDA | 12.x |
| Python | 3.11 |

### Prérequis GPU recommandés

- **Minimum :** GPU avec ≥ 16 Go VRAM (ex. RTX 3090, A4000) pour les configurations à batch size réduit.
- **Recommandé :** GPU avec ≥ 24 Go VRAM (ex. RTX 4090, A5000) pour reproduire les résultats dans les conditions d'origine.
- L'entraînement sur CPU est théoriquement possible mais non testé et prohibitif en temps.

> **Note.** Le LODO-CV complet sur 35 folds pour les modèles multimodaux est coûteux en calcul. Les scripts exploratoires sont limités à 5 folds représentatifs (dialogues `13_1988`, `22_1996`, `25_2000`, `31_2004`, `46_2020`) pour réduire le temps d'exécution.

---
 Le dataset MM-USED-Fallacy est soumis à ses propres conditions d'utilisation telles que distribuées via MAMKit ; consulter le [dépôt MAMKit](https://github.com/lt-nlp-lab-unibo/mamkit) pour les détails. Le code de ce dépôt est distribué sous licence MIT.
