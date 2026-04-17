"""
Selected folds for multimodal experiments.

Chosen based on:
- Class diversity (n_classes >= 5, prioritized over size)
- Fold size (n_samples >= 40 as secondary criterion)
- Excluded: 35_2008 (1 class, 3 samples), 43_2016 (1 class, 1 sample), 5_1976 (6 samples)
"""

MULTIMODAL_TEST_DIALOGUES = [
    "13_1988",  # 6 classes, 58 samples
    "31_2004",  # 6 classes, 43 samples
    "25_2000",  # 6 classes, 40 samples
    "22_1996",  # 5 classes, 62 samples
    "46_2020",  # 5 classes, 52 samples
]
