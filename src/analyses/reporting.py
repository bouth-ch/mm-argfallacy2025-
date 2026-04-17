from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.configs.text_configs import get_roberta_afc_config, get_roberta_afd_config
from src.evaluation.metrics import load_results
from src.experiments.mmused_text import make_mmused_fallacy_loader


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_name: str
    task_name: str
    paper_baseline: float
    metric_label: str
    label_names: dict[int, str]
    config_factory: callable


EXPERIMENT_SPECS = {
    "roberta_afc": ExperimentSpec(
        experiment_name="roberta_afc",
        task_name="afc",
        paper_baseline=0.3925,
        metric_label="Macro F1",
        label_names={
            0: "AppealToEmotion",
            1: "AppealToAuthority",
            2: "AdHominem",
            3: "FalseCause",
            4: "SlipperySlope",
            5: "Slogans",
        },
        config_factory=get_roberta_afc_config,
    ),
    "roberta_afd": ExperimentSpec(
        experiment_name="roberta_afd",
        task_name="afd",
        paper_baseline=0.2770,
        metric_label="Binary F1",
        label_names={
            0: "NonFallacy",
            1: "Fallacy",
        },
        config_factory=get_roberta_afd_config,
    ),
}


def get_experiment_spec(experiment_name: str) -> ExperimentSpec:
    try:
        return EXPERIMENT_SPECS[experiment_name]
    except KeyError as exc:
        known = ", ".join(sorted(EXPERIMENT_SPECS))
        raise KeyError(f"Unknown experiment {experiment_name!r}. Known: {known}") from exc


def build_analysis_context(experiment_name: str) -> dict:
    spec = get_experiment_spec(experiment_name)
    config = spec.config_factory()
    loader = make_mmused_fallacy_loader(spec.task_name)
    return {
        "experiment_name": spec.experiment_name,
        "task_name": spec.task_name,
        "config": config,
        "paper_baseline": spec.paper_baseline,
        "metric_label": spec.metric_label,
        "label_names": spec.label_names,
        "class_names": [spec.label_names[i] for i in sorted(spec.label_names)],
        "loader": loader,
    }


def summarize_experiment(experiment_name, results_path=None):
    if results_path is None:
        from src.evaluation.metrics import _DEFAULT_RESULTS
        results_path = _DEFAULT_RESULTS
    result = load_results(experiment_name, results_path=results_path)
    spec = get_experiment_spec(experiment_name)
    scores = np.asarray(result["scores"], dtype=float)
    if scores.size == 0:
        raise ValueError(f"No scores stored for {experiment_name!r}")

    return pd.DataFrame(
        [
            {
                "experiment": experiment_name,
                "task": spec.task_name,
                "metric": result.get("metric"),
                "metric_label": spec.metric_label,
                "folds": int(scores.size),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "median": float(np.median(scores)),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "paper_baseline": float(spec.paper_baseline),
                "delta_vs_paper": float(scores.mean() - spec.paper_baseline),
            }
        ]
    )


def summarize_many_experiments(experiment_names, results_path=None):
    frames = [
        summarize_experiment(name, results_path=results_path)
        for name in experiment_names
    ]
    return pd.concat(frames, ignore_index=True)
