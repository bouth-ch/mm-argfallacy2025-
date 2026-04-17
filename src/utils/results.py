import json
import numpy as np
from pathlib import Path


class ResultsManager:
    """
    Saves and loads experiment results to JSON after every fold.
    Crash-safe: results are written incrementally.
    """

    def __init__(self, results_path: str = None):
        if results_path is None:
            results_path = str(Path(__file__).resolve().parents[2] / 'results' / 'results.json')
        self.results_path = Path(results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

        if self.results_path.exists():
            with open(self.results_path, 'r') as f:
                self.data = json.load(f)
            for _name, exp in self.data.items():
                n = len(exp.get('scores', []))
                if 'dialogue_ids' not in exp:
                    exp['dialogue_ids'] = [None] * n
                elif len(exp['dialogue_ids']) < n:
                    exp['dialogue_ids'].extend(
                        [None] * (n - len(exp['dialogue_ids']))
                    )
        else:
            self.data = {}

    def load(self, experiment_name):
        return self.data.get(experiment_name, None)

    def add_fold_result(self, experiment_name, fold_result, config):
        metric_key = 'test_macro_f1' if config['task_name'] == 'afc' else 'test_binary_f1'
        if metric_key not in fold_result:
            raise KeyError(
                f"Expected metric '{metric_key}' not in fold result. "
                f"Got keys: {list(fold_result.keys())}"
            )
        score = fold_result[metric_key]

        if experiment_name not in self.data:
            self.data[experiment_name] = {
                'scores': [],
                'dialogue_ids': [],
                'predictions': [],
                'true_labels': [],
                'mean': 0.0,
                'std': 0.0,
                'metric': metric_key,
                'model_card': config['model_card'],
                'task_name': config['task_name'],
            }

        exp = self.data[experiment_name]
        if 'dialogue_ids' not in exp:
            exp['dialogue_ids'] = []

        self.data[experiment_name]['scores'].append(float(score))
        exp['dialogue_ids'].append(fold_result.get('held_out_dialogue_id'))
        self.data[experiment_name]['predictions'].extend(fold_result.get('predictions', []))
        self.data[experiment_name]['true_labels'].extend(fold_result.get('true_labels', []))
        scores = self.data[experiment_name]['scores']
        self.data[experiment_name]['mean'] = float(np.mean(scores))
        self.data[experiment_name]['std'] = float(np.std(scores))

        self._save()

    def summary(self, experiment_name):
        return self.data.get(experiment_name, {})

    def print_comparison_table(self):
        """Print all experiments as a comparison table."""
        print(f"\n{'='*65}")
        print(f"{'Experiment':<25} {'Task':<6} {'Metric':<15} {'Mean':>8} {'Std':>8} {'Folds':>6}")
        print(f"{'='*65}")

        for name, result in self.data.items():
            print(
                f"{name:<25} "
                f"{result['task_name']:<6} "
                f"{result['metric']:<15} "
                f"{result['mean']:>8.4f} "
                f"{result['std']:>8.4f} "
                f"{len(result['scores']):>6}"
            )

        print(f"{'='*65}")

        # Paper baselines for comparison
        print(f"\n{'Paper baselines':}")
        print(f"{'roberta_afc (paper)':<25} {'afc':<6} {'macro_f1':<15} {'0.3925':>8}")
        print(f"{'roberta_afd (paper)':<25} {'afd':<6} {'binary_f1':<15} {'0.2770':>8}")

    def _save(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.data, f, indent=2)

