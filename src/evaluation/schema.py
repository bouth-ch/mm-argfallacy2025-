# column names differ between AFC and AFD in the mamkit dataframes


def text_column(task_name: str) -> str:
    if task_name == "afc":
        return "snippet"
    if task_name == "afd":
        return "sentence"
    raise ValueError(f"Unknown task_name: {task_name!r}")


def label_column(task_name: str) -> str:
    if task_name == "afc":
        return "fallacy"
    if task_name == "afd":
        return "label"
    raise ValueError(f"Unknown task_name: {task_name!r}")


def context_column(task_name: str) -> str:
    if task_name == "afc":
        return "dialogue"
    if task_name == "afd":
        return "context"
    raise ValueError(f"Unknown task_name: {task_name!r}")


def score_column_from_metric(metric_key: str) -> str:
    """Human-readable column name for per-fold tables."""
    if metric_key == "test_macro_f1":
        return "macro_f1"
    if metric_key == "test_binary_f1":
        return "binary_f1"
    return "score"
