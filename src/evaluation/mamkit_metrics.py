"""
TorchMetrics builders matching the stack used by mamkit's MAMKitLightingModel
(val_/test_ metric keys come from these names).
"""

from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score


def build_val_test_metrics(task_name: str, num_classes: int) -> MetricCollection:
    """
    Same metric definitions as training (macro F1 for AFC, binary F1 for AFD).
    """
    if task_name == "afc":
        return MetricCollection(
            {
                "macro_f1": MulticlassF1Score(
                    num_classes=num_classes,
                    average="macro",
                    zero_division=0,
                )
            }
        )
    if task_name == "afd":
        return MetricCollection(
            {
                "binary_f1": BinaryF1Score(zero_division=0),
            }
        )
    raise ValueError(f"Unknown task_name: {task_name!r}")
