"""
Utilities for mamkit LDOCV splits.

Mamkit iterates folds via a set(), so the order isn't deterministic.
We infer the held-out dialogue by matching test samples back to the dataframe.
"""

from collections import Counter

import numpy as np
import pandas as pd

from src.evaluation.schema import label_column, text_column


def infer_held_out_dialogue_id(loader, split_info):
    """Figure out which dialogue_id was left out in split_info.test."""
    test = split_info.test
    if test is None or len(test) == 0:
        raise ValueError("split_info.test is empty; cannot infer dialogue_id")

    df = loader.data
    task = loader.task_name
    txt_col = text_column(task)
    lbl_col = label_column(task)

    test_inputs = test.inputs if hasattr(test, 'inputs') else test.texts
    n = len(test_inputs)
    t = pd.DataFrame(
        {"_row": np.arange(n, dtype=np.int64), txt_col: test_inputs, lbl_col: test.labels}
    )
    d = df[[txt_col, lbl_col, "dialogue_id"]]
    m = t.merge(d, on=[txt_col, lbl_col], how="left")
    pairs = m.dropna(subset=["dialogue_id"]).drop_duplicates(subset=["_row", "dialogue_id"])
    dialogue_votes: Counter = Counter(pairs["dialogue_id"])

    miss = t.loc[~t["_row"].isin(pairs["_row"]), ["_row", txt_col]]
    if not miss.empty:
        m2 = miss.merge(df[[txt_col, "dialogue_id"]], on=txt_col, how="left")
        p2 = m2.dropna(subset=["dialogue_id"]).drop_duplicates(subset=["_row", "dialogue_id"])
        dialogue_votes.update(Counter(p2["dialogue_id"]))

    if not dialogue_votes:
        raise RuntimeError(
            f"Could not match any test {txt_col} to loader.data; "
            "check task_name and dataframe columns."
        )

    ranked = dialogue_votes.most_common(2)
    if len(ranked) == 1:
        return ranked[0][0]

    top_did, top_votes = ranked[0]
    second_did, second_votes = ranked[1]
    if top_votes == second_votes:
        raise RuntimeError(
            "Ambiguous held-out dialogue after vote-based inference "
            f"(tie: {top_did}={top_votes}, {second_did}={second_votes})."
        )
    return top_did


def sort_ldocv_splits(loader, splits):
    """Sort folds by held-out dialogue_id so the order is consistent across runs."""
    keyed = [(infer_held_out_dialogue_id(loader, sp), sp) for sp in splits]
    keyed.sort(key=lambda t: t[0])
    return [sp for _, sp in keyed]
