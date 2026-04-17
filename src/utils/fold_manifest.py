"""Fold-to-dialogue mapping, stored next to checkpoints."""

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS_ROOT = _ROOT / "checkpoints"
DEFAULT_RESULTS_PATH = _ROOT / "results" / "results.json"


def record_fold(experiment_name, fold_idx, dialogue_id, task_name, model_card, *, root=DEFAULT_CHECKPOINTS_ROOT):
    """Save fold → dialogue_id to fold_dialogues.json next to the checkpoints."""
    path = root / experiment_name / "fold_dialogues.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if path.exists():
        data = json.loads(path.read_text())

    key = str(fold_idx)
    prev = data.get(key)
    if prev and prev.get("held_out_dialogue_id") != dialogue_id:
        raise ValueError(
            f"Fold {fold_idx} in {path} already maps to {prev.get('held_out_dialogue_id')!r}, "
            f"can't overwrite with {dialogue_id!r}"
        )

    data[key] = {"held_out_dialogue_id": dialogue_id, "task_name": task_name, "model_card": model_card}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_per_fold_checkpoint_manifest(checkpoint_dir, experiment_name, fold_idx, dialogue_id, task_name, model_card):
    """Write a small JSON next to best.ckpt so we always know which dialogue was held out."""
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_name": experiment_name,
        "fold_idx": fold_idx,
        "held_out_dialogue_id": dialogue_id,
        "task_name": task_name,
        "model_card": model_card,
    }
    (d / "fold_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def dialogue_for_fold(
    experiment_name,
    fold_idx,
    *,
    results_path=DEFAULT_RESULTS_PATH,
    checkpoints_root=DEFAULT_CHECKPOINTS_ROOT,
    strict=False,
):
    """
    Look up which dialogue was held out for a given fold index.

    Checks the global manifest, per-fold manifest, and results.json in that order.
    Set strict=True when loading a checkpoint for inference - it will raise if the
    checkpoint exists but there's no manifest (instead of silently returning None).
    """
    results_path = Path(results_path)
    root = checkpoints_root

    # check global manifest (fold_dialogues.json)
    global_map = {}
    gpath = root / experiment_name / "fold_dialogues.json"
    if gpath.exists():
        global_map = json.loads(gpath.read_text())
    g_entry = global_map.get(str(fold_idx))
    d_global = g_entry.get("held_out_dialogue_id") if g_entry else None

    # check per-fold manifest (fold_N/fold_manifest.json)
    pf = root / experiment_name / f"fold_{fold_idx}" / "fold_manifest.json"
    d_fold = None
    if pf.exists():
        d_fold = json.loads(pf.read_text()).get("held_out_dialogue_id")

    if d_global and d_fold and d_global != d_fold:
        raise ValueError(
            f"Manifest conflict for {experiment_name} fold {fold_idx}: "
            f"fold_dialogues.json says {d_global!r} but {pf} says {d_fold!r}"
        )
    d_from_manifest = d_global or d_fold

    # fallback: check results.json dialogue_ids list
    d_from_results = None
    if results_path.exists():
        blob = json.loads(results_path.read_text())
        ids = blob.get(experiment_name, {}).get("dialogue_ids", [])
        if fold_idx < len(ids):
            d_from_results = ids[fold_idx]

    if d_from_manifest and d_from_results and d_from_manifest != d_from_results:
        raise ValueError(
            f"Manifest disagrees with results.json for {experiment_name} fold {fold_idx}: "
            f"{d_from_manifest!r} vs {d_from_results!r}. Fix one or re-train."
        )

    if strict:
        fold_ckpt = root / experiment_name / f"fold_{fold_idx}" / "best.ckpt"
        if not fold_ckpt.is_file():
            raise FileNotFoundError(f"No checkpoint at {fold_ckpt}")
        if not d_from_manifest:
            raise FileNotFoundError(
                f"Checkpoint exists at {fold_ckpt} but no training manifest found. "
                "Re-run training to generate manifests."
            )
        return d_from_manifest

    return d_from_manifest or d_from_results
