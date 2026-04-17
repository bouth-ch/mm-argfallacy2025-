import shutil
import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import torch as th
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mamkit.models.text import Transformer
from mamkit.utility.model import MAMKitLightingModel
from mamkit.data.collators import TextTransformerCollator, UnimodalCollator

from src.evaluation.mamkit_metrics import build_val_test_metrics
from src.utils.fold_manifest import (
    DEFAULT_CHECKPOINTS_ROOT,
    record_fold,
    write_per_fold_checkpoint_manifest,
)
from src.utils.results import ResultsManager
from src.utils.splits import infer_held_out_dialogue_id, sort_ldocv_splits


def _best_middle_worst(scores):
    """Returns fold indices (0-based) for best, middle and worst test scores."""
    arr = np.asarray(scores)
    order = np.argsort(arr)
    return {
        'best': int(order[-1]),
        'middle': int(order[len(order) // 2]),
        'worst': int(order[0]),
    }


class BaseTrainer:
    """Shared CV loop logic. Subclasses implement build_model() and build_collator()."""

    def __init__(self, config, results_manager):
        self.config = config
        self.results_manager = results_manager

    def _set_seed(self):
        seed = self.config.get('seed')
        if seed is not None:
            L.seed_everything(int(seed), workers=True)

    def build_model(self):
        raise NotImplementedError

    def build_collator(self):
        raise NotImplementedError

    def build_metrics(self):
        raise NotImplementedError

    def build_dataloader(self, dataset, collator=None, shuffle=False, use_weighted_sampler=False):
        # build collator lazily if not provided (e.g. called from outside run_fold)
        if collator is None:
            collator = self.build_collator()

        sampler = None
        if use_weighted_sampler:
            labels = [int(dataset[i][1]) for i in range(len(dataset))]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
            sample_weights = [class_weights[l] for l in labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False  # can't use both shuffle and sampler

        return DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            drop_last=False,
        )

    def run_fold(self, split_info, checkpoint_dir=None):
        self._set_seed()

        # build collator once and share across all three loaders for this fold
        collator = self.build_collator()
        use_sampler = self.config.get('use_weighted_sampler', False)
        train_loader = self.build_dataloader(split_info.train, collator, shuffle=True, use_weighted_sampler=use_sampler)
        val_loader   = self.build_dataloader(split_info.val, collator)
        test_loader  = self.build_dataloader(split_info.test, collator)

        model    = self.build_model()
        metrics  = self.build_metrics()

        lit_model = MAMKitLightingModel(
            model=model,
            loss_function=self.config['loss_function'],
            num_classes=self.config['num_classes'],
            optimizer_class=self.config['optimizer'],
            val_metrics=metrics.clone(),
            test_metrics=metrics.clone(),
            **self.config['optimizer_args']
        )

        # decide where to put checkpoints
        if checkpoint_dir is not None:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            ckpt_dir = str(checkpoint_dir)
            cleanup = False
        elif self.config.get('tmp_checkpoint_root'):
            ckpt_dir = str(Path(self.config['tmp_checkpoint_root']) / 'fold_tmp_ckpt')
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            cleanup = True
        else:
            ckpt_dir = tempfile.mkdtemp()
            cleanup = True

        fold_result = None
        try:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=self.config['patience'], mode='min'),
                ModelCheckpoint(
                    dirpath=ckpt_dir,
                    monitor='val_loss',
                    mode='min',
                    save_top_k=1,
                    save_weights_only=True,
                    filename='best'
                ),
            ]

            trainer = L.Trainer(
                max_epochs=self.config['max_epochs'],
                accelerator='gpu',
                devices=1,
                callbacks=callbacks,
                enable_progress_bar=True,
                logger=False,
            )

            trainer.fit(lit_model, train_loader, val_loader)
            results = trainer.test(lit_model, test_loader, ckpt_path='best', verbose=False)

            # collect predictions from the test set
            all_preds, all_labels = [], []
            device = next(lit_model.parameters()).device
            lit_model.eval()
            with th.no_grad():
                for batch in test_loader:
                    inputs, labels = batch
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(device) if isinstance(v, th.Tensor) else v
                                  for k, v in inputs.items()}
                    elif isinstance(inputs, th.Tensor):
                        inputs = inputs.to(device)
                    logits = lit_model.model(inputs)
                    preds = th.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())

            results[0]['predictions'] = all_preds
            results[0]['true_labels'] = all_labels
            fold_result = results[0]

        finally:
            if cleanup and Path(ckpt_dir).exists():
                shutil.rmtree(ckpt_dir, ignore_errors=True)

        return fold_result

    def _checkpoints_root(self):
        return Path(self.config.get('checkpoints_root', DEFAULT_CHECKPOINTS_ROOT))

    def refit_folds_save_checkpoints_only(self, loader, fold_indices, splits=None):
        """Re-train specific folds and save their checkpoints. Doesn't touch results.json."""
        if splits is None:
            self._set_seed()
            splits = list(loader.get_splits('mancini-et-al-2024'))

        name = self.config['name']
        root = self._checkpoints_root()

        for fold_idx in sorted(fold_indices):
            if fold_idx < 0 or fold_idx >= len(splits):
                raise IndexError(f"fold_idx {fold_idx} out of range ({len(splits)} splits)")

            ckpt_dir = root / name / f'fold_{fold_idx}'
            print(f"\n--- REFIT (checkpoint only) | {name} | Fold {fold_idx + 1} ---")

            held_out = infer_held_out_dialogue_id(loader, splits[fold_idx])
            result = self.run_fold(splits[fold_idx], checkpoint_dir=ckpt_dir)
            result['held_out_dialogue_id'] = held_out

            record_fold(name, fold_idx, held_out, loader.task_name, self.config['model_card'], root=root)
            write_per_fold_checkpoint_manifest(ckpt_dir, name, fold_idx, held_out, loader.task_name, self.config['model_card'])
            print(f"Saved: {ckpt_dir / 'best.ckpt'}")

    def maybe_save_bm3_checkpoints(self, loader, scores, splits=None):
        """After a full CV, refit and save checkpoints for best / middle / worst folds."""
        name = self.config['name']
        root = self._checkpoints_root()

        if splits is None:
            self._set_seed()
            splits = list(loader.get_splits('mancini-et-al-2024'))

        if len(scores) < len(splits):
            print(f"Skipping BM3: only {len(scores)} scores, need {len(splits)}.")
            return None

        bm = _best_middle_worst(scores)
        missing = {i for i in bm.values() if not (root / name / f'fold_{i}' / 'best.ckpt').is_file()}

        if not missing:
            print(f"BM3 checkpoints already exist for {name}: {bm}")
            return bm

        print(f"Saving BM3 checkpoints for {name}: {bm}  (missing: {sorted(missing)})")
        self.refit_folds_save_checkpoints_only(loader, missing, splits=splits)
        return bm

    def run_experiment(self, loader, max_folds=None, save_checkpoint_folds=None,
                       save_bm3_checkpoints_after=False, test_dialogues=None):
        name = self.config['name']
        ck_root = self._checkpoints_root()

        # resume if the run was interrupted
        existing = self.results_manager.load(name)
        start_fold = len(existing['scores']) if existing else 0
        if start_fold > 0:
            print(f"Resuming {name} from fold {start_fold + 1}")

        self._set_seed()
        splits = list(loader.get_splits('mancini-et-al-2024'))
        splits = sort_ldocv_splits(loader, splits)

        # filter to selected dialogues only
        if test_dialogues is not None:
            splits = [(infer_held_out_dialogue_id(loader, sp), sp) for sp in splits]
            splits = [(did, sp) for did, sp in splits if did in test_dialogues]
            splits = [sp for _, sp in splits]
            print(f"Running on {len(splits)} selected folds: {test_dialogues}")

        for fold_idx, split_info in enumerate(splits):
            if fold_idx < start_fold:
                continue
            if max_folds is not None and fold_idx >= max_folds:
                break

            ckpt_dir = None
            if save_checkpoint_folds and fold_idx in save_checkpoint_folds:
                ckpt_dir = ck_root / name / f'fold_{fold_idx}'

            print(f"\n--- {name} | Fold {fold_idx + 1} ---")
            held_out = infer_held_out_dialogue_id(loader, split_info)
            result = self.run_fold(split_info, checkpoint_dir=ckpt_dir)
            result['held_out_dialogue_id'] = held_out
            print(f"Result: {result}")

            record_fold(name, fold_idx, held_out, loader.task_name, self.config['model_card'], root=ck_root)
            if ckpt_dir is not None:
                write_per_fold_checkpoint_manifest(ckpt_dir, name, fold_idx, held_out, loader.task_name, self.config['model_card'])

            # save after every fold so a crash doesn't lose everything
            self.results_manager.add_fold_result(name, result, self.config)

        summary = self.results_manager.summary(name)
        if save_bm3_checkpoints_after:
            self.maybe_save_bm3_checkpoints(loader, summary.get('scores') or [], splits=splits)
        return summary


class TextTrainer(BaseTrainer):

    def build_collator(self):
        return UnimodalCollator(
            features_collator=TextTransformerCollator(
                model_card=self.config['model_card'],
                tokenizer_args=self.config.get('tokenizer_args', {})
            ),
            label_collator=lambda labels: th.tensor([int(l) for l in labels], dtype=th.long)
        )

    def build_model(self):
        return Transformer(
            model_card=self.config['model_card'],
            head=self.config['head'],
            dropout_rate=self.config['dropout_rate'],
            is_transformer_trainable=self.config['is_transformer_trainable']
        )

    def build_metrics(self):
        return build_val_test_metrics(self.config['task_name'], self.config['num_classes'])


