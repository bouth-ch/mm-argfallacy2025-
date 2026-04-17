import shutil
import tempfile
from pathlib import Path

import lightning as L
import numpy as np
import torch as th
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from mamkit.models.text_audio import MMTransformer
from mamkit.utility.model import MAMKitLightingModel
from mamkit.data.collators import TextTransformerCollator, MultimodalCollator, AudioCollator
from mamkit.data.processing import MultimodalProcessor, AudioTransformerExtractor

from src.evaluation.mamkit_metrics import build_val_test_metrics
from src.training.trainer import BaseTrainer


class MultimodalTrainer(BaseTrainer):

    def build_collator(self):
        return MultimodalCollator(
            text_collator=TextTransformerCollator(
                model_card=self.config['model_card'],
                tokenizer_args={}
            ),
            audio_collator=AudioCollator(),
            label_collator=lambda labels: th.tensor([int(l) for l in labels], dtype=th.long)
        )

    def build_model(self):
        return MMTransformer(
            model_card=self.config['model_card'],
            head=self.config['head'],
            audio_embedding_dim=self.config['audio_embedding_dim'],
            lstm_weights=self.config['lstm_weights'],
            text_dropout_rate=self.config['text_dropout_rate'],
            audio_dropout_rate=self.config['audio_dropout_rate'],
            is_transformer_trainable=self.config['is_transformer_trainable']
        )

    def build_metrics(self):
        return build_val_test_metrics(self.config['task_name'], self.config['num_classes'])

    def build_processor(self):
        return MultimodalProcessor(
            audio_processor=AudioTransformerExtractor(
                model_card=self.config['audio_model_card'],
                sampling_rate=self.config['sampling_rate'],
                aggregate=False,
            )
        )

    def build_dataloader(self, dataset, collator=None, shuffle=False, use_weighted_sampler=False):
        if collator is None:
            collator = self.build_collator()

        sampler = None
        if use_weighted_sampler:
            # label is at index 2 in MultimodalDataset
            labels = [int(dataset[i][2]) for i in range(len(dataset))]
            class_counts = np.bincount(labels)
            class_weights = 1.0 / np.where(class_counts == 0, 1, class_counts)
            sample_weights = [class_weights[l] for l in labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False

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

        # preprocess audio per fold to avoid data leakage
        processor = self.build_processor()
        processor.fit(split_info.train)
        train_data = processor(split_info.train)
        val_data = processor(split_info.val)
        test_data = processor(split_info.test)
        processor.clear()

        collator = self.build_collator()
        use_sampler = self.config.get('use_weighted_sampler', False)
        train_loader = self.build_dataloader(train_data, collator, shuffle=True, use_weighted_sampler=use_sampler)
        val_loader   = self.build_dataloader(val_data, collator)
        test_loader  = self.build_dataloader(test_data, collator)

        model   = self.build_model()
        metrics = self.build_metrics()

        lit_model = MAMKitLightingModel(
            model=model,
            loss_function=self.config['loss_function'],
            num_classes=self.config['num_classes'],
            optimizer_class=self.config['optimizer'],
            val_metrics=metrics.clone(),
            test_metrics=metrics.clone(),
            **self.config['optimizer_args']
        )

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

            all_preds, all_labels = [], []
            device = next(lit_model.parameters()).device
            lit_model.eval()
            with th.no_grad():
                for batch in test_loader:
                    inputs, labels = batch
                    inputs = {k: v.to(device) if isinstance(v, th.Tensor) else v
                              for k, v in inputs.items()}
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
