import torch as th

from src.training.losses import FocalLoss


def get_wavlm_roberta_afc_config():
    return {
        'name': 'wavlm_roberta_afc',
        # Text
        'model_card': 'roberta-base',
        'is_transformer_trainable': True,
        'text_dropout_rate': 0.1,
        # Audio
        'audio_model_card': 'microsoft/wavlm-base',
        'audio_embedding_dim': 768,
        'lstm_weights': [128],
        'audio_dropout_rate': 0.1,
        'sampling_rate': 16000,
        # Head: text_dim(768) + audio_bilstm_dim(128*2) = 1024
        'head': lambda: th.nn.Sequential(
            th.nn.Linear(1024, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 6)
        ),
        'num_classes': 6,
        'batch_size': 8,
        'loss_function': lambda: th.nn.CrossEntropyLoss(
            weight=th.tensor([0.2662, 1.1152, 1.4295, 3.8036, 4.6304, 5.9167])
        ),
        'optimizer': th.optim.Adam,
        'optimizer_args': {'lr': 1e-5, 'weight_decay': 1e-5},
        'max_epochs': 20,
        'patience': 5,
        'task_name': 'afc',
        'seed': 42,
    }


def get_wavlm_roberta_afd_config():
    return {
        'name': 'wavlm_roberta_afd',
        # Text
        'model_card': 'roberta-base',
        'is_transformer_trainable': True,
        'text_dropout_rate': 0.1,
        # Audio
        'audio_model_card': 'microsoft/wavlm-base',
        'audio_embedding_dim': 768,
        'lstm_weights': [128],
        'audio_dropout_rate': 0.1,
        'sampling_rate': 16000,
        # Head: text_dim(768) + audio_bilstm_dim(128*2) = 1024
        'head': lambda: th.nn.Sequential(
            th.nn.Linear(1024, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 2)
        ),
        'num_classes': 2,
        'batch_size': 8,
        'loss_function': lambda: th.nn.CrossEntropyLoss(
            weight=th.tensor([0.5498, 5.4598])
        ),
        'optimizer': th.optim.Adam,
        'optimizer_args': {'lr': 1e-5, 'weight_decay': 1e-5},
        'max_epochs': 20,
        'patience': 5,
        'task_name': 'afd',
        'seed': 42,
    }


def get_wavlm_roberta_afc_context_config():
    config = get_wavlm_roberta_afc_config()
    config['name'] = 'wavlm_roberta_afc_context'
    return config


def get_wavlm_roberta_afc_focal_config():
    config = get_wavlm_roberta_afc_config()
    config['name'] = 'wavlm_roberta_afc_focal'
    config['loss_function'] = lambda: FocalLoss(
        alpha=th.tensor([0.2662, 1.1152, 1.4295, 3.8036, 4.6304, 5.9167]),
        gamma=2.0
    )
    config['use_weighted_sampler'] = True
    return config
