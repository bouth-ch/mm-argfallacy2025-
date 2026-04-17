import torch as th

from src.training.losses import FocalLoss


def get_roberta_afc_config():
    return {
        'name': 'roberta_afc',
        'model_card': 'roberta-base',
        'head': lambda: th.nn.Sequential(
            th.nn.Linear(768, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 50),
            th.nn.ReLU(),
            th.nn.Linear(50, 6)
        ),
        'num_classes': 6,
        'dropout_rate': 0.1,
        'is_transformer_trainable': True,
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


def get_roberta_afd_config():
    return {
        'name': 'roberta_afd',
        'model_card': 'roberta-base',
        'head': lambda: th.nn.Sequential(
            th.nn.Linear(768, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 50),
            th.nn.ReLU(),
            th.nn.Linear(50, 2)
        ),
        'num_classes': 2,
        'dropout_rate': 0.1,
        'is_transformer_trainable': True,
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


def get_deberta_afc_config():
    config = get_roberta_afc_config()
    config['name'] = 'deberta_afc'
    config['model_card'] = 'microsoft/deberta-v3-base'
    return config


def get_deberta_afd_config():
    config = get_roberta_afd_config()
    config['name'] = 'deberta_afd'
    config['model_card'] = 'microsoft/deberta-v3-base'
    return config


def get_longformer_afd_config():
    config = get_roberta_afd_config()
    config['name'] = 'longformer_afd'
    config['model_card'] = 'allenai/longformer-base-4096'
    config['tokenizer_args'] = {'truncation': True, 'max_length': 4096}
    return config


def get_longformer_afc_context_config():
    config = get_roberta_afc_config()
    config['name'] = 'longformer_afc_context'
    config['model_card'] = 'allenai/longformer-base-4096'
    config['tokenizer_args'] = {'truncation': True, 'max_length': 4096}
    return config


def get_roberta_afc_focal_config():
    """
    AFC with Focal Loss (gamma=2) + WeightedRandomSampler.
    Addresses class imbalance beyond plain class-weighted CE.
    """
    return {
        'name': 'roberta_afc_focal',
        'model_card': 'roberta-base',
        'head': lambda: th.nn.Sequential(
            th.nn.Linear(768, 100),
            th.nn.ReLU(),
            th.nn.Linear(100, 50),
            th.nn.ReLU(),
            th.nn.Linear(50, 6)
        ),
        'num_classes': 6,
        'dropout_rate': 0.1,
        'is_transformer_trainable': True,
        'batch_size': 8,
        'loss_function': lambda: FocalLoss(
            alpha=th.tensor([0.2662, 1.1152, 1.4295, 3.8036, 4.6304, 5.9167]),
            gamma=2.0
        ),
        'optimizer': th.optim.Adam,
        'optimizer_args': {'lr': 1e-5, 'weight_decay': 1e-5},
        'max_epochs': 20,
        'patience': 5,
        'task_name': 'afc',
        'use_weighted_sampler': True,
        'seed': 42,
    }
