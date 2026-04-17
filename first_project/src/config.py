from __future__ import annotations
import argparse
import copy
import yaml
from pathlib import Path


def _as_dict(value):
    return dict(value) if isinstance(value, dict) else {}


def _merge_alias(target, source, key, aliases, default=None):
    if key not in target or target.get(key) in (None, ""):
        for alias in aliases:
            if alias in source and source[alias] not in (None, ""):
                target[key] = source[alias]
                return
        if default is not None and key not in target:
            target[key] = default


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results and tensorboard logs')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    cfg = copy.deepcopy(cfg)

    data = _as_dict(cfg.get('data'))
    model_value = cfg.get('model')
    model = _as_dict(model_value)
    train = _as_dict(cfg.get('train'))

    # Flat-config aliases for dataset section.
    _merge_alias(data, cfg, 'root', ['data_dir', 'data_root'])
    _merge_alias(data, cfg, 'name', ['dataset', 'dataset_name'], default='CIFAR100')
    _merge_alias(data, cfg, 'batch_size', ['batch_size'], default=32)
    _merge_alias(data, cfg, 'num_workers', ['num_workers'], default=4)
    _merge_alias(data, cfg, 'download', ['download'], default=True)
    _merge_alias(data, cfg, 'k_train', ['k_train'], default=-1)
    _merge_alias(data, cfg, 'label_level', ['label_level'], default='fine')
    _merge_alias(data, cfg, 'randaugment_enable', ['randaugment_enable'], default=False)
    _merge_alias(data, cfg, 'randaugment_num_ops', ['randaugment_num_ops'], default=2)
    _merge_alias(data, cfg, 'randaugment_magnitude', ['randaugment_magnitude'], default=9)
    _merge_alias(data, cfg, 'pin_memory', ['pin_memory'], default=True)
    if isinstance(data.get('name'), str):
        normalized_name = data['name'].replace('-', '').lower()
        if normalized_name == 'cifar100':
            data['name'] = 'CIFAR100'

    # Flat-config aliases for model section.
    if isinstance(model_value, str):
        model['name'] = model_value
    _merge_alias(model, cfg, 'name', ['model_name'], default='dhvt')
    _merge_alias(model, cfg, 'hybrid_ratio', ['hybrid_ratio'], default=0.5)
    _merge_alias(model, cfg, 'num_classes', ['num_classes'], default=100)
    _merge_alias(model, cfg, 'pretrained', ['pretrained'], default=False)
    _merge_alias(model, cfg, 'dropout', ['dropout', 'dropout_rate'], default=0.1)
    _merge_alias(model, cfg, 'num_coarse_classes', ['num_coarse_classes'], default=20)
    _merge_alias(model, cfg, 'dhvt_dims', ['dhvt_dims'])
    _merge_alias(model, cfg, 'dhvt_depths', ['dhvt_depths'])
    _merge_alias(model, cfg, 'dhvt_heads', ['dhvt_heads'])
    _merge_alias(model, cfg, 'dhvt_mlp_ratio', ['dhvt_mlp_ratio'], default=3.0)
    _merge_alias(model, cfg, 'attn_dropout', ['attn_dropout'], default=0.0)
    _merge_alias(model, cfg, 'drop_path_rate', ['drop_path_rate'], default=0.2)
    _merge_alias(model, cfg, 'layer_scale_init', ['layer_scale_init'], default=1e-4)

    # Flat-config aliases for train section.
    _merge_alias(train, cfg, 'epochs', ['epochs', 'num_epochs'], default=100)
    _merge_alias(train, cfg, 'lr', ['lr'], default=1e-3)
    _merge_alias(train, cfg, 'weight_decay', ['weight_decay'], default=1e-4)
    _merge_alias(train, cfg, 'log_interval', ['log_interval'], default=50)
    _merge_alias(train, cfg, 'device', ['device'], default='cuda')
    _merge_alias(train, cfg, 'optimizer', ['optimizer'], default='sgd')
    _merge_alias(train, cfg, 'momentum', ['momentum'], default=0.9)
    _merge_alias(train, cfg, 'nesterov', ['nesterov'], default=True)
    _merge_alias(train, cfg, 'scheduler', ['scheduler', 'lr_scheduler'], default='cosine')
    _merge_alias(train, cfg, 'warmup_epochs', ['warmup_epochs'], default=0)
    _merge_alias(train, cfg, 'min_lr', ['min_lr'], default=0.0)
    _merge_alias(train, cfg, 'plateau_mode', ['plateau_mode'], default='max')
    _merge_alias(train, cfg, 'plateau_factor', ['plateau_factor'], default=0.5)
    _merge_alias(train, cfg, 'plateau_patience', ['plateau_patience'], default=10)
    _merge_alias(train, cfg, 'plateau_threshold', ['plateau_threshold'], default=1e-4)
    _merge_alias(train, cfg, 'plateau_cooldown', ['plateau_cooldown'], default=0)
    _merge_alias(train, cfg, 'grad_clip_norm', ['grad_clip_norm'], default=0.0)
    _merge_alias(train, cfg, 'label_smoothing', ['label_smoothing'], default=0.0)
    _merge_alias(train, cfg, 'mixup_alpha', ['mixup_alpha'], default=0.0)
    _merge_alias(train, cfg, 'cutmix_alpha', ['cutmix_alpha'], default=0.0)
    _merge_alias(train, cfg, 'cutmix_prob', ['cutmix_prob'], default=0.5)
    _merge_alias(train, cfg, 'augmentation_mode', ['augmentation_mode'], default='mixup')
    _merge_alias(train, cfg, 'augmentation_prob', ['augmentation_prob'], default=1.0)
    _merge_alias(train, cfg, 'mixup_scope', ['mixup_scope'], default='same_superclass')
    _merge_alias(train, cfg, 'ema_enable', ['ema_enable'], default=False)
    _merge_alias(train, cfg, 'ema_decay', ['ema_decay'], default=0.999)
    _merge_alias(train, cfg, 'ema_update_after_step', ['ema_update_after_step'], default=0)
    _merge_alias(train, cfg, 'ema_use_for_eval', ['ema_use_for_eval'], default=True)
    _merge_alias(train, cfg, 'lambda_coarse', ['lambda_coarse'], default=1.0)
    _merge_alias(train, cfg, 'superclass_smooth_alpha', ['superclass_smooth_alpha'], default=0.0)
    _merge_alias(train, cfg, 'early_stop_patience', ['early_stop_patience', 'early_stopping_patience'], default=0)
    _merge_alias(train, cfg, 'early_stop_min_delta', ['early_stop_min_delta'], default=0.0)
    _merge_alias(train, cfg, 'early_stop_metric', ['early_stop_metric', 'early_stopping_metric'], default='valid_top1')
    _merge_alias(train, cfg, 'early_stop_mode', ['early_stop_mode', 'early_stopping_mode'], default='max')
    _merge_alias(
        train,
        cfg,
        'early_stop_top1_weight',
        ['early_stop_top1_weight', 'early_stopping_top1_weight', 'top1_weight'],
        default=0.5,
    )
    _merge_alias(
        train,
        cfg,
        'early_stop_superclass_weight',
        ['early_stop_superclass_weight', 'early_stopping_superclass_weight', 'superclass_weight'],
        default=0.5,
    )

    cfg['data'] = data
    cfg['model'] = model
    cfg['train'] = train
    cfg['seed'] = int(cfg.get('seed', train.get('seed', 42)))
    cfg['output_dir'] = args.output

    return cfg
