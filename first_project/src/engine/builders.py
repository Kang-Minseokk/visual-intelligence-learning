from pathlib import Path

import torch

from src.models.net.base import BaseNet
from src.models.net.densenet import DenseNet
from src.models.net.pyramidnet import PyramidNet
from src.models.net.wideresnet import WideResNet
from torch.utils.tensorboard import SummaryWriter
from src.engine.trainer import Trainer
from src.dataset.get_dataset import get_dataset_loaders

def _dataset_loader_kwargs(cfg, *, download=None):
    k_train = cfg['data'].get('k_train', -1)
    if download is None:
        download = cfg['data']['download']
    return {
        'root': cfg['data']['root'],
        'batch_size': int(cfg['data']['batch_size']),
        'num_workers': cfg['data']['num_workers'],
        'download': download,
        'seed': cfg['seed'],
        'k_train': k_train,
        'num_classes': int(cfg['model']['num_classes']),
        'dataset_name': str(cfg['data'].get('name', 'FashionMNIST')),
        'label_level': str(cfg['data'].get('label_level', 'coarse')),
        'randaugment_enable': bool(cfg['data'].get('randaugment_enable', False)),
        'randaugment_num_ops': int(cfg['data'].get('randaugment_num_ops', 2)),
        'randaugment_magnitude': int(cfg['data'].get('randaugment_magnitude', 9)),
    }

def build_dataset_loaders(cfg):
    return get_dataset_loaders(**_dataset_loader_kwargs(cfg))


def build_model(cfg, device, model_name: str, in_features=None):
    """
        모델을 만들어내는 빌더 함수입니다.
    """
    input_shape = eval(str(in_features))

    if model_name == "base":
        if isinstance(input_shape, (tuple, list)):
            base_in_features = int(input_shape[0])
        else:
            base_in_features = int(input_shape)

        model = BaseNet(
            in_features=base_in_features,
            hidden_features=int(cfg['model']['hidden_features']),
            depth=cfg['model']['depth'],            
            dropout=cfg['model']['dropout'],
            num_classes=cfg['model']['num_classes'],
        )
    elif model_name == "wideresnet":
        if len(input_shape) != 3:
            raise ValueError(f"WideResNet expects 3D image input (C,H,W), got: {input_shape}")

        model = WideResNet(
            in_channels=int(input_shape[0]),
            depth=int(cfg['model'].get('wrn_depth', 28)),
            widen_factor=int(cfg['model'].get('widen_factor', 2)),
            dropout=float(cfg['model'].get('dropout', 0.0)),
            num_classes=int(cfg['model']['num_classes']),
            num_coarse_classes=int(cfg['model'].get('num_coarse_classes', 20)),
            coarse_hidden_dim=int(cfg['model'].get('coarse_hidden_dim', 512)),
            coarse_dropout=float(cfg['model'].get('coarse_dropout', 0.1)),
        )
    elif model_name == "densenet":
        if len(input_shape) != 3:
            raise ValueError(f"DenseNet expects 3D image input (C,H,W), got: {input_shape}")

        block_config = cfg['model'].get('dense_block_config', [6, 12, 24, 16])
        model = DenseNet(
            in_channels=int(input_shape[0]),
            num_classes=int(cfg['model']['num_classes']),
            growth_rate=int(cfg['model'].get('growth_rate', 12)),
            block_config=tuple(int(v) for v in block_config),
            num_init_features=int(cfg['model'].get('num_init_features', 24)),
            bn_size=int(cfg['model'].get('bn_size', 4)),
            drop_rate=float(cfg['model'].get('dropout', 0.0)),
            compression=float(cfg['model'].get('compression', 0.5)),
        )
    elif model_name == "pyramidnet":
        if len(input_shape) != 3:
            raise ValueError(f"PyramidNet expects 3D image input (C,H,W), got: {input_shape}")

        model = PyramidNet(
            in_channels=int(input_shape[0]),
            num_classes=int(cfg['model']['num_classes']),
            depth=int(cfg['model'].get('pyramid_depth', 110)),
            alpha=int(cfg['model'].get('pyramid_alpha', 48)),
            dropout=float(cfg['model'].get('dropout', 0.0)),
        )
    elif model_name == "vit":
        from src.models.net.vit import ViT

        model = ViT(
            image_size=32,
            patch_size=16, 
            num_classes=int(cfg['model']['num_classes']),
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim = 2048,
            dropout=0.1,
            emb_dropout=0.1,                        
        )
    elif model_name == "cct":
        from src.models.net.cct import CCT

        if len(input_shape) != 3:
            raise ValueError(f"CCT expects 3D image input (C,H,W), got: {input_shape}")

        model = CCT(
            in_channels=int(input_shape[0]),
            image_size=int(input_shape[1]),
            num_classes=int(cfg['model']['num_classes']),
            num_coarse_classes=int(cfg['model'].get('num_coarse_classes', 20)),
            embedding_dim=int(cfg['model'].get('cct_embedding_dim', 256)),
            transformer_layers=int(cfg['model'].get('cct_transformer_layers', 7)),
            transformer_heads=int(cfg['model'].get('cct_transformer_heads', 4)),
            mlp_ratio=float(cfg['model'].get('cct_mlp_ratio', 2.0)),
            kernel_size=int(cfg['model'].get('cct_kernel_size', 3)),
            stride=int(cfg['model'].get('cct_stride', 1)),
            padding=int(cfg['model'].get('cct_padding', 1)),
            n_conv_layers=int(cfg['model'].get('cct_n_conv_layers', 1)),
            tokenizer_pooling_kernel_size=int(cfg['model'].get('cct_tokenizer_pooling_kernel_size', 3)),
            tokenizer_pooling_stride=int(cfg['model'].get('cct_tokenizer_pooling_stride', 2)),
            tokenizer_pooling_padding=int(cfg['model'].get('cct_tokenizer_pooling_padding', 1)),
            dropout=float(cfg['model'].get('dropout', 0.0)),
            emb_dropout=float(cfg['model'].get('emb_dropout', 0.0)),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    print(f"✅ Model Check!\n{model}")

    return model.to(device)


def build_trainer(model, device, out_dir, cfg, label_info=None):
    """
        학습과 관련된 메서드를 저장하고 있는 객체인
        Trainer를 만들어내는 빌더 함수입니다.
    """
    return Trainer(
        model=model,
        device=device,
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
        log_interval=cfg['train']['log_interval'],
        out_dir=out_dir,
        config=cfg['train'],
        label_info=label_info,
    )
    
    
def build_writer(cfg):
    log_dir = Path(cfg["output_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def build_device(cfg):
    return torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')