from pathlib import Path

import torch

from src.models.net.base import BaseNet
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
    }

def build_dataset_loaders(cfg):
    return get_dataset_loaders(**_dataset_loader_kwargs(cfg))


def build_model(cfg, device, model_name: str, in_features=None):
    """
        모델을 만들어내는 빌더 함수입니다.
    """
    if model_name == "base":
        model = BaseNet(
            in_features=eval(str(in_features)),
            hidden_features=int(cfg['model']['hidden_features']),
            depth=cfg['model']['depth'],            
            dropout=cfg['model']['dropout'],
            num_classes=cfg['model']['num_classes'],
        )        
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    print(f"✅ Model Check!\n{model}")

    return model.to(device)


def build_trainer(model, device, out_dir, cfg):
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
    )
    
    
def build_writer(cfg):
    log_dir = Path(cfg["output_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def build_device(cfg):
    return torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')