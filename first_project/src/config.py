from __future__ import annotations
import argparse, yaml
from pathlib import Path

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results and tensorboard logs') 
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    cfg['output_dir'] = args.output
    
    return cfg
