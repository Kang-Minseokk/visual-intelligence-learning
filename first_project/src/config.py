from __future__ import annotations
import argparse, yaml
from pathlib import Path

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base_config.yaml')
    parser.add_argument('--output', type=str, required=False, default=None,
                        help='Output directory. Defaults to logs/<config_stem> if omitted.')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # Priority: CLI --output > yaml output_dir > auto-derive from config filename
    if args.output is not None:
        cfg['output_dir'] = args.output
    elif not cfg.get('output_dir'):
        cfg['output_dir'] = f"logs/{Path(args.config).stem}"
    # else: keep yaml's output_dir value

    return cfg
