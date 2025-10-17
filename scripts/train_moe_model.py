#!/usr/bin/env python3
"""Train Lacuna MoE model

Usage:
    python scripts/train_moe_model.py \
        --config configs/training/synthetic_training.yaml \
        --data-dir /mnt/data/project_lacuna/synthetic
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train Lacuna model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    
    args = parser.parse_args()
    
    # TODO: Implement training
    # See spec section 5.2
    print(f"Training with config: {args.config}")
    print(f"Data: {args.data_dir}")


if __name__ == "__main__":
    main()
