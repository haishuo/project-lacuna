#!/usr/bin/env python3
"""
Training script for semi-synthetic data.

Uses real datasets with synthetic missingness mechanisms.

Usage:
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml
    python scripts/train_semisynthetic.py --config configs/training/semisynthetic.yaml --name my_experiment
"""

import argparse
import time
from pathlib import Path
import torch
import yaml

from lacuna.config import LacunaConfig
from lacuna.models import LacunaModel
from lacuna.generators import create_minimal_registry
from lacuna.generators.priors import GeneratorPrior
from lacuna.data import (
    create_default_catalog,
    SemiSyntheticDataLoader,
)
from lacuna.training import (
    Trainer,
    TrainerConfig,
    CheckpointData,
    save_checkpoint,
    create_logger,
)


def load_raw_datasets(catalog, dataset_names: list, max_cols: int):
    """Load raw datasets from catalog, filtering by max_cols."""
    datasets = []
    skipped = []
    
    for name in dataset_names:
        try:
            raw = catalog.load(name)
            if raw.d <= max_cols:
                datasets.append(raw)
                print(f"    Loaded: {name} ({raw.n} samples, {raw.d} features)")
            else:
                skipped.append((name, raw.d))
        except Exception as e:
            print(f"    Warning: Could not load '{name}': {e}")
    
    if skipped:
        print(f"    Skipped (too many columns): {skipped}")
    
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Train Lacuna on semi-synthetic data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LACUNA TRAINING (Semi-Synthetic)")
    print("=" * 60)
    
    # Load config
    print(f"\nLoading config: {args.config}")
    config = LacunaConfig.from_yaml(args.config)
    
    # Create experiment directory
    exp_name = args.name or f"semisyn_{int(time.time())}"
    exp_dir = Path(config.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print(f"  Model: hidden={config.model.hidden_dim}, layers={config.model.n_layers}, heads={config.model.n_heads}")
    print(f"  Training: epochs={config.training.epochs}, lr={config.training.lr}, batch={config.training.batch_size}")
    print(f"  Data: max_rows={config.data.max_rows}, max_cols={config.data.max_cols}")
    
    # Check device
    if config.device == "cuda" and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, falling back to CPU")
        config.device = "cpu"
    
    # Set seed
    print(f"\nSetting seed: {config.seed}")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create registry
    print("\nCreating generator registry...")
    registry = create_minimal_registry()
    print(f"  Generators: {registry.K}")
    print(f"  Class distribution: {registry.class_counts()}")
    
    prior = GeneratorPrior.uniform(registry)
    
    # Load datasets
    print("\nLoading datasets from catalog...")
    catalog = create_default_catalog()
    
    # Get dataset names from config or use defaults
    train_datasets = getattr(config.data, 'train_datasets', None) or [
        "breast_cancer", "diabetes", "wine"
    ]
    val_datasets = getattr(config.data, 'val_datasets', None) or [
        "iris"  # Hold out for validation
    ]
    
    print("  Training datasets:")
    train_raw = load_raw_datasets(catalog, train_datasets, config.data.max_cols)
    
    print("  Validation datasets:")
    val_raw = load_raw_datasets(catalog, val_datasets, config.data.max_cols)
    
    if len(train_raw) == 0:
        print("\nError: No training datasets loaded. Exiting.")
        return
    
    if len(val_raw) == 0:
        print("\nWarning: No validation datasets. Using last training dataset.")
        val_raw = [train_raw[-1]]
        train_raw = train_raw[:-1]
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = SemiSyntheticDataLoader(
        raw_datasets=train_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=config.training.batches_per_epoch,
        seed=config.seed,
    )
    
    val_loader = SemiSyntheticDataLoader(
        raw_datasets=val_raw,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=config.training.val_batches,
        seed=config.seed + 1000000,
    )
    
    print(f"  Train: {len(train_loader)} batches/epoch")
    print(f"  Val: {len(val_loader)} batches")
    
    # Create model
    print("\nCreating model...")
    class_mapping = registry.get_class_mapping()
    model = LacunaModel.from_config(config, class_mapping)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} ({n_trainable:,} trainable)")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer_config = TrainerConfig(
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        grad_clip=config.training.grad_clip,
        epochs=config.training.epochs,
        warmup_steps=config.training.warmup_steps,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        checkpoint_dir=str(exp_dir / "checkpoints"),
        save_best_only=True,
    )
    
    trainer = Trainer(model, trainer_config, device=config.device)
    trainer.set_logger(create_logger(exp_dir))
    
    # Train!
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    result = trainer.fit(train_loader, val_loader)
    total_time = time.time() - start_time
    
    # Final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Epochs completed: {result['epochs_completed']}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Best val loss: {result['best_val_loss']:.4f}")
    print(f"  Best val acc (generator): {result['best_val_acc']*100:.1f}%")
    print(f"  Training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    
    # Save final checkpoint
    final_ckpt = CheckpointData(
        model_state=model.state_dict(),
        step=result['total_steps'],
        epoch=result['epochs_completed'],
        best_val_loss=result['best_val_loss'],
        metadata={
            "training_time": total_time,
            "best_val_acc": result['best_val_acc'],
            "data_mode": "semisynthetic",
            "train_datasets": [ds.name for ds in train_raw],
            "val_datasets": [ds.name for ds in val_raw],
        },
    )
    save_checkpoint(final_ckpt, exp_dir / "checkpoints" / "final.pt")
    
    print(f"\nCheckpoints saved to: {exp_dir / 'checkpoints'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
