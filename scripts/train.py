#!/usr/bin/env python3
"""
Lacuna Training Script (MVP - Fully Synthetic)

Usage:
    python scripts/train.py --config configs/training/minimal.yaml
    python scripts/train.py --config configs/training/default.yaml --device cuda
    python scripts/train.py --config configs/training/default.yaml --dry-run
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config.schema import LacunaConfig
from lacuna.config.load import load_config, save_config
from lacuna.generators import create_minimal_registry, GeneratorPrior
from lacuna.data.batching import SyntheticDataLoader
from lacuna.models.assembly import LacunaModel
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.training.checkpoint import save_checkpoint, CheckpointData


def parse_args():
    parser = argparse.ArgumentParser(description="Train Lacuna model on synthetic data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    return parser.parse_args()


def setup_experiment(config: LacunaConfig, name: str = None) -> Path:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = name or f"lacuna_{timestamp}"
    
    output_dir = Path(config.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    return output_dir


def create_logger(output_dir: Path):
    """Create logging function."""
    log_file = output_dir / "logs" / "training.log"
    
    def log(metrics: dict):
        step = metrics.get("step", metrics.get("epoch", "?"))
        
        if "val_loss" in metrics:
            print(f"  Epoch {metrics.get('epoch', '?'):3d} | "
                  f"train_loss: {metrics.get('train_loss', 0):.4f} | "
                  f"val_loss: {metrics['val_loss']:.4f} | "
                  f"val_acc_gen: {metrics.get('val_acc_generator', 0)*100:.1f}% | "
                  f"val_acc_cls: {metrics.get('val_acc_class', 0)*100:.1f}%")
        elif "loss_total" in metrics and metrics.get("step", 0) % 50 == 0:
            print(f"  Step {step:5d} | "
                  f"loss: {metrics['loss_total']:.4f} | "
                  f"acc_gen: {metrics.get('acc_generator', 0)*100:.1f}% | "
                  f"lr: {metrics.get('lr', 0):.2e}")
        
        with open(log_file, "a") as f:
            f.write(f"{metrics}\n")
    
    return log


def main():
    args = parse_args()
    
    print("=" * 60)
    print("LACUNA TRAINING (MVP - Fully Synthetic)")
    print("=" * 60)
    
    # Load config
    print(f"\nLoading config: {args.config}")
    config = load_config(args.config)
    
    # Apply overrides
    if args.device:
        config.device = args.device
    if args.seed:
        config.seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Setup experiment
    exp_dir = setup_experiment(config, args.name)
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    save_config(config, exp_dir / "config.yaml")
    
    # Print config summary
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Seed: {config.seed}")
    print(f"  Model: hidden={config.model.hidden_dim}, layers={config.model.n_layers}, heads={config.model.n_heads}")
    print(f"  Training: epochs={config.training.epochs}, lr={config.training.lr}, batch={config.training.batch_size}")
    print(f"  Data: n_range={config.data.n_range}, d_range={config.data.d_range}")
    print(f"  Data: max_rows={config.data.max_rows}, max_cols={config.data.max_cols}")
    print(f"  Generators: K={config.generator.n_generators}")
    
    if args.dry_run:
        print("\n[DRY RUN] Config validated. Exiting.")
        return
    
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
    
    # Create prior
    prior = GeneratorPrior.uniform(registry)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader = SyntheticDataLoader(
        registry=registry,
        prior=prior,
        n_range=config.data.n_range,
        d_range=config.data.d_range,
        max_rows=config.data.max_rows,
        max_cols=config.data.max_cols,
        batch_size=config.training.batch_size,
        batches_per_epoch=config.training.batches_per_epoch,
        seed=config.seed,
    )
    
    val_loader = SyntheticDataLoader(
        registry=registry,
        prior=prior,
        n_range=config.data.n_range,
        d_range=config.data.d_range,
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
    
    # Train
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
        metadata={"training_time": total_time, "best_val_acc": result['best_val_acc']},
    )
    save_checkpoint(final_ckpt, exp_dir / "checkpoints" / "final.pt")
    
    print(f"\nCheckpoints saved to: {exp_dir / 'checkpoints'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
