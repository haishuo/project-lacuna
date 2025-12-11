#!/usr/bin/env python3
"""
Train Lacuna MVP model.

Usage:
    python scripts/train_mvp.py --n-train 10000 --n-val 2000 --epochs 20

This script:
1. Generates synthetic MAR/MNAR data
2. Trains the LacunaMini transformer
3. Trains the logistic regression baseline
4. Compares performance
"""

import argparse
from pathlib import Path
import sys

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lacuna.data.simulators.generator import SyntheticGenerator, GeneratorConfig
from lacuna.data.dataset import LacunaDataset
from lacuna.models.lacuna_mini import LacunaModel, LacunaMiniConfig
from lacuna.training.trainer import Trainer, TrainerConfig
from lacuna.baselines.logistic import LogisticBaseline


def main():
    parser = argparse.ArgumentParser(description="Train Lacuna MVP")
    parser.add_argument("--n-train", type=int, default=10000, 
                        help="Number of training datasets")
    parser.add_argument("--n-val", type=int, default=2000,
                        help="Number of validation datasets")
    parser.add_argument("--max-cols", type=int, default=15,
                        help="Maximum columns per dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, 
                        default=Path("/mnt/artifacts/project_lacuna/mvp"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Training config: {args}")
    print("=" * 60)
    
    # Generate data
    print("\n1. Generating synthetic data...")
    gen_config = GeneratorConfig(
        n_rows_range=(100, 500),
        n_cols_range=(5, args.max_cols),
        missing_rate_range=(0.1, 0.4),
        seed=args.seed
    )
    
    train_gen = SyntheticGenerator(gen_config)
    train_datasets, train_labels = train_gen.generate_training_data(args.n_train)
    print(f"   Generated {len(train_datasets)} training datasets")
    
    val_gen = SyntheticGenerator(GeneratorConfig(
        n_rows_range=(100, 500),
        n_cols_range=(5, args.max_cols),
        missing_rate_range=(0.1, 0.4),
        seed=args.seed + 1000
    ))
    val_datasets, val_labels = val_gen.generate_training_data(args.n_val)
    print(f"   Generated {len(val_datasets)} validation datasets")
    
    # Create PyTorch datasets
    print("\n2. Extracting features...")
    train_ds = LacunaDataset(train_datasets, train_labels, args.max_cols)
    val_ds = LacunaDataset(val_datasets, val_labels, args.max_cols)
    print(f"   Feature dim per column: {train_ds.feature_dim}")
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    
    # Train baseline first
    print("\n3. Training logistic regression baseline...")
    baseline = LogisticBaseline(max_cols=args.max_cols)
    baseline.fit(train_datasets, train_labels)
    baseline_metrics = baseline.evaluate(val_datasets, val_labels)
    print(f"   Baseline accuracy:     {baseline_metrics['accuracy']:.3f}")
    print(f"   Baseline MAR accuracy: {baseline_metrics['mar_accuracy']:.3f}")
    print(f"   Baseline MNAR accuracy:{baseline_metrics['mnar_accuracy']:.3f}")
    
    # Create model
    print("\n4. Creating LacunaMini model...")
    model_config = LacunaMiniConfig(
        n_col_features=train_ds.feature_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=4,
        max_cols=args.max_cols
    )
    model = LacunaModel(model_config, n_classes=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")
    
    # Train
    print("\n5. Training transformer...")
    trainer_config = TrainerConfig(
        lr=args.lr,
        epochs=args.epochs,
        warmup_epochs=2,
        patience=5,
        checkpoint_dir=args.output_dir
    )
    trainer = Trainer(model, trainer_config, device=device)
    history = trainer.train(train_loader, val_loader)
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nBaseline (Logistic Regression):")
    print(f"   Overall:  {baseline_metrics['accuracy']:.3f}")
    print(f"   MAR:      {baseline_metrics['mar_accuracy']:.3f}")
    print(f"   MNAR:     {baseline_metrics['mnar_accuracy']:.3f}")
    
    print(f"\nTransformer (LacunaMini):")
    print(f"   Overall:  {trainer.best_val_acc:.3f}")
    print(f"   MAR:      {max(history['val_mar_acc']):.3f}")
    print(f"   MNAR:     {max(history['val_mnar_acc']):.3f}")
    
    improvement = trainer.best_val_acc - baseline_metrics['accuracy']
    if improvement > 0:
        print(f"\n✓ Transformer beats baseline by {improvement:.3f}")
    else:
        print(f"\n✗ Baseline beats transformer by {-improvement:.3f}")
    
    # Save final model
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'baseline_metrics': baseline_metrics,
            'transformer_metrics': {
                'accuracy': trainer.best_val_acc,
                'history': history
            }
        }, args.output_dir / 'final_model.pt')
        print(f"\nModel saved to {args.output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    main()