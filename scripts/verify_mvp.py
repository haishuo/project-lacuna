#!/usr/bin/env python3
"""
Quick verification that all MVP components work.

Run this first to catch import/runtime errors before training.

Usage:
    python scripts/verify_mvp.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch


def test_feature_extractor():
    print("Testing ColumnFeatureExtractor...")
    from lacuna.data.feature_extractor import ColumnFeatureExtractor
    
    # Create test data with MNAR pattern
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x0': np.random.randn(n),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
    })
    # MNAR: high values of x1 are missing
    data.loc[data['x1'] > 0.5, 'x1'] = np.nan
    
    extractor = ColumnFeatureExtractor(max_cols=10)
    features, mask = extractor.extract(data)
    
    assert features.shape == (10, extractor.feature_dim)
    assert mask.shape == (10,)
    assert mask[:3].all() and not mask[3:].any()
    print(f"   ✓ Features shape: {features.shape}")
    print(f"   ✓ Mask correct: {mask[:5]}")


def test_generator():
    print("\nTesting SyntheticGenerator...")
    from lacuna.data.simulators.generator import SyntheticGenerator, GeneratorConfig
    
    config = GeneratorConfig(
        n_rows_range=(50, 100),
        n_cols_range=(3, 5),
        seed=42
    )
    gen = SyntheticGenerator(config)
    
    datasets, labels = gen.generate_training_data(10)
    
    assert len(datasets) == 10
    assert len(labels) == 10
    assert set(labels) == {0, 1}
    print(f"   ✓ Generated {len(datasets)} datasets")
    print(f"   ✓ Labels: {labels}")


def test_dataset():
    print("\nTesting LacunaDataset...")
    from lacuna.data.simulators.generator import SyntheticGenerator, GeneratorConfig
    from lacuna.data.dataset import LacunaDataset
    
    config = GeneratorConfig(n_cols_range=(3, 5), seed=42)
    gen = SyntheticGenerator(config)
    datasets, labels = gen.generate_training_data(20)
    
    ds = LacunaDataset(datasets, labels, max_cols=10)
    
    features, mask, label = ds[0]
    assert features.shape == (10, ds.feature_dim)
    assert mask.shape == (10,)
    assert label.shape == ()
    print(f"   ✓ Dataset size: {len(ds)}")
    print(f"   ✓ Sample shapes: features={features.shape}, mask={mask.shape}")


def test_model():
    print("\nTesting LacunaModel...")
    from lacuna.models.lacuna_mini import LacunaModel, LacunaMiniConfig
    
    config = LacunaMiniConfig(
        n_col_features=24,
        hidden_dim=64,  # Small for testing
        n_layers=2,
        max_cols=10
    )
    model = LacunaModel(config)
    
    # Fake batch
    batch_size = 4
    features = torch.randn(batch_size, 10, 24)
    mask = torch.ones(batch_size, 10, dtype=torch.bool)
    mask[:, 5:] = False
    
    logits = model(features, mask)
    
    assert logits.shape == (batch_size, 2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Parameters: {n_params:,}")


def test_baseline():
    print("\nTesting LogisticBaseline...")
    from lacuna.data.simulators.generator import SyntheticGenerator, GeneratorConfig
    from lacuna.baselines.logistic import LogisticBaseline
    
    config = GeneratorConfig(n_cols_range=(3, 5), seed=42)
    gen = SyntheticGenerator(config)
    
    train_datasets, train_labels = gen.generate_training_data(100)
    test_datasets, test_labels = gen.generate_training_data(20)
    
    baseline = LogisticBaseline(max_cols=10)
    baseline.fit(train_datasets, train_labels)
    metrics = baseline.evaluate(test_datasets, test_labels)
    
    assert 0 <= metrics['accuracy'] <= 1
    print(f"   ✓ Baseline accuracy: {metrics['accuracy']:.3f}")


def test_training_step():
    print("\nTesting training step...")
    from lacuna.data.simulators.generator import SyntheticGenerator, GeneratorConfig
    from lacuna.data.dataset import LacunaDataset
    from lacuna.models.lacuna_mini import LacunaModel, LacunaMiniConfig
    from lacuna.training.trainer import Trainer, TrainerConfig
    
    # Small data
    config = GeneratorConfig(n_cols_range=(3, 5), seed=42)
    gen = SyntheticGenerator(config)
    datasets, labels = gen.generate_training_data(32)
    
    ds = LacunaDataset(datasets, labels, max_cols=10)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    
    # Small model
    model_config = LacunaMiniConfig(
        n_col_features=ds.feature_dim,
        hidden_dim=32,
        n_layers=1,
        max_cols=10
    )
    model = LacunaModel(model_config)
    
    # One training step
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer_config = TrainerConfig(epochs=1)
    trainer = Trainer(model, trainer_config, device=device)
    
    metrics = trainer.train_epoch(loader)
    
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    print(f"   ✓ Train loss: {metrics['loss']:.4f}")
    print(f"   ✓ Train accuracy: {metrics['accuracy']:.3f}")


def main():
    print("=" * 60)
    print("LACUNA MVP VERIFICATION")
    print("=" * 60)
    
    try:
        test_feature_extractor()
        test_generator()
        test_dataset()
        test_model()
        test_baseline()
        test_training_step()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nReady to train! Run:")
        print("  python scripts/train_mvp.py --n-train 10000 --epochs 20")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()