#!/usr/bin/env python3
"""
Diagnostic script to analyze reconstruction error patterns.

This script investigates whether the MARHead cross-attention is producing
discriminative reconstruction errors that distinguish MAR from MNAR.

The theory:
    - Under MAR: Missing values ARE predictable from observed values in other columns
                 -> MARHead should have LOWER error than MCARHead
    - Under MNAR: Missing values are NOT predictable from other columns
                  (the value itself determines missingness)
                 -> MARHead should have SIMILAR or HIGHER error than MCARHead

If the reconstruction errors don't show this pattern, the MoE gating network
has no signal to work with, and MAR/MNAR classification becomes random guessing.

Usage:
    python scripts/diagnose_reconstruction.py --checkpoint path/to/best_model.pt
    python scripts/diagnose_reconstruction.py --config configs/training/semisynthetic_balanced.yaml
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.config import load_config
from lacuna.models import create_lacuna_model, LacunaModel
from lacuna.generators import create_minimal_registry
from lacuna.generators.priors import GeneratorPrior
from lacuna.data import create_default_catalog, SemiSyntheticDataLoader
from lacuna.core.types import MCAR, MAR, MNAR, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose reconstruction error patterns")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--n_batches", type=int, default=50, help="Number of batches to analyze")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    return parser.parse_args()


def load_model_and_config(args) -> Tuple[LacunaModel, dict]:
    """Load model from checkpoint or create fresh model from config."""
    
    if args.checkpoint:
        # Load from checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        
        # Try to load config from checkpoint directory
        config_path = checkpoint_path.parent.parent / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            # Use default config
            print("Warning: No config found in checkpoint directory, using defaults")
            config = None
        
        # Create model
        if config:
            model = create_lacuna_model(
                hidden_dim=config.model.hidden_dim,
                evidence_dim=config.model.evidence_dim,
                n_layers=config.model.n_layers,
                n_heads=config.model.n_heads,
                max_cols=config.data.max_cols,
                dropout=config.model.dropout,
            )
        else:
            model = create_lacuna_model()
        
        # Load weights - check for different key names
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume the checkpoint IS the state dict
            model.load_state_dict(checkpoint)
        print(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")
        
        return model, config
    
    elif args.config:
        # Create fresh model from config
        config = load_config(args.config)
        model = create_lacuna_model(
            hidden_dim=config.model.hidden_dim,
            evidence_dim=config.model.evidence_dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            max_cols=config.data.max_cols,
            dropout=config.model.dropout,
        )
        print("Created fresh model from config (no trained weights)")
        return model, config
    
    else:
        raise ValueError("Must provide either --checkpoint or --config")


def create_data_loader(config, seed: int) -> SemiSyntheticDataLoader:
    """Create data loader for diagnostic evaluation."""
    
    # Create generator registry
    registry = create_minimal_registry()
    prior = GeneratorPrior.uniform(registry)
    
    # Load validation datasets
    catalog = create_default_catalog()
    max_cols = config.data.max_cols if config else 32
    
    # Get dataset names from config or use defaults
    if config and config.data.val_datasets:
        dataset_names = config.data.val_datasets
    else:
        dataset_names = ["iris", "wine", "breast_cancer"]
    
    raw_datasets = []
    for name in dataset_names:
        if name in catalog:
            ds = catalog.load(name)
            if ds.d <= max_cols:
                raw_datasets.append(ds)
                print(f"  Loaded: {name} ({ds.n} samples, {ds.d} features)")
    
    if not raw_datasets:
        raise ValueError("No valid datasets found")
    
    # Create loader
    loader = SemiSyntheticDataLoader(
        raw_datasets=raw_datasets,
        registry=registry,
        prior=prior,
        max_rows=config.data.max_rows if config else 128,
        max_cols=max_cols,
        batch_size=16,
        batches_per_epoch=100,
        seed=seed,
    )
    
    return loader, registry


def analyze_reconstruction_errors(
    model: LacunaModel,
    loader: SemiSyntheticDataLoader,
    registry,
    n_batches: int,
    device: str,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Analyze reconstruction errors by mechanism class.
    
    Returns:
        Dict mapping class_name -> head_name -> list of errors
    """
    model.to(device)
    model.eval()
    
    # Storage: class_name -> head_name -> list of per-sample errors
    errors_by_class = {
        "MCAR": defaultdict(list),
        "MAR": defaultdict(list),
        "MNAR": defaultdict(list),
    }
    
    # Also track raw error values for detailed analysis
    raw_errors_by_class = {
        "MCAR": [],
        "MAR": [],
        "MNAR": [],
    }
    
    # Track predictions vs actuals
    predictions = []
    actuals = []
    
    print(f"\nAnalyzing {n_batches} batches...")
    
    # Create iterator from loader
    loader_iter = iter(loader)
    
    with torch.no_grad():
        for batch_idx in range(n_batches):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Reset iterator if we run out of batches
                loader_iter = iter(loader)
                batch = next(loader_iter)
            
            batch = batch.to(device)
            
            # Forward pass with reconstruction
            output = model(batch, compute_reconstruction=True, compute_decision=True)
            
            # Get class labels
            class_ids = batch.class_ids.cpu().numpy()
            
            # Get predictions
            pred_ids = output.posterior.p_class.argmax(dim=1).cpu().numpy()
            predictions.extend(pred_ids)
            actuals.extend(class_ids)
            
            # Get reconstruction errors (natural errors if available)
            recon_errors = output.posterior.reconstruction_errors
            
            if recon_errors is None:
                print(f"  Warning: No reconstruction errors in batch {batch_idx}")
                continue
            
            # Process each sample in the batch
            B = len(class_ids)
            for i in range(B):
                class_id = class_ids[i]
                class_name = CLASS_NAMES[class_id]
                
                # Store errors for each head
                sample_errors = {}
                for head_name, errors in recon_errors.items():
                    error_val = errors[i].item()
                    errors_by_class[class_name][head_name].append(error_val)
                    sample_errors[head_name] = error_val
                
                raw_errors_by_class[class_name].append(sample_errors)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{n_batches} batches")
    
    return errors_by_class, raw_errors_by_class, predictions, actuals


def compute_statistics(errors_by_class: Dict) -> Dict:
    """Compute summary statistics for reconstruction errors."""
    
    stats = {}
    
    for class_name, head_errors in errors_by_class.items():
        stats[class_name] = {}
        
        for head_name, errors in head_errors.items():
            if len(errors) == 0:
                continue
            
            errors_arr = np.array(errors)
            stats[class_name][head_name] = {
                "mean": np.mean(errors_arr),
                "std": np.std(errors_arr),
                "median": np.median(errors_arr),
                "min": np.min(errors_arr),
                "max": np.max(errors_arr),
                "n": len(errors_arr),
            }
    
    return stats


def compute_discriminative_ratios(stats: Dict) -> Dict:
    """
    Compute ratios that should discriminate MAR from MNAR.
    
    Key insight:
        ratio = MAR_head_error / MCAR_head_error
        
        Under MAR mechanism: ratio should be LOW (MAR head predicts well)
        Under MNAR mechanism: ratio should be HIGH (MAR head can't predict)
    """
    
    ratios = {}
    
    for class_name in ["MCAR", "MAR", "MNAR"]:
        if class_name not in stats:
            continue
        
        class_stats = stats[class_name]
        
        # Get MCAR head error as baseline
        if "mcar" not in class_stats:
            continue
        
        mcar_error = class_stats["mcar"]["mean"]
        
        ratios[class_name] = {}
        
        for head_name, head_stats in class_stats.items():
            if head_name == "mcar":
                ratios[class_name][head_name] = 1.0
            else:
                # Ratio relative to MCAR
                if mcar_error > 1e-8:
                    ratios[class_name][head_name] = head_stats["mean"] / mcar_error
                else:
                    ratios[class_name][head_name] = float("inf")
    
    return ratios


def print_analysis_report(
    stats: Dict,
    ratios: Dict,
    predictions: List[int],
    actuals: List[int],
):
    """Print a detailed analysis report."""
    
    print("\n" + "=" * 70)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("=" * 70)
    
    # 1. Raw statistics
    print("\n1. MEAN RECONSTRUCTION ERRORS BY CLASS AND HEAD")
    print("-" * 70)
    
    # Get all head names
    all_heads = set()
    for class_stats in stats.values():
        all_heads.update(class_stats.keys())
    all_heads = sorted(all_heads)
    
    # Print header
    header = "Class".ljust(8) + "".join(h.ljust(15) for h in all_heads)
    print(header)
    print("-" * len(header))
    
    for class_name in ["MCAR", "MAR", "MNAR"]:
        if class_name not in stats:
            continue
        
        row = class_name.ljust(8)
        for head in all_heads:
            if head in stats[class_name]:
                mean = stats[class_name][head]["mean"]
                std = stats[class_name][head]["std"]
                cell = "{:.4f}+/-{:.4f}".format(mean, std)
                row += cell.ljust(15)
            else:
                row += "N/A".ljust(15)
        print(row)
    
    # 2. Error ratios (discriminative signal)
    print("\n2. ERROR RATIOS (head_error / mcar_error)")
    print("-" * 70)
    print("   Theory: Under MAR, 'mar' ratio should be < 1 (cross-attention helps)")
    print("           Under MNAR, 'mar' ratio should be >= 1 (cross-attention doesn't help)")
    print()
    
    header = "Class".ljust(8) + "".join(h.ljust(12) for h in all_heads)
    print(header)
    print("-" * len(header))
    
    for class_name in ["MCAR", "MAR", "MNAR"]:
        if class_name not in ratios:
            continue
        
        row = class_name.ljust(8)
        for head in all_heads:
            if head in ratios[class_name]:
                ratio = ratios[class_name][head]
                cell = "{:.4f}".format(ratio)
                row += cell.ljust(12)
            else:
                row += "N/A".ljust(12)
        print(row)
    
    # 3. Key discriminative comparison
    print("\n3. KEY DISCRIMINATIVE SIGNAL: MAR head error ratio")
    print("-" * 70)
    
    mar_ratios = {}
    for class_name in ["MCAR", "MAR", "MNAR"]:
        if class_name in ratios and "mar" in ratios[class_name]:
            mar_ratios[class_name] = ratios[class_name]["mar"]
    
    if len(mar_ratios) >= 2:
        mcar_ratio = mar_ratios.get("MCAR", float("nan"))
        mar_ratio = mar_ratios.get("MAR", float("nan"))
        mnar_ratio = mar_ratios.get("MNAR", float("nan"))
        
        print("   Under MCAR mechanism: MAR/MCAR ratio = {:.4f}".format(mcar_ratio))
        print("   Under MAR mechanism:  MAR/MCAR ratio = {:.4f}".format(mar_ratio))
        print("   Under MNAR mechanism: MAR/MCAR ratio = {:.4f}".format(mnar_ratio))
        
        print()
        
        # Check if the signal is discriminative
        if "MAR" in mar_ratios and "MNAR" in mar_ratios:
            mar_under_mar = mar_ratios["MAR"]
            mar_under_mnar = mar_ratios["MNAR"]
            
            if mar_under_mar < mar_under_mnar:
                gap = mar_under_mnar - mar_under_mar
                print("   [OK] Signal IS discriminative: gap = {:.4f}".format(gap))
                print("     (MAR mechanism has lower MAR-head ratio than MNAR mechanism)")
            else:
                print("   [FAIL] Signal is NOT discriminative!")
                print("     MAR mechanism has HIGHER MAR-head ratio than MNAR mechanism")
                print("     This explains why the model can't distinguish MAR from MNAR.")
    
    # 4. Confusion matrix
    print("\n4. CLASSIFICATION CONFUSION MATRIX")
    print("-" * 70)
    
    # Build confusion matrix
    n_classes = 3
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for pred, actual in zip(predictions, actuals):
        confusion[actual, pred] += 1
    
    # Print header
    header_label = "Actual / Pred"
    header_row = header_label.ljust(14)
    for i in range(n_classes):
        header_row += CLASS_NAMES[i].ljust(12)
    print(header_row)
    
    # Print rows
    for i in range(n_classes):
        row = CLASS_NAMES[i].ljust(14)
        total = confusion[i].sum()
        for j in range(n_classes):
            count = confusion[i, j]
            pct = 100 * count / total if total > 0 else 0
            cell = "{:4d} ({:4.1f}%)".format(count, pct)
            row += cell.ljust(12)
        print(row)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(n_classes):
        total = confusion[i].sum()
        correct = confusion[i, i]
        acc = 100 * correct / total if total > 0 else 0
        print("  {}: {:.1f}%".format(CLASS_NAMES[i], acc))
    
    # 5. Diagnosis
    print("\n5. DIAGNOSIS")
    print("-" * 70)
    
    if "MAR" in mar_ratios and "MNAR" in mar_ratios:
        mar_under_mar = mar_ratios["MAR"]
        mar_under_mnar = mar_ratios["MNAR"]
        
        if mar_under_mar >= mar_under_mnar:
            print("   PROBLEM IDENTIFIED: Reconstruction errors are NOT discriminative.")
            print()
            print("   Possible causes:")
            print("   1. MARHead cross-attention isn't learning to predict from observed values")
            print("   2. MAR generators aren't creating strong predictor->target dependencies")
            print("   3. The 'natural errors' aren't being computed on truly missing cells")
            print("   4. Token representations have lost raw value information by encoder output")
            print()
            print("   Recommended next steps:")
            print("   a. Check that MAR generators have strong alpha1 coefficients")
            print("   b. Verify MARHead attention weights (are they attending to predictors?)")
            print("   c. Add direct MAR/MCAR error difference as explicit feature to classifier")
        else:
            gap = mar_under_mnar - mar_under_mar
            if gap < 0.1:
                print("   WARNING: Reconstruction signal is weak (gap < 0.1)")
                print("   The MoE gating network may not have enough signal to discriminate.")
                print()
                print("   Consider:")
                print("   - Strengthening MAR generators (larger alpha1)")
                print("   - Using MARMultiColumn generators for stronger signal")
                print("   - Adding explicit error-difference features to classifier")
            else:
                print("   Reconstruction signal appears discriminative.")
                print("   If MAR accuracy is still low, check:")
                print("   - Whether MoE is receiving the reconstruction errors")
                print("   - Whether error features are weighted appropriately in gating")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("LACUNA RECONSTRUCTION DIAGNOSTICS")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model, config = load_model_and_config(args)
    
    # Create data loader
    print("\nCreating data loader...")
    loader, registry = create_data_loader(config, args.seed)
    
    # Analyze reconstruction errors
    errors_by_class, raw_errors, predictions, actuals = analyze_reconstruction_errors(
        model=model,
        loader=loader,
        registry=registry,
        n_batches=args.n_batches,
        device=args.device,
    )
    
    # Compute statistics
    stats = compute_statistics(errors_by_class)
    ratios = compute_discriminative_ratios(stats)
    
    # Print report
    print_analysis_report(stats, ratios, predictions, actuals)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()