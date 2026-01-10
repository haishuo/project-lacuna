#!/usr/bin/env python3
"""
Test script to verify missingness features discriminate between mechanisms.

This script:
1. Generates synthetic data with known MCAR, MAR, and MNAR mechanisms
2. Extracts missingness pattern features
3. Analyzes whether the features discriminate between mechanisms
4. Reports which features are most discriminative

Usage:
    python scripts/test_missingness_features.py
"""

import sys
from pathlib import Path

import torch
import numpy as np
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.types import MCAR, MAR, MNAR, CLASS_NAMES
from lacuna.generators import create_minimal_registry
from lacuna.generators.priors import GeneratorPrior
from lacuna.data import tokenize_and_batch
from lacuna.data.missingness_features import (
    extract_missingness_features,
    get_feature_names,
    MissingnessFeatureConfig,
)
from lacuna.core.rng import RNGState


def generate_test_batches(n_batches: int = 50, batch_size: int = 16, seed: int = 42):
    """Generate batches with known mechanism labels."""
    
    registry = create_minimal_registry()
    prior = GeneratorPrior.uniform(registry)
    
    # Map generator IDs to class IDs
    class_mapping = {g.generator_id: g.class_id for g in registry.generators}
    
    # Master RNG for reproducibility
    master_rng = RNGState(seed=seed)
    
    batches = []
    
    for batch_idx in range(n_batches):
        datasets = []
        generator_ids = []
        
        for i in range(batch_size):
            # Sample a generator using the prior
            sample_rng = master_rng.spawn()
            gen_id = prior.sample(sample_rng)
            generator = registry[gen_id]
            
            # Random dimensions for this sample
            n = int(sample_rng.randint(50, 200, (1,)).item())
            d = int(sample_rng.randint(5, 15, (1,)).item())
            
            # Generate observed data
            observed = generator.sample_observed(
                rng=sample_rng.spawn(),
                n=n,
                d=d,
                dataset_id=f"batch{batch_idx}_item{i}",
            )
            
            datasets.append(observed)
            generator_ids.append(gen_id)
        
        # Tokenize and batch
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=128,
            max_cols=32,
            generator_ids=generator_ids,
            class_mapping=class_mapping,
        )
        
        batches.append(batch)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Generated {batch_idx + 1}/{n_batches} batches")
    
    return batches, registry


def analyze_features(batches, feature_names):
    """Analyze features by mechanism class."""
    
    # Collect features by class
    features_by_class = {
        "MCAR": [],
        "MAR": [],
        "MNAR": [],
    }
    
    for batch in batches:
        # Extract features
        features = extract_missingness_features(
            batch.tokens,
            batch.row_mask,
            batch.col_mask,
        )
        
        # Group by class
        class_ids = batch.class_ids.numpy()
        features_np = features.numpy()
        
        for i, class_id in enumerate(class_ids):
            class_name = CLASS_NAMES[class_id]
            features_by_class[class_name].append(features_np[i])
    
    # Convert to arrays
    for class_name in features_by_class:
        if len(features_by_class[class_name]) > 0:
            features_by_class[class_name] = np.array(features_by_class[class_name])
        else:
            features_by_class[class_name] = np.array([]).reshape(0, len(feature_names))
    
    return features_by_class


def compute_discriminability(features_by_class, feature_names):
    """Compute how well each feature discriminates between classes."""
    
    n_features = len(feature_names)
    
    # Compute mean and std for each class
    stats = {}
    for class_name, features in features_by_class.items():
        if len(features) > 0:
            stats[class_name] = {
                "mean": np.mean(features, axis=0),
                "std": np.std(features, axis=0),
                "n": len(features),
            }
        else:
            stats[class_name] = {
                "mean": np.zeros(n_features),
                "std": np.zeros(n_features),
                "n": 0,
            }
    
    # Compute discriminability metrics
    results = []
    
    for i, name in enumerate(feature_names):
        # Get values for each class
        mcar_mean = stats["MCAR"]["mean"][i]
        mar_mean = stats["MAR"]["mean"][i]
        mnar_mean = stats["MNAR"]["mean"][i]
        
        mcar_std = stats["MCAR"]["std"][i]
        mar_std = stats["MAR"]["std"][i]
        mnar_std = stats["MNAR"]["std"][i]
        
        # Pooled std (avoid division by zero)
        pooled_std = np.sqrt((mcar_std**2 + mar_std**2 + mnar_std**2) / 3)
        
        # Effect sizes (Cohen's d style)
        if pooled_std > 1e-6:
            d_mcar_mar = abs(mcar_mean - mar_mean) / pooled_std
            d_mcar_mnar = abs(mcar_mean - mnar_mean) / pooled_std
            d_mar_mnar = abs(mar_mean - mnar_mean) / pooled_std
        else:
            d_mcar_mar = d_mcar_mnar = d_mar_mnar = 0
        
        # Overall discriminability
        overall = (d_mcar_mar + d_mcar_mnar + d_mar_mnar) / 3
        
        results.append({
            "name": name,
            "mcar_mean": mcar_mean,
            "mar_mean": mar_mean,
            "mnar_mean": mnar_mean,
            "d_mcar_mar": d_mcar_mar,
            "d_mcar_mnar": d_mcar_mnar,
            "d_mar_mnar": d_mar_mnar,
            "overall": overall,
        })
    
    return results, stats


def print_report(results, stats, feature_names):
    """Print analysis report."""
    
    print("\n" + "=" * 80)
    print("MISSINGNESS FEATURE DISCRIMINABILITY ANALYSIS")
    print("=" * 80)
    
    # Sample sizes
    print("\n1. SAMPLE SIZES")
    print("-" * 40)
    for class_name, class_stats in stats.items():
        print(f"  {class_name}: {class_stats['n']} samples")
    
    # Feature means by class
    print("\n2. FEATURE MEANS BY CLASS")
    print("-" * 80)
    print(f"{'Feature':<25} {'MCAR':>12} {'MAR':>12} {'MNAR':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['mcar_mean']:>12.4f} {r['mar_mean']:>12.4f} {r['mnar_mean']:>12.4f}")
    
    # Effect sizes
    print("\n3. EFFECT SIZES (Cohen's d)")
    print("-" * 80)
    print(f"{'Feature':<25} {'MCAR-MAR':>12} {'MCAR-MNAR':>12} {'MAR-MNAR':>12} {'Overall':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<25} {r['d_mcar_mar']:>12.3f} {r['d_mcar_mnar']:>12.3f} {r['d_mar_mnar']:>12.3f} {r['overall']:>12.3f}")
    
    # Top discriminative features
    print("\n4. TOP DISCRIMINATIVE FEATURES")
    print("-" * 80)
    
    sorted_results = sorted(results, key=lambda x: x["overall"], reverse=True)
    
    print("\nOverall (all pairwise comparisons):")
    for i, r in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {r['name']}: d = {r['overall']:.3f}")
    
    print("\nFor MCAR vs MAR discrimination:")
    sorted_mcar_mar = sorted(results, key=lambda x: x["d_mcar_mar"], reverse=True)
    for i, r in enumerate(sorted_mcar_mar[:5]):
        print(f"  {i+1}. {r['name']}: d = {r['d_mcar_mar']:.3f}")
    
    print("\nFor MAR vs MNAR discrimination:")
    sorted_mar_mnar = sorted(results, key=lambda x: x["d_mar_mnar"], reverse=True)
    for i, r in enumerate(sorted_mar_mnar[:5]):
        print(f"  {i+1}. {r['name']}: d = {r['d_mar_mnar']:.3f}")
    
    # Interpretation
    print("\n5. INTERPRETATION")
    print("-" * 80)
    
    # Check if features are discriminative
    best_mcar_mar = sorted_mcar_mar[0]
    best_mar_mnar = sorted_mar_mnar[0]
    
    if best_mcar_mar["d_mcar_mar"] < 0.2:
        print("  [WARN] No features strongly discriminate MCAR from MAR (d < 0.2)")
    elif best_mcar_mar["d_mcar_mar"] < 0.5:
        print(f"  [OK] Weak MCAR-MAR discrimination via {best_mcar_mar['name']} (d = {best_mcar_mar['d_mcar_mar']:.3f})")
    else:
        print(f"  [GOOD] Strong MCAR-MAR discrimination via {best_mcar_mar['name']} (d = {best_mcar_mar['d_mcar_mar']:.3f})")
    
    if best_mar_mnar["d_mar_mnar"] < 0.2:
        print("  [WARN] No features strongly discriminate MAR from MNAR (d < 0.2)")
    elif best_mar_mnar["d_mar_mnar"] < 0.5:
        print(f"  [OK] Weak MAR-MNAR discrimination via {best_mar_mnar['name']} (d = {best_mar_mnar['d_mar_mnar']:.3f})")
    else:
        print(f"  [GOOD] Strong MAR-MNAR discrimination via {best_mar_mnar['name']} (d = {best_mar_mnar['d_mar_mnar']:.3f})")
    
    # Overall assessment
    print("\n6. OVERALL ASSESSMENT")
    print("-" * 80)
    
    avg_mcar_mar = np.mean([r["d_mcar_mar"] for r in results])
    avg_mar_mnar = np.mean([r["d_mar_mnar"] for r in results])
    avg_mcar_mnar = np.mean([r["d_mcar_mnar"] for r in results])
    
    print(f"  Average effect size MCAR vs MAR:  {avg_mcar_mar:.3f}")
    print(f"  Average effect size MAR vs MNAR:  {avg_mar_mnar:.3f}")
    print(f"  Average effect size MCAR vs MNAR: {avg_mcar_mnar:.3f}")
    
    if avg_mcar_mar > 0.3 and avg_mar_mnar > 0.3:
        print("\n  [SUCCESS] Missingness features provide discriminative signal for all class pairs!")
    elif avg_mcar_mar > 0.2 or avg_mar_mnar > 0.2:
        print("\n  [PARTIAL] Some discriminative signal exists, but may need stronger generators or more features.")
    else:
        print("\n  [WEAK] Missingness features alone may not be sufficient. Consider:")
        print("    - Strengthening MAR generators (larger alpha1)")
        print("    - Adding MARMultiColumn generators")
        print("    - Combining with reconstruction error signals")


def main():
    print("=" * 80)
    print("TESTING MISSINGNESS PATTERN FEATURES")
    print("=" * 80)
    
    print("\nGenerating test batches...")
    batches, registry = generate_test_batches(n_batches=30, batch_size=16, seed=42)
    print(f"  Generated {len(batches)} batches total")
    
    # Get feature names
    feature_names = get_feature_names()
    print(f"  Extracting {len(feature_names)} features per sample")
    
    print("\nAnalyzing features by mechanism class...")
    features_by_class = analyze_features(batches, feature_names)
    
    print("\nComputing discriminability metrics...")
    results, stats = compute_discriminability(features_by_class, feature_names)
    
    print_report(results, stats, feature_names)
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()