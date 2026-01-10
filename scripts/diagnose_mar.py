#!/usr/bin/env python3
"""
Deep diagnostic for MAR detection failure.

Investigates:
1. What do MAR generators actually produce?
2. What features distinguish MAR from MCAR/MNAR in the data?
3. What does the model see (token distributions)?
4. Where does the model's prediction go wrong?

Usage:
    python scripts/diagnose_mar.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.core.rng import RNGState
from lacuna.core.types import MCAR, MAR, MNAR
from lacuna.generators import create_minimal_registry
from lacuna.generators.priors import GeneratorPrior
from lacuna.data.tokenization import tokenize_and_batch
from lacuna.data import create_default_catalog, SemiSyntheticDataLoader
from lacuna.models import create_lacuna_model
from lacuna.training import load_checkpoint


def analyze_generator_outputs():
    """
    Part 1: Examine what each generator actually produces.
    """
    print("\n" + "=" * 70)
    print("PART 1: GENERATOR OUTPUT ANALYSIS")
    print("=" * 70)
    
    registry = create_minimal_registry()
    rng = RNGState(seed=42)
    
    n, d = 200, 10  # Sample dataset size
    
    for gen in registry.generators:
        class_name = ["MCAR", "MAR", "MNAR"][gen.class_id]
        print(f"\n--- {gen.name} (class: {class_name}) ---")
        
        # Generate synthetic data
        X, R = gen.sample(rng.spawn(), n, d)
        
        # Basic missingness stats
        miss_rate = 1.0 - R.float().mean().item()
        print(f"  Overall missing rate: {miss_rate*100:.1f}%")
        
        # Per-column missing rates
        col_miss = 1.0 - R.float().mean(dim=0)
        print(f"  Per-column missing rates: min={col_miss.min():.2f}, max={col_miss.max():.2f}, std={col_miss.std():.3f}")
        
        # Key diagnostic: correlation between X values and missingness
        # For MCAR: no correlation
        # For MAR: missingness in col j correlates with VALUES in other columns
        # For MNAR: missingness in col j correlates with VALUES in col j itself
        
        print(f"  Missingness-value correlations:")
        
        # Self-correlation (MNAR signature): does missingness in col j depend on X[:,j]?
        self_corrs = []
        for j in range(d):
            if R[:, j].float().std() > 0.01:  # Has variation in missingness
                # Compare X values where observed vs full column
                # Since we have full X from generator, we can check
                miss_mask = ~R[:, j]
                if miss_mask.sum() > 5 and (~miss_mask).sum() > 5:
                    x_missing = X[miss_mask, j].mean().item()
                    x_observed = X[~miss_mask, j].mean().item()
                    self_corrs.append(x_missing - x_observed)
        
        if self_corrs:
            avg_self_corr = np.mean(np.abs(self_corrs))
            print(f"    Self-correlation (MNAR signal): {avg_self_corr:.3f}")
        
        # Cross-correlation (MAR signature): does missingness in col j depend on X[:,k] for k != j?
        cross_corrs = []
        for j in range(d):
            if R[:, j].float().std() > 0.01:
                miss_mask = ~R[:, j]
                for k in range(d):
                    if k != j and miss_mask.sum() > 5 and (~miss_mask).sum() > 5:
                        x_when_j_missing = X[miss_mask, k].mean().item()
                        x_when_j_observed = X[~miss_mask, k].mean().item()
                        cross_corrs.append(x_when_j_missing - x_when_j_observed)
        
        if cross_corrs:
            avg_cross_corr = np.mean(np.abs(cross_corrs))
            print(f"    Cross-correlation (MAR signal): {avg_cross_corr:.3f}")


def analyze_tokenized_features():
    """
    Part 2: What features are available in the tokenized data?
    """
    print("\n" + "=" * 70)
    print("PART 2: TOKENIZED FEATURE ANALYSIS")
    print("=" * 70)
    
    registry = create_minimal_registry()
    rng = RNGState(seed=42)
    
    n, d = 100, 8
    max_rows, max_cols = 128, 16
    
    # Generate one batch per class
    for class_id, class_name in [(MCAR, "MCAR"), (MAR, "MAR"), (MNAR, "MNAR")]:
        print(f"\n--- {class_name} ---")
        
        # Get generators for this class
        class_gens = [g for g in registry.generators if g.class_id == class_id]
        
        # Generate datasets
        datasets = []
        for gen in class_gens:
            X, R = gen.sample(rng.spawn(), n, d)
            from lacuna.core.types import ObservedDataset
            ds = ObservedDataset(
                x=X * R.float(),  # Zero out missing
                r=R,
                n=n,
                d=d,
                feature_names=tuple(f"col_{i}" for i in range(d)),
                dataset_id=f"{gen.name}_sample",
                meta=None,
            )
            datasets.append(ds)
        
        # Tokenize
        gen_ids = [g.generator_id for g in class_gens]
        class_mapping = {g.generator_id: g.class_id for g in registry.generators}
        
        batch = tokenize_and_batch(
            datasets=datasets,
            max_rows=max_rows,
            max_cols=max_cols,
            generator_ids=gen_ids,
            class_mapping=class_mapping,
        )
        
        # Analyze token statistics
        tokens = batch.tokens  # [B, max_rows, max_cols, 4]
        row_mask = batch.row_mask  # [B, max_rows]
        col_mask = batch.col_mask  # [B, max_cols]
        
        # Token dimensions: [value, is_observed, mask_type, feature_id]
        values = tokens[:, :, :, 0]
        is_observed = tokens[:, :, :, 1]
        
        # What the model sees: statistics across the batch
        B = tokens.shape[0]
        for b in range(B):
            valid_rows = row_mask[b].sum().item()
            valid_cols = col_mask[b].sum().item()
            
            obs_mask = is_observed[b, :int(valid_rows), :int(valid_cols)]
            val_data = values[b, :int(valid_rows), :int(valid_cols)]
            
            obs_rate = obs_mask.mean().item()
            
            # Per-column observation rates
            col_obs_rates = obs_mask.mean(dim=0)
            
            # Key: can we detect cross-column patterns from tokens alone?
            # The model only sees: value (0 if missing), is_observed flag, mask_type, feature_id
            # It does NOT directly see: which other columns predict this missingness
            
            print(f"  Sample {b}: {valid_rows:.0f} rows x {valid_cols:.0f} cols, obs_rate={obs_rate:.2f}")
            print(f"    Col obs rates: {col_obs_rates[:int(valid_cols)].tolist()[:5]}...")


def analyze_model_internals(ckpt_path: str = None):
    """
    Part 3: What does the model actually compute for MAR vs others?
    """
    print("\n" + "=" * 70)
    print("PART 3: MODEL INTERNAL ANALYSIS")
    print("=" * 70)
    
    if ckpt_path is None:
        # Find most recent checkpoint
        runs_dir = Path("/mnt/artifacts/project_lacuna/runs")
        checkpoints = list(runs_dir.glob("*/checkpoints/best_model.pt"))
        if not checkpoints:
            print("  No checkpoints found, skipping model analysis")
            return
        ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    print(f"  Loading: {ckpt_path}")
    
    ckpt = load_checkpoint(ckpt_path)
    model = create_lacuna_model(
        hidden_dim=128, evidence_dim=64, n_layers=4, n_heads=4,
        max_cols=48, dropout=0.2
    )
    model.load_state_dict(ckpt.model_state)
    model.eval()
    model.cuda()
    
    # Generate controlled test cases
    registry = create_minimal_registry()
    rng = RNGState(seed=12345)
    
    n, d = 100, 8
    
    print("\n  Generating controlled test cases...")
    
    results_by_class = {MCAR: [], MAR: [], MNAR: []}
    
    for gen in registry.generators:
        X, R = gen.sample(rng.spawn(), n, d)
        
        from lacuna.core.types import ObservedDataset
        ds = ObservedDataset(
            x=X * R.float(),
            r=R,
            n=n,
            d=d,
            feature_names=tuple(f"col_{i}" for i in range(d)),
            dataset_id=f"{gen.name}_test",
            meta=None,
        )
        
        class_mapping = {g.generator_id: g.class_id for g in registry.generators}
        batch = tokenize_and_batch(
            datasets=[ds],
            max_rows=128,
            max_cols=48,
            generator_ids=[gen.generator_id],
            class_mapping=class_mapping,
        )
        
        batch = batch.to('cuda')
        
        with torch.no_grad():
            output = model(batch)
            probs = output.posterior.p_class[0].cpu()  # [3]
            evidence = output.evidence[0].cpu()  # [evidence_dim]
        
        results_by_class[gen.class_id].append({
            'gen_name': gen.name,
            'probs': probs,
            'evidence': evidence,
            'pred': probs.argmax().item(),
        })
    
    # Analyze predictions
    class_names = ["MCAR", "MAR", "MNAR"]
    
    for class_id in [MCAR, MAR, MNAR]:
        print(f"\n  {class_names[class_id]} generators:")
        for r in results_by_class[class_id]:
            pred_name = class_names[r['pred']]
            probs = r['probs']
            correct = "✓" if r['pred'] == class_id else "✗"
            print(f"    {r['gen_name']}: pred={pred_name} {correct}")
            print(f"      P(MCAR)={probs[0]:.3f}, P(MAR)={probs[1]:.3f}, P(MNAR)={probs[2]:.3f}")
    
    # Compare evidence vectors
    print("\n  Evidence vector analysis:")
    
    all_evidence = {c: [] for c in [MCAR, MAR, MNAR]}
    for class_id in [MCAR, MAR, MNAR]:
        for r in results_by_class[class_id]:
            all_evidence[class_id].append(r['evidence'])
    
    # Stack and compute class centroids
    for class_id in [MCAR, MAR, MNAR]:
        if all_evidence[class_id]:
            stacked = torch.stack(all_evidence[class_id])
            centroid = stacked.mean(dim=0)
            std = stacked.std(dim=0).mean()
            print(f"    {class_names[class_id]}: evidence norm={centroid.norm():.3f}, std={std:.3f}")
    
    # Compute pairwise distances between class centroids
    print("\n  Evidence centroid distances:")
    centroids = {}
    for class_id in [MCAR, MAR, MNAR]:
        if all_evidence[class_id]:
            centroids[class_id] = torch.stack(all_evidence[class_id]).mean(dim=0)
    
    for i in [MCAR, MAR, MNAR]:
        for j in [MCAR, MAR, MNAR]:
            if i < j and i in centroids and j in centroids:
                dist = (centroids[i] - centroids[j]).norm().item()
                print(f"    {class_names[i]} <-> {class_names[j]}: {dist:.3f}")


def analyze_mar_generators_detail():
    """
    Part 4: Deep dive into what MAR generators actually do.
    """
    print("\n" + "=" * 70)
    print("PART 4: MAR GENERATOR DEEP DIVE")
    print("=" * 70)
    
    registry = create_minimal_registry()
    
    # Find MAR generators
    mar_gens = [g for g in registry.generators if g.class_id == MAR]
    
    for gen in mar_gens:
        print(f"\n--- {gen.name} ---")
        print(f"  Class: {gen.__class__.__name__}")
        print(f"  Params: {gen.params}")
        
        # Generate multiple samples to understand the mechanism
        rng = RNGState(seed=42)
        
        n, d = 500, 10
        X, R = gen.sample(rng, n, d)
        
        print(f"\n  Sample analysis (n={n}, d={d}):")
        
        # Which columns have missingness?
        col_miss_rates = 1.0 - R.float().mean(dim=0)
        missing_cols = (col_miss_rates > 0.01).nonzero().flatten().tolist()
        print(f"    Columns with missingness: {missing_cols}")
        print(f"    Missing rates: {[f'{col_miss_rates[c]:.2f}' for c in missing_cols]}")
        
        # For each column with missingness, check what predicts it
        for target_col in missing_cols[:3]:  # First 3
            miss_mask = ~R[:, target_col]
            n_missing = miss_mask.sum().item()
            n_observed = (~miss_mask).sum().item()
            
            if n_missing < 10 or n_observed < 10:
                continue
            
            print(f"\n    Column {target_col} missingness analysis:")
            print(f"      Missing: {n_missing}, Observed: {n_observed}")
            
            # Check correlation with each other column's VALUES
            correlations = []
            for pred_col in range(d):
                x_when_missing = X[miss_mask, pred_col].mean().item()
                x_when_observed = X[~miss_mask, pred_col].mean().item()
                diff = x_when_missing - x_when_observed
                
                # Also compute actual correlation
                x_vals = X[:, pred_col].numpy()
                miss_indicator = miss_mask.float().numpy()
                corr = np.corrcoef(x_vals, miss_indicator)[0, 1]
                
                correlations.append((pred_col, diff, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            
            print(f"      Top predictors of missingness (by correlation with X):")
            for pred_col, diff, corr in correlations[:5]:
                marker = "*SELF*" if pred_col == target_col else ""
                print(f"        Col {pred_col}: corr={corr:+.3f}, mean_diff={diff:+.3f} {marker}")


def analyze_what_model_sees_for_mar():
    """
    Part 5: The critical question - what information about MAR 
    is actually preserved in the tokenized input?
    """
    print("\n" + "=" * 70)
    print("PART 5: WHAT INFORMATION SURVIVES TOKENIZATION?")
    print("=" * 70)
    
    print("""
    The tokenization produces, for each cell:
      - value: the observed value (0 if missing)
      - is_observed: 1.0 if observed, 0.0 if missing
      - mask_type: natural (0) vs artificial (1)
      - feature_id: column index normalized to [0,1]
    
    For MAR detection, the model needs to detect:
      "Missingness in column j is predicted by VALUES in column k (k ≠ j)"
    
    But after tokenization:
      - The model sees which cells are missing (is_observed=0)
      - The model sees values for observed cells
      - The model does NOT directly see: "missingness pattern correlates with values"
    
    The model must LEARN to compute cross-column correlations during forward pass.
    This requires attention between columns to compare:
      - Rows where col_j is missing
      - Values of col_k in those same rows
    """)
    
    # Demonstrate the information loss
    registry = create_minimal_registry()
    rng = RNGState(seed=42)
    
    # Get one MAR and one MNAR generator
    mar_gen = [g for g in registry.generators if g.class_id == MAR][0]
    mnar_gen = [g for g in registry.generators if g.class_id == MNAR][0]
    
    n, d = 100, 6
    
    print(f"\n  Comparing {mar_gen.name} vs {mnar_gen.name}:")
    
    for gen, name in [(mar_gen, "MAR"), (mnar_gen, "MNAR")]:
        X, R = gen.sample(rng.spawn(), n, d)
        
        # What the model receives after tokenization (simplified view)
        observed_values = X * R.float()  # Missing = 0
        
        print(f"\n  {name}:")
        
        # Compute what a human would use to distinguish MAR vs MNAR:
        
        # For column 0 (likely the target column with missingness)
        target_col = 0
        miss_mask = ~R[:, target_col]
        
        if miss_mask.sum() > 10:
            # MNAR signal: values in target col differ for missing vs observed rows
            # But wait - for missing rows, we don't SEE the target col value!
            # This is only detectable if we have the FULL X
            
            # MAR signal: values in OTHER columns differ for missing vs observed rows
            # This IS visible even after masking!
            
            print(f"    Target column {target_col}: {miss_mask.sum().item()} missing")
            
            # What the model can see: for rows where col 0 is missing,
            # what are the values in other columns?
            for other_col in range(1, min(4, d)):
                # Values in other_col, split by whether col 0 is missing
                vals_when_target_missing = observed_values[miss_mask, other_col]
                vals_when_target_observed = observed_values[~miss_mask, other_col]
                
                # But we also need to check if other_col is observed in those rows
                other_observed_when_target_missing = R[miss_mask, other_col]
                other_observed_when_target_observed = R[~miss_mask, other_col]
                
                # Only count rows where other_col is actually observed
                if other_observed_when_target_missing.sum() > 5 and other_observed_when_target_observed.sum() > 5:
                    mean_when_missing = vals_when_target_missing[other_observed_when_target_missing].mean().item()
                    mean_when_observed = vals_when_target_observed[other_observed_when_target_observed].mean().item()
                    diff = mean_when_missing - mean_when_observed
                    print(f"      Col {other_col} mean when col0 missing: {mean_when_missing:.3f}")
                    print(f"      Col {other_col} mean when col0 observed: {mean_when_observed:.3f}")
                    print(f"      Difference (MAR signal): {diff:+.3f}")


def main():
    print("=" * 70)
    print("MAR DETECTION DIAGNOSTIC")
    print("=" * 70)
    
    # Run all analyses
    analyze_generator_outputs()
    analyze_tokenized_features()
    analyze_mar_generators_detail()
    analyze_what_model_sees_for_mar()
    
    # Only run model analysis if checkpoint exists
    runs_dir = Path("/mnt/artifacts/project_lacuna/runs")
    checkpoints = list(runs_dir.glob("*/checkpoints/best_model.pt"))
    if checkpoints:
        ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        analyze_model_internals(str(ckpt_path))
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
    The MAR detection problem likely stems from:
    
    1. INFORMATION BOTTLENECK: After tokenization, the model must learn to
       compute cross-column correlations. This is possible but requires
       the attention mechanism to specifically learn this pattern.
    
    2. SIMILAR SURFACE STATISTICS: MAR and MNAR both produce non-random
       missingness patterns. Without computing cross-column correlations,
       they look similar (both "not MCAR").
    
    3. TRAINING SIGNAL: If MAR and MNAR produce similar token-level 
       statistics, the model may not get enough gradient signal to
       distinguish them.
    
    POTENTIAL FIXES:
    
    A. Add explicit cross-column correlation features to tokenization
    B. Add auxiliary loss that encourages learning cross-column patterns  
    C. Modify architecture to compute row-level cross-column comparisons
    D. Use reconstruction error differences (MAR should be reconstructable
       from other columns, MNAR should not be)
    """)


if __name__ == "__main__":
    main()