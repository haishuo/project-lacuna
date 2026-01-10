#!/usr/bin/env python3
"""
Evaluate a trained Lacuna model and display confusion matrix.

Usage:
    python scripts/eval_confusion.py /mnt/artifacts/project_lacuna/runs/lacuna_semisyn_20260110_053134/checkpoints/best_model.pt
"""

import sys
from pathlib import Path
from collections import defaultdict

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.models import create_lacuna_model
from lacuna.training import load_checkpoint
from lacuna.generators import create_minimal_registry
from lacuna.generators.priors import GeneratorPrior
from lacuna.data import create_default_catalog, SemiSyntheticDataLoader


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/eval_confusion.py <checkpoint_path>")
        sys.exit(1)
    
    ckpt_path = Path(sys.argv[1])
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load checkpoint
    ckpt = load_checkpoint(ckpt_path)
    print(f"  Step: {ckpt.step}, Epoch: {ckpt.epoch}")
    print(f"  Best val_loss: {ckpt.best_val_loss:.4f}")
    print(f"  Best val_acc: {ckpt.best_val_acc:.4f}")
    
    # Create model with same architecture
    model = create_lacuna_model(
        hidden_dim=128,
        evidence_dim=64,
        n_layers=4,
        n_heads=4,
        max_cols=48,
        dropout=0.2,
    )
    model.load_state_dict(ckpt.model_state)
    model.eval()
    model.cuda()
    print(f"  Model loaded successfully")
    
    # Create validation loader
    registry = create_minimal_registry()
    prior = GeneratorPrior.uniform(registry)
    catalog = create_default_catalog()
    
    val_dataset_names = [
        'pulsar_stars', 'steel_plates', 'banknote',
        'heart_disease', 'ionosphere', 'parkinsons'
    ]
    
    print(f"\nLoading validation datasets...")
    val_datasets = []
    for name in val_dataset_names:
        try:
            ds = catalog.load(name)
            val_datasets.append(ds)
            print(f"  {name}: {ds.n} samples, {ds.d} features")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    
    val_loader = SemiSyntheticDataLoader(
        raw_datasets=val_datasets,
        registry=registry,
        prior=prior,
        max_rows=128,
        max_cols=48,
        batch_size=16,
        batches_per_epoch=100,  # More batches for stable estimate
        seed=999,
    )
    
    # Evaluate
    print(f"\nEvaluating on {len(val_loader)} batches...")
    
    class_names = ['MCAR', 'MAR', 'MNAR']
    confusion = defaultdict(lambda: defaultdict(int))
    total_correct = 0
    total_samples = 0
    
    # Also track confidence
    confidence_correct = []
    confidence_incorrect = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to('cuda')
            output = model(batch)
            
            probs = output.posterior.p_class  # [B, 3]
            preds = probs.argmax(dim=-1)
            confidences = probs.max(dim=-1).values
            
            for i, (true, pred, conf) in enumerate(zip(
                batch.class_ids.cpu(),
                preds.cpu(),
                confidences.cpu()
            )):
                true_idx = true.item()
                pred_idx = pred.item()
                confusion[true_idx][pred_idx] += 1
                total_samples += 1
                
                if true_idx == pred_idx:
                    total_correct += 1
                    confidence_correct.append(conf.item())
                else:
                    confidence_incorrect.append(conf.item())
    
    # Print results
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("(rows = true class, columns = predicted class)\n")
    
    # Header
    print(f"{'':8s}  {'MCAR':>6s}  {'MAR':>6s}  {'MNAR':>6s}  {'Total':>6s}  {'Acc':>6s}")
    print("-" * 50)
    
    # Rows
    for i, name in enumerate(class_names):
        row = [confusion[i][j] for j in range(3)]
        total = sum(row)
        acc = row[i] / total * 100 if total > 0 else 0
        print(f"{name:8s}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}  {total:6d}  {acc:5.1f}%")
    
    print("-" * 50)
    
    # Column totals
    col_totals = [sum(confusion[i][j] for i in range(3)) for j in range(3)]
    print(f"{'Pred Tot':8s}  {col_totals[0]:6d}  {col_totals[1]:6d}  {col_totals[2]:6d}")
    
    # Overall accuracy
    print(f"\nOverall Accuracy: {total_correct}/{total_samples} = {total_correct/total_samples*100:.1f}%")
    
    # Per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    
    for i, name in enumerate(class_names):
        tp = confusion[i][i]
        fn = sum(confusion[i][j] for j in range(3) if j != i)
        fp = sum(confusion[j][i] for j in range(3) if j != i)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{name}:")
        print(f"  Precision: {precision*100:.1f}%  (TP={tp}, FP={fp})")
        print(f"  Recall:    {recall*100:.1f}%  (TP={tp}, FN={fn})")
        print(f"  F1 Score:  {f1*100:.1f}%")
    
    # Confidence analysis
    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS")
    print("=" * 60)
    
    if confidence_correct:
        avg_conf_correct = sum(confidence_correct) / len(confidence_correct)
        print(f"Avg confidence on CORRECT predictions: {avg_conf_correct:.3f}")
    
    if confidence_incorrect:
        avg_conf_incorrect = sum(confidence_incorrect) / len(confidence_incorrect)
        print(f"Avg confidence on INCORRECT predictions: {avg_conf_incorrect:.3f}")
    
    # Common misclassifications
    print("\n" + "=" * 60)
    print("MISCLASSIFICATION PATTERNS")
    print("=" * 60)
    
    misclass = []
    for i in range(3):
        for j in range(3):
            if i != j and confusion[i][j] > 0:
                misclass.append((confusion[i][j], class_names[i], class_names[j]))
    
    misclass.sort(reverse=True)
    for count, true_name, pred_name in misclass:
        pct = count / total_samples * 100
        print(f"  {true_name} → {pred_name}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()