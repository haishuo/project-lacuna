"""
lacuna.training.logging

Logging utilities for training.
"""

from pathlib import Path
from typing import Callable


def create_logger(output_dir: Path) -> Callable[[dict], None]:
    """Create logging function for training metrics.
    
    Args:
        output_dir: Experiment output directory (must have 'logs' subdirectory).
    
    Returns:
        Logging callback function that accepts a metrics dict.
    """
    log_file = output_dir / "logs" / "training.log"
    
    def log(metrics: dict):
        if "val_loss" in metrics:
            # Validation metrics - show per-class accuracy breakdown
            epoch = metrics.get('epoch', '?')
            epoch_str = f"{epoch:3d}" if isinstance(epoch, int) else str(epoch)
            
            # Get per-class accuracies (these are what validate() actually returns)
            mcar_acc = metrics.get('val_mcar_acc', 0)
            mar_acc = metrics.get('val_mar_acc', 0)
            mnar_acc = metrics.get('val_mnar_acc', 0)
            overall_acc = metrics.get('val_acc', 0)
            
            print(f"  Epoch {epoch_str} | "
                  f"val_loss: {metrics['val_loss']:.4f} | "
                  f"val_acc: {overall_acc*100:.1f}% | "
                  f"MCAR: {mcar_acc*100:.1f}% | "
                  f"MAR: {mar_acc*100:.1f}% | "
                  f"MNAR: {mnar_acc*100:.1f}%")
        
        # Write all metrics to log file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"{metrics}\n")
    
    return log