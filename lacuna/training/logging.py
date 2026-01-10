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
        step = metrics.get("step", metrics.get("epoch", "?"))
        
        if "val_loss" in metrics:
            # End of epoch validation summary
            epoch = metrics.get('epoch', '?')
            epoch_str = f"{epoch:3d}" if isinstance(epoch, int) else str(epoch)
            print(f"  Epoch {epoch_str} | "
                  f"train_loss: {metrics.get('train_loss', 0):.4f} | "
                  f"val_loss: {metrics['val_loss']:.4f} | "
                  f"val_acc_gen: {metrics.get('val_acc_generator', 0)*100:.1f}% | "
                  f"val_acc_cls: {metrics.get('val_acc_class', 0)*100:.1f}%")
        elif "loss_total" in metrics:
            # Training step (log periodically)
            step_num = metrics.get("step", 0)
            if step_num % 50 == 0:
                print(f"  Step {step_num:5d} | "
                      f"loss: {metrics['loss_total']:.4f} | "
                      f"acc_gen: {metrics.get('acc_generator', 0)*100:.1f}% | "
                      f"lr: {metrics.get('lr', 0):.2e}")
        
        # Write all metrics to log file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(f"{metrics}\n")
    
    return log