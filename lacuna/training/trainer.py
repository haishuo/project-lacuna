"""
lacuna.training.trainer

Purpose: Training loop for Lacuna model.

Design Principles:
- UNIX Philosophy: Do ONE thing well
- Fail fast and loud
- Target: <250 lines
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
import time


@dataclass
class TrainerConfig:
    """Training configuration."""
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 20
    warmup_epochs: int = 2
    patience: int = 5  # Early stopping
    checkpoint_dir: Optional[Path] = None


class Trainer:
    """Trains LacunaModel."""
    
    def __init__(self, model: nn.Module, config: TrainerConfig, device: str = 'cuda'):
        if config is None:
            raise ValueError("config is required")
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler with warmup
        self.scheduler = None  # Set in train()
        
        # Tracking
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, mask, labels in train_loader:
            features = features.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(features, mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        for features, mask, labels in val_loader:
            features = features.to(self.device)
            mask = mask.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(features, mask)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        
        # Per-class accuracy
        mar_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == 0)
        mar_total = sum(1 for l in all_labels if l == 0)
        mnar_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l == 1)
        mnar_total = sum(1 for l in all_labels if l == 1)
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            'mar_accuracy': mar_correct / mar_total if mar_total > 0 else 0,
            'mnar_accuracy': mnar_correct / mnar_total if mnar_total > 0 else 0
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, list]:
        """Full training loop."""
        history = {'train_loss': [], 'train_acc': [], 
                   'val_loss': [], 'val_acc': [], 'val_mar_acc': [], 'val_mnar_acc': []}
        
        # Warmup + cosine annealing
        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = len(train_loader) * self.config.warmup_epochs
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"Training for {self.config.epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Log
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_mar_acc'].append(val_metrics['mar_accuracy'])
            history['val_mnar_acc'].append(val_metrics['mnar_accuracy'])
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} | "
                  f"MAR: {val_metrics['mar_accuracy']:.3f} MNAR: {val_metrics['mnar_accuracy']:.3f} | "
                  f"{epoch_time:.1f}s")
            
            # Early stopping
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.epochs_without_improvement = 0
                if self.config.checkpoint_dir:
                    self.save_checkpoint(self.config.checkpoint_dir / 'best.pt')
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training complete in {total_time:.1f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.3f}")
        
        return history
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']