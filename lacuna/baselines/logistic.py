"""
lacuna.baselines.logistic

Purpose: Logistic regression baseline for comparison.

Design Principles:
- UNIX Philosophy: Do ONE thing well
- Fail fast and loud
- Target: <100 lines
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict

from ..data.feature_extractor import ColumnFeatureExtractor


class LogisticBaseline:
    """Logistic regression on flattened column features.
    
    This is the simplest possible baseline. If the transformer
    can't beat this, we haven't learned anything useful about
    cross-column interactions.
    """
    
    def __init__(self, max_cols: int):
        if max_cols is None:
            raise ValueError("max_cols is required")
        self.max_cols = max_cols
        self.extractor = ColumnFeatureExtractor(max_cols)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
    
    def _extract_flat_features(self, datasets: List[pd.DataFrame]) -> np.ndarray:
        """Extract and flatten features from datasets."""
        X = []
        for data in datasets:
            features, _ = self.extractor.extract(data)
            X.append(features.flatten())
        return np.array(X)
    
    def fit(self, datasets: List[pd.DataFrame], labels: np.ndarray):
        """Fit the baseline model.
        
        Args:
            datasets: List of DataFrames
            labels: Array of labels (0=MAR, 1=MNAR)
        """
        X = self._extract_flat_features(datasets)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, labels)
    
    def predict(self, datasets: List[pd.DataFrame]) -> np.ndarray:
        """Predict labels."""
        X = self._extract_flat_features(datasets)
        X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, datasets: List[pd.DataFrame]) -> np.ndarray:
        """Predict probabilities."""
        X = self._extract_flat_features(datasets)
        X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def evaluate(self, datasets: List[pd.DataFrame], 
                 labels: np.ndarray) -> Dict[str, float]:
        """Evaluate accuracy on test data."""
        preds = self.predict(datasets)
        
        accuracy = (preds == labels).mean()
        mar_mask = labels == 0
        mnar_mask = labels == 1
        
        mar_acc = (preds[mar_mask] == labels[mar_mask]).mean() if mar_mask.sum() > 0 else 0
        mnar_acc = (preds[mnar_mask] == labels[mnar_mask]).mean() if mnar_mask.sum() > 0 else 0
        
        return {
            'accuracy': accuracy,
            'mar_accuracy': mar_acc,
            'mnar_accuracy': mnar_acc
        }


# Quick test
if __name__ == "__main__":
    from ..data.simulators.generator import SyntheticGenerator, GeneratorConfig
    
    config = GeneratorConfig(n_cols_range=(5, 10), seed=42)
    gen = SyntheticGenerator(config)
    
    # Generate train and test
    train_datasets, train_labels = gen.generate_training_data(1000)
    
    test_gen = SyntheticGenerator(GeneratorConfig(n_cols_range=(5, 10), seed=123))
    test_datasets, test_labels = test_gen.generate_training_data(200)
    
    # Train baseline
    baseline = LogisticBaseline(max_cols=15)
    baseline.fit(train_datasets, train_labels)
    
    # Evaluate
    metrics = baseline.evaluate(test_datasets, test_labels)
    print(f"Baseline accuracy: {metrics['accuracy']:.3f}")
    print(f"MAR accuracy: {metrics['mar_accuracy']:.3f}")
    print(f"MNAR accuracy: {metrics['mnar_accuracy']:.3f}")