"""
Tests for lacuna.training.loss

Verify loss functions behave correctly.
"""

import pytest
import torch
from lacuna.training.loss import (
    generator_cross_entropy,
    class_cross_entropy,
    combined_loss,
    compute_accuracy,
    compute_topk_accuracy,
)
from lacuna.core.types import PosteriorResult


class TestGeneratorCrossEntropy:
    """Tests for generator_cross_entropy."""
    
    def test_perfect_prediction_low_loss(self):
        B, K = 4, 6
        targets = torch.tensor([0, 1, 2, 3])
        
        # Strong logits for correct classes
        logits = torch.zeros(B, K)
        for i, t in enumerate(targets):
            logits[i, t] = 10.0
        
        loss = generator_cross_entropy(logits, targets)
        assert loss.item() < 0.01
    
    def test_uniform_prediction_high_loss(self):
        B, K = 4, 6
        targets = torch.tensor([0, 1, 2, 3])
        logits = torch.zeros(B, K)  # Uniform
        
        loss = generator_cross_entropy(logits, targets)
        # Should be close to log(K)
        expected = torch.log(torch.tensor(float(K)))
        assert abs(loss.item() - expected.item()) < 0.1
    
    def test_reduction_none(self):
        B, K = 4, 6
        logits = torch.randn(B, K)
        targets = torch.randint(0, K, (B,))
        
        loss = generator_cross_entropy(logits, targets, reduction="none")
        assert loss.shape == (B,)
    
    def test_reduction_sum(self):
        B, K = 4, 6
        logits = torch.randn(B, K)
        targets = torch.randint(0, K, (B,))
        
        loss_mean = generator_cross_entropy(logits, targets, reduction="mean")
        loss_sum = generator_cross_entropy(logits, targets, reduction="sum")
        
        assert abs(loss_sum.item() - loss_mean.item() * B) < 1e-5


class TestClassCrossEntropy:
    """Tests for class_cross_entropy."""
    
    def test_correct_class_low_loss(self):
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        
        # High probability for correct class
        p_class = torch.zeros(B, 3)
        for i, t in enumerate(targets):
            p_class[i, t] = 0.99
            p_class[i, (t + 1) % 3] = 0.005
            p_class[i, (t + 2) % 3] = 0.005
        
        loss = class_cross_entropy(p_class, targets)
        assert loss.item() < 0.02
    
    def test_uniform_high_loss(self):
        B = 4
        targets = torch.tensor([0, 1, 2, 0])
        p_class = torch.ones(B, 3) / 3
        
        loss = class_cross_entropy(p_class, targets)
        expected = torch.log(torch.tensor(3.0))
        assert abs(loss.item() - expected.item()) < 0.1


class TestCombinedLoss:
    """Tests for combined_loss."""
    
    def test_no_class_weight(self):
        B, K = 4, 6
        posterior = PosteriorResult(
            p_generator=torch.softmax(torch.randn(B, K), dim=-1),
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_generator=torch.rand(B),
            entropy_class=torch.rand(B),
            logits_generator=torch.randn(B, K),
        )
        gen_targets = torch.randint(0, K, (B,))
        cls_targets = torch.randint(0, 3, (B,))
        
        loss, metrics = combined_loss(posterior, gen_targets, cls_targets, class_weight=0.0)
        
        assert "loss_generator" in metrics
        assert "loss_class" not in metrics
        assert abs(loss.item() - metrics["loss_generator"]) < 1e-5
    
    def test_with_class_weight(self):
        B, K = 4, 6
        posterior = PosteriorResult(
            p_generator=torch.softmax(torch.randn(B, K), dim=-1),
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_generator=torch.rand(B),
            entropy_class=torch.rand(B),
            logits_generator=torch.randn(B, K),
        )
        gen_targets = torch.randint(0, K, (B,))
        cls_targets = torch.randint(0, 3, (B,))
        
        loss, metrics = combined_loss(posterior, gen_targets, cls_targets, class_weight=0.5)
        
        assert "loss_generator" in metrics
        assert "loss_class" in metrics
        expected = metrics["loss_generator"] + 0.5 * metrics["loss_class"]
        assert abs(loss.item() - expected) < 1e-5


class TestComputeAccuracy:
    """Tests for compute_accuracy."""
    
    def test_perfect_accuracy(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        
        acc = compute_accuracy(logits, targets)
        assert acc == 1.0
    
    def test_zero_accuracy(self):
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        targets = torch.tensor([1, 2, 0])  # All wrong
        
        acc = compute_accuracy(logits, targets)
        assert acc == 0.0
    
    def test_partial_accuracy(self):
        logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
        targets = torch.tensor([0, 0, 0, 1])  # 2 correct, 2 wrong
        
        acc = compute_accuracy(logits, targets)
        assert acc == 0.5


class TestTopkAccuracy:
    """Tests for compute_topk_accuracy."""
    
    def test_top1_equals_accuracy(self):
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        top1 = compute_topk_accuracy(logits, targets, k=1)
        acc = compute_accuracy(logits, targets)
        
        assert abs(top1 - acc) < 1e-6
    
    def test_topk_geq_top1(self):
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        
        top1 = compute_topk_accuracy(logits, targets, k=1)
        top3 = compute_topk_accuracy(logits, targets, k=3)
        top5 = compute_topk_accuracy(logits, targets, k=5)
        
        assert top3 >= top1
        assert top5 >= top3
    
    def test_topk_equals_k(self):
        # If k >= K, accuracy should be 1.0
        logits = torch.randn(10, 5)
        targets = torch.randint(0, 5, (10,))
        
        top10 = compute_topk_accuracy(logits, targets, k=10)
        assert top10 == 1.0
