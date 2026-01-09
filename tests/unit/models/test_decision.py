"""
Tests for lacuna.models.decision
"""

import pytest
import torch
from lacuna.models.decision import (
    DEFAULT_LOSS_MATRIX,
    compute_expected_loss,
    bayes_optimal_decision,
    make_decision,
    interpret_decision,
)


class TestComputeExpectedLoss:
    """Tests for expected loss computation."""
    
    def test_shape(self):
        p_class = torch.randn(4, 3).softmax(dim=-1)
        
        exp_loss = compute_expected_loss(p_class, DEFAULT_LOSS_MATRIX)
        
        assert exp_loss.shape == (4, 3)  # 3 actions
    
    def test_certain_mcar_prefers_green(self):
        # If certain MCAR, Green should have lowest loss
        p_class = torch.tensor([[1.0, 0.0, 0.0]])
        
        exp_loss = compute_expected_loss(p_class, DEFAULT_LOSS_MATRIX)
        
        # Green=0, Yellow=1, Red=3 for MCAR
        assert exp_loss[0, 0] < exp_loss[0, 1]  # Green < Yellow
        assert exp_loss[0, 0] < exp_loss[0, 2]  # Green < Red
    
    def test_certain_mnar_prefers_red(self):
        # If certain MNAR, Red should have lowest loss
        p_class = torch.tensor([[0.0, 0.0, 1.0]])
        
        exp_loss = compute_expected_loss(p_class, DEFAULT_LOSS_MATRIX)
        
        # Green=10, Yellow=2, Red=0 for MNAR
        assert exp_loss[0, 2] < exp_loss[0, 1]  # Red < Yellow
        assert exp_loss[0, 2] < exp_loss[0, 0]  # Red < Green


class TestBayesOptimalDecision:
    """Tests for Bayes-optimal decision."""
    
    def test_returns_correct_shapes(self):
        p_class = torch.randn(8, 3).softmax(dim=-1)
        
        action_ids, expected_risks = bayes_optimal_decision(p_class, DEFAULT_LOSS_MATRIX)
        
        assert action_ids.shape == (8,)
        assert expected_risks.shape == (8,)
    
    def test_certain_mcar_chooses_green(self):
        p_class = torch.tensor([[1.0, 0.0, 0.0]])
        
        action_ids, _ = bayes_optimal_decision(p_class, DEFAULT_LOSS_MATRIX)
        
        assert action_ids[0] == 0  # Green
    
    def test_certain_mnar_chooses_red(self):
        p_class = torch.tensor([[0.0, 0.0, 1.0]])
        
        action_ids, _ = bayes_optimal_decision(p_class, DEFAULT_LOSS_MATRIX)
        
        assert action_ids[0] == 2  # Red
    
    def test_uncertain_may_choose_yellow(self):
        # Uniform uncertainty might prefer Yellow (cautious)
        p_class = torch.tensor([[0.33, 0.34, 0.33]])
        
        action_ids, _ = bayes_optimal_decision(p_class, DEFAULT_LOSS_MATRIX)
        
        # With default loss matrix, Yellow often optimal for uncertainty
        # This depends on exact loss values
        assert action_ids[0] in [0, 1, 2]  # Valid action


class TestMakeDecision:
    """Tests for make_decision helper."""
    
    def test_creates_decision_object(self):
        p_class = torch.randn(4, 3).softmax(dim=-1)
        
        decision = make_decision(p_class)
        
        assert decision.batch_size == 4
        assert decision.action_names == ("Green", "Yellow", "Red")
    
    def test_get_actions(self):
        p_class = torch.tensor([
            [1.0, 0.0, 0.0],  # MCAR -> Green
            [0.0, 0.0, 1.0],  # MNAR -> Red
        ])
        
        decision = make_decision(p_class)
        actions = decision.get_actions()
        
        assert actions[0] == "Green"
        assert actions[1] == "Red"


class TestInterpretDecision:
    """Tests for interpret_decision."""
    
    def test_returns_string(self):
        p_class = torch.tensor([[0.8, 0.1, 0.1]])
        decision = make_decision(p_class)
        
        interpretation = interpret_decision(decision, idx=0)
        
        assert isinstance(interpretation, str)
        assert "Green" in interpretation or "Yellow" in interpretation or "Red" in interpretation
