"""
Tests for lacuna.core.types
"""

import pytest
import torch
from lacuna.core.types import (
    ObservedDataset,
    TokenBatch,
    PosteriorResult,
    Decision,
    MCAR, MAR, MNAR,
    CLASS_NAMES,
)


class TestObservedDataset:
    """Tests for ObservedDataset."""
    
    def test_valid_construction(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        
        ds = ObservedDataset(x=x, r=r, n=100, d=5)
        
        assert ds.n == 100
        assert ds.d == 5
        assert ds.missing_rate == 0.0
        assert ds.n_observed == 500
    
    def test_missing_rate(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        r[:20, :] = False  # 20% missing
        
        ds = ObservedDataset(x=x, r=r, n=100, d=5)
        
        assert abs(ds.missing_rate - 0.2) < 0.01
    
    def test_wrong_x_shape_raises(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5, dtype=torch.bool)
        
        with pytest.raises(ValueError):
            ObservedDataset(x=x, r=r, n=50, d=5)  # Wrong n
    
    def test_wrong_r_shape_raises(self):
        x = torch.randn(100, 5)
        r = torch.ones(50, 5, dtype=torch.bool)  # Wrong shape
        
        with pytest.raises(ValueError):
            ObservedDataset(x=x, r=r, n=100, d=5)
    
    def test_wrong_r_dtype_raises(self):
        x = torch.randn(100, 5)
        r = torch.ones(100, 5)  # Float, not bool
        
        with pytest.raises(TypeError):
            ObservedDataset(x=x, r=r, n=100, d=5)
    
    def test_with_feature_names(self):
        x = torch.randn(100, 3)
        r = torch.ones(100, 3, dtype=torch.bool)
        
        ds = ObservedDataset(
            x=x, r=r, n=100, d=3,
            feature_names=("a", "b", "c"),
        )
        
        assert ds.feature_names == ("a", "b", "c")
    
    def test_with_metadata(self):
        x = torch.randn(100, 3)
        r = torch.ones(100, 3, dtype=torch.bool)
        
        ds = ObservedDataset(
            x=x, r=r, n=100, d=3,
            meta={"source": "test"},
        )
        
        assert ds.meta["source"] == "test"


class TestTokenBatch:
    """Tests for TokenBatch."""
    
    def test_valid_construction(self):
        B, max_rows, max_cols, token_dim = 4, 64, 16, 2
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
        )
        
        assert batch.batch_size == 4
        assert batch.max_rows == 64
        assert batch.max_cols == 16
        assert batch.token_dim == 2
    
    def test_with_labels(self):
        B, max_rows, max_cols, token_dim = 4, 64, 16, 2
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0, 1, 2, 3]),
            class_ids=torch.tensor([0, 0, 1, 1]),
        )
        
        assert batch.generator_ids.shape == (4,)
        assert batch.class_ids.shape == (4,)
    
    def test_wrong_row_mask_shape_raises(self):
        B, max_rows, max_cols, token_dim = 4, 64, 16, 2
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, 32, dtype=torch.bool),  # Wrong
                col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            )
    
    def test_wrong_col_mask_shape_raises(self):
        B, max_rows, max_cols, token_dim = 4, 64, 16, 2
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, max_rows, dtype=torch.bool),
                col_mask=torch.ones(B, 8, dtype=torch.bool),  # Wrong
            )
    
    def test_wrong_generator_ids_shape_raises(self):
        B, max_rows, max_cols, token_dim = 4, 64, 16, 2
        
        with pytest.raises(ValueError):
            TokenBatch(
                tokens=torch.randn(B, max_rows, max_cols, token_dim),
                row_mask=torch.ones(B, max_rows, dtype=torch.bool),
                col_mask=torch.ones(B, max_cols, dtype=torch.bool),
                generator_ids=torch.tensor([0, 1]),  # Wrong size
            )
    
    def test_to_device(self):
        B, max_rows, max_cols, token_dim = 2, 32, 8, 2
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_rows, max_cols, token_dim),
            row_mask=torch.ones(B, max_rows, dtype=torch.bool),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            generator_ids=torch.tensor([0, 1]),
        )
        
        batch_cpu = batch.to("cpu")
        assert batch_cpu.tokens.device.type == "cpu"


class TestPosteriorResult:
    """Tests for PosteriorResult."""
    
    def test_valid_construction(self):
        B, K = 4, 6
        
        result = PosteriorResult(
            p_generator=torch.softmax(torch.randn(B, K), dim=-1),
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_generator=torch.rand(B),
            entropy_class=torch.rand(B),
            logits_generator=torch.randn(B, K),
        )
        
        assert result.p_generator.shape == (B, K)
        assert result.p_class.shape == (B, 3)


class TestDecision:
    """Tests for Decision."""
    
    def test_valid_construction(self):
        decision = Decision(
            action_ids=torch.tensor([0, 1, 2, 0]),
            expected_risks=torch.rand(4),
        )
        
        assert decision.batch_size == 4
        assert decision.action_names == ("Green", "Yellow", "Red")
    
    def test_get_actions(self):
        decision = Decision(
            action_ids=torch.tensor([0, 1, 2]),
            expected_risks=torch.rand(3),
        )
        
        actions = decision.get_actions()
        assert actions == ["Green", "Yellow", "Red"]


class TestClassConstants:
    """Tests for class constants."""
    
    def test_class_ids(self):
        assert MCAR == 0
        assert MAR == 1
        assert MNAR == 2
    
    def test_class_names(self):
        assert CLASS_NAMES == ("MCAR", "MAR", "MNAR")
