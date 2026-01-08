"""
Tests for lacuna.core.types

Verify data types validate correctly and catch errors.
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
        r[0, 0] = False  # One missing value
        
        ds = ObservedDataset(
            x=x,
            r=r,
            n=100,
            d=5,
            feature_names=("a", "b", "c", "d", "e"),
            dataset_id="test_001",
        )
        
        assert ds.n == 100
        assert ds.d == 5
        assert ds.n_observed == 499
        assert ds.n_missing == 1
    
    def test_wrong_x_shape_raises(self):
        with pytest.raises(ValueError, match="x shape"):
            ObservedDataset(
                x=torch.randn(50, 5),  # Wrong n
                r=torch.ones(100, 5, dtype=torch.bool),
                n=100,
                d=5,
                feature_names=("a", "b", "c", "d", "e"),
                dataset_id="test",
            )
    
    def test_wrong_r_shape_raises(self):
        with pytest.raises(ValueError, match="r shape"):
            ObservedDataset(
                x=torch.randn(100, 5),
                r=torch.ones(100, 3, dtype=torch.bool),  # Wrong d
                n=100,
                d=5,
                feature_names=("a", "b", "c", "d", "e"),
                dataset_id="test",
            )
    
    def test_wrong_r_dtype_raises(self):
        with pytest.raises(TypeError, match="r.dtype must be bool"):
            ObservedDataset(
                x=torch.randn(100, 5),
                r=torch.ones(100, 5),  # float, not bool
                n=100,
                d=5,
                feature_names=("a", "b", "c", "d", "e"),
                dataset_id="test",
            )
    
    def test_wrong_feature_names_count_raises(self):
        with pytest.raises(ValueError, match=r"len\(feature_names\)"):
            ObservedDataset(
                x=torch.randn(100, 5),
                r=torch.ones(100, 5, dtype=torch.bool),
                n=100,
                d=5,
                feature_names=("a", "b", "c"),  # Only 3
                dataset_id="test",
            )
    
    def test_duplicate_feature_names_raises(self):
        with pytest.raises(ValueError, match="must be unique"):
            ObservedDataset(
                x=torch.randn(100, 5),
                r=torch.ones(100, 5, dtype=torch.bool),
                n=100,
                d=5,
                feature_names=("a", "a", "c", "d", "e"),  # Duplicate
                dataset_id="test",
            )
    
    def test_all_missing_raises(self):
        with pytest.raises(ValueError, match="at least one observed"):
            ObservedDataset(
                x=torch.randn(100, 5),
                r=torch.zeros(100, 5, dtype=torch.bool),  # All missing
                n=100,
                d=5,
                feature_names=("a", "b", "c", "d", "e"),
                dataset_id="test",
            )
    
    def test_missing_rate_property(self):
        x = torch.randn(100, 10)
        r = torch.ones(100, 10, dtype=torch.bool)
        r[:, 0] = False  # 10% missing
        
        ds = ObservedDataset(
            x=x, r=r, n=100, d=10,
            feature_names=tuple(f"f{i}" for i in range(10)),
            dataset_id="test",
        )
        
        assert abs(ds.missing_rate - 0.1) < 1e-6
    
    def test_frozen(self):
        ds = ObservedDataset(
            x=torch.randn(10, 3),
            r=torch.ones(10, 3, dtype=torch.bool),
            n=10,
            d=3,
            feature_names=("a", "b", "c"),
            dataset_id="test",
        )
        
        with pytest.raises(AttributeError):
            ds.n = 20


class TestTokenBatch:
    """Tests for TokenBatch."""
    
    def test_valid_construction(self):
        B, max_cols, q = 8, 16, 32
        
        batch = TokenBatch(
            tokens=torch.randn(B, max_cols, q),
            col_mask=torch.ones(B, max_cols, dtype=torch.bool),
            generator_ids=torch.randint(0, 6, (B,)),
            class_ids=torch.randint(0, 3, (B,)),
        )
        
        assert batch.batch_size == B
        assert batch.max_cols == max_cols
        assert batch.token_dim == q
    
    def test_wrong_col_mask_shape_raises(self):
        with pytest.raises(ValueError, match="col_mask shape"):
            TokenBatch(
                tokens=torch.randn(8, 16, 32),
                col_mask=torch.ones(8, 10, dtype=torch.bool),  # Wrong
            )
    
    def test_wrong_generator_ids_shape_raises(self):
        with pytest.raises(ValueError, match="generator_ids shape"):
            TokenBatch(
                tokens=torch.randn(8, 16, 32),
                col_mask=torch.ones(8, 16, dtype=torch.bool),
                generator_ids=torch.randint(0, 6, (10,)),  # Wrong
            )
    
    def test_to_device(self):
        batch = TokenBatch(
            tokens=torch.randn(4, 8, 16),
            col_mask=torch.ones(4, 8, dtype=torch.bool),
            generator_ids=torch.tensor([0, 1, 2, 3]),
        )
        
        batch_cpu = batch.to("cpu")
        assert batch_cpu.tokens.device.type == "cpu"


class TestPosteriorResult:
    """Tests for PosteriorResult."""
    
    def test_valid_construction(self):
        B, K = 8, 6
        
        result = PosteriorResult(
            p_generator=torch.softmax(torch.randn(B, K), dim=-1),
            p_class=torch.softmax(torch.randn(B, 3), dim=-1),
            entropy_generator=torch.rand(B),
            entropy_class=torch.rand(B),
            logits_generator=torch.randn(B, K),
        )
        
        assert result.batch_size == B
        assert result.n_generators == K
    
    def test_wrong_p_class_dims_raises(self):
        with pytest.raises(ValueError, match="3 classes"):
            PosteriorResult(
                p_generator=torch.randn(8, 6),
                p_class=torch.randn(8, 2),  # Wrong
                entropy_generator=torch.rand(8),
                entropy_class=torch.rand(8),
                logits_generator=torch.randn(8, 6),
            )
    
    def test_confidence_property(self):
        result = PosteriorResult(
            p_generator=torch.randn(4, 6),
            p_class=torch.randn(4, 3),
            entropy_generator=torch.rand(4),
            entropy_class=torch.zeros(4),  # Zero entropy = max confidence
            logits_generator=torch.randn(4, 6),
        )
        
        conf = result.confidence
        assert torch.allclose(conf, torch.ones(4))


class TestDecision:
    """Tests for Decision."""
    
    def test_valid_construction(self):
        decision = Decision(
            action_ids=torch.tensor([0, 1, 2, 0]),
            action_names=("Green", "Yellow", "Red"),
            expected_risks=torch.rand(4),
        )
        
        assert decision.batch_size == 4
    
    def test_get_actions(self):
        decision = Decision(
            action_ids=torch.tensor([0, 2, 1]),
            action_names=("Green", "Yellow", "Red"),
            expected_risks=torch.rand(3),
        )
        
        actions = decision.get_actions()
        assert actions == ["Green", "Red", "Yellow"]
    
    def test_wrong_action_names_count_raises(self):
        with pytest.raises(ValueError, match="exactly 3"):
            Decision(
                action_ids=torch.tensor([0, 1]),
                action_names=("A", "B"),  # Only 2
                expected_risks=torch.rand(2),
            )


class TestClassConstants:
    """Tests for class ID constants."""
    
    def test_class_ids(self):
        assert MCAR == 0
        assert MAR == 1
        assert MNAR == 2
    
    def test_class_names(self):
        assert CLASS_NAMES == ("MCAR", "MAR", "MNAR")
        assert CLASS_NAMES[MCAR] == "MCAR"
        assert CLASS_NAMES[MAR] == "MAR"
        assert CLASS_NAMES[MNAR] == "MNAR"