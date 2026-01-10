"""
lacuna.generators.families.mnar

MNAR (Missing Not At Random) generators with variant support.

MNAR Variants:
    - MNAR_SELF_CENSORING: Missingness depends on the value itself
      (e.g., high income more likely to be missing)
    
    - MNAR_THRESHOLD: Values beyond a threshold are systematically missing
      (e.g., lab values below detection limit)
    
    - MNAR_LATENT: Missingness driven by an unobserved latent factor
      (e.g., "health status" affects both values and missingness)

Each variant produces distinct patterns in the data that the reconstruction
heads are designed to recognize.
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from lacuna.core.types import ObservedDataset, MNAR
from lacuna.core.rng import RNGState
from lacuna.generators.base import BaseGenerator, GeneratorParams


# =============================================================================
# MNAR Variant IDs
# =============================================================================

MNAR_SELF_CENSORING = 0
MNAR_THRESHOLD = 1
MNAR_LATENT = 2

MNAR_VARIANT_NAMES = {
    MNAR_SELF_CENSORING: "self_censoring",
    MNAR_THRESHOLD: "threshold",
    MNAR_LATENT: "latent",
}


# =============================================================================
# MNAR Self-Censoring Generator
# =============================================================================

@dataclass
class MNARSelfCensoringParams(GeneratorParams):
    """Parameters for self-censoring MNAR mechanism.
    
    Attributes:
        base_missing_rate: Baseline probability of missingness.
        censoring_direction: "high" (high values more likely missing) or
                            "low" (low values more likely missing).
        censoring_strength: How strongly value magnitude affects missingness.
                           0 = no effect, 1 = strong effect.
        affected_columns: Fraction of columns affected by self-censoring.
    """
    base_missing_rate: float = 0.2
    censoring_direction: str = "high"
    censoring_strength: float = 0.5
    affected_columns: float = 0.5


class MNARSelfCensoringGenerator(BaseGenerator):
    """
    Generator for self-censoring MNAR mechanism.
    
    High (or low) values are more likely to be missing. This pattern
    occurs in practice when:
    - People with high income don't report it
    - Extreme health measures cause patient dropout
    - Outliers are flagged and removed
    
    The missingness probability for value x is:
        p(missing | x) = base_rate + strength * sigmoid(direction * zscore(x))
    
    Attributes:
        params: MNARSelfCensoringParams configuration.
        variant_id: MNAR_SELF_CENSORING (0).
    """
    
    def __init__(
        self,
        generator_id: int,
        params: Optional[MNARSelfCensoringParams] = None,
    ):
        super().__init__(
            generator_id=generator_id,
            class_id=MNAR,
            name="mnar_self_censoring",
        )
        self.params = params or MNARSelfCensoringParams()
        self.variant_id = MNAR_SELF_CENSORING
    
    def sample_observed(
        self,
        rng: RNGState,
        n: int,
        d: int,
        dataset_id: Optional[str] = None,
    ) -> ObservedDataset:
        """Generate dataset with self-censoring missingness.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows (samples).
            d: Number of columns (features).
            dataset_id: Optional identifier for the dataset.
        
        Returns:
            ObservedDataset with self-censoring MNAR pattern.
        """
        np_rng = rng.numpy_rng
        
        # Generate complete data (multivariate normal)
        X = np_rng.standard_normal((n, d))
        
        # Initialize missingness mask (all observed)
        R = np.ones((n, d), dtype=bool)
        
        # Determine which columns are affected
        n_affected = max(1, int(d * self.params.affected_columns))
        affected_cols = np_rng.choice(d, n_affected, replace=False)
        
        # Apply self-censoring to affected columns
        for j in affected_cols:
            # Compute z-scores for this column
            col_mean = X[:, j].mean()
            col_std = X[:, j].std() + 1e-8
            z_scores = (X[:, j] - col_mean) / col_std
            
            # Direction: high values missing or low values missing
            if self.params.censoring_direction == "high":
                effect = z_scores
            else:
                effect = -z_scores
            
            # Compute missingness probability
            # Sigmoid to map to [0, 1], then scale
            sigmoid = 1 / (1 + np.exp(-effect))
            p_missing = (
                self.params.base_missing_rate
                + self.params.censoring_strength * (sigmoid - 0.5)
            )
            p_missing = np.clip(p_missing, 0.01, 0.99)
            
            # Sample missingness
            missing = np_rng.random(n) < p_missing
            R[:, j] = ~missing
        
        # Create observed data (NaN for missing)
        X_obs = X.copy()
        X_obs[~R] = np.nan
        
        return ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id=dataset_id or f"mnar_sc_{self.generator_id}",
            n_original=n,
            d_original=d,
        )


# =============================================================================
# MNAR Threshold Generator
# =============================================================================

@dataclass
class MNARThresholdParams(GeneratorParams):
    """Parameters for threshold MNAR mechanism.
    
    Attributes:
        threshold_type: "lower" (below threshold missing) or
                       "upper" (above threshold missing) or
                       "both" (outside range missing).
        threshold_percentile: Percentile at which to set threshold.
        affected_columns: Fraction of columns affected.
        noise: Random noise in threshold application.
    """
    threshold_type: str = "lower"
    threshold_percentile: float = 0.1
    affected_columns: float = 0.3
    noise: float = 0.1


class MNARThresholdGenerator(BaseGenerator):
    """
    Generator for threshold MNAR mechanism.
    
    Values beyond a threshold are systematically missing. This pattern
    occurs in practice when:
    - Lab values below detection limit
    - Sensor readings outside calibrated range
    - Income above reporting threshold
    
    Attributes:
        params: MNARThresholdParams configuration.
        variant_id: MNAR_THRESHOLD (1).
    """
    
    def __init__(
        self,
        generator_id: int,
        params: Optional[MNARThresholdParams] = None,
    ):
        super().__init__(
            generator_id=generator_id,
            class_id=MNAR,
            name="mnar_threshold",
        )
        self.params = params or MNARThresholdParams()
        self.variant_id = MNAR_THRESHOLD
    
    def sample_observed(
        self,
        rng: RNGState,
        n: int,
        d: int,
        dataset_id: Optional[str] = None,
    ) -> ObservedDataset:
        """Generate dataset with threshold missingness.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows (samples).
            d: Number of columns (features).
            dataset_id: Optional identifier for the dataset.
        
        Returns:
            ObservedDataset with threshold MNAR pattern.
        """
        np_rng = rng.numpy_rng
        
        # Generate complete data
        X = np_rng.standard_normal((n, d))
        
        # Initialize missingness mask
        R = np.ones((n, d), dtype=bool)
        
        # Determine affected columns
        n_affected = max(1, int(d * self.params.affected_columns))
        affected_cols = np_rng.choice(d, n_affected, replace=False)
        
        # Apply threshold to affected columns
        for j in affected_cols:
            col = X[:, j]
            
            if self.params.threshold_type == "lower":
                # Values below threshold are missing
                threshold = np.percentile(col, self.params.threshold_percentile * 100)
                # Add noise to threshold decision
                noise = np_rng.standard_normal(n) * self.params.noise
                missing = (col + noise) < threshold
                
            elif self.params.threshold_type == "upper":
                # Values above threshold are missing
                threshold = np.percentile(col, (1 - self.params.threshold_percentile) * 100)
                noise = np_rng.standard_normal(n) * self.params.noise
                missing = (col + noise) > threshold
                
            else:  # both
                # Values outside range are missing
                lower = np.percentile(col, self.params.threshold_percentile * 100 / 2)
                upper = np.percentile(col, (1 - self.params.threshold_percentile / 2) * 100)
                noise = np_rng.standard_normal(n) * self.params.noise
                col_noisy = col + noise
                missing = (col_noisy < lower) | (col_noisy > upper)
            
            R[:, j] = ~missing
        
        # Create observed data
        X_obs = X.copy()
        X_obs[~R] = np.nan
        
        return ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id=dataset_id or f"mnar_th_{self.generator_id}",
            n_original=n,
            d_original=d,
        )


# =============================================================================
# MNAR Latent Generator
# =============================================================================

@dataclass
class MNARLatentParams(GeneratorParams):
    """Parameters for latent MNAR mechanism.
    
    Attributes:
        latent_dim: Dimension of latent factor.
        latent_effect_on_values: How strongly latent affects values.
        latent_effect_on_missing: How strongly latent affects missingness.
        base_missing_rate: Baseline missingness rate.
        affected_columns: Fraction of columns affected by latent.
    """
    latent_dim: int = 1
    latent_effect_on_values: float = 0.5
    latent_effect_on_missing: float = 0.5
    base_missing_rate: float = 0.2
    affected_columns: float = 0.5


class MNARLatentGenerator(BaseGenerator):
    """
    Generator for latent-driven MNAR mechanism.
    
    An unobserved latent factor affects both the values and the
    missingness. This creates a confounding structure where the
    missingness depends on unobserved information.
    
    Example scenarios:
    - "Health status" affects both lab values and whether patient
      returns for follow-up
    - "Engagement level" affects both survey responses and completion
    - "Socioeconomic status" affects both income and reporting
    
    Attributes:
        params: MNARLatentParams configuration.
        variant_id: MNAR_LATENT (2).
    """
    
    def __init__(
        self,
        generator_id: int,
        params: Optional[MNARLatentParams] = None,
    ):
        super().__init__(
            generator_id=generator_id,
            class_id=MNAR,
            name="mnar_latent",
        )
        self.params = params or MNARLatentParams()
        self.variant_id = MNAR_LATENT
    
    def sample_observed(
        self,
        rng: RNGState,
        n: int,
        d: int,
        dataset_id: Optional[str] = None,
    ) -> ObservedDataset:
        """Generate dataset with latent-driven missingness.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows (samples).
            d: Number of columns (features).
            dataset_id: Optional identifier for the dataset.
        
        Returns:
            ObservedDataset with latent MNAR pattern.
        """
        np_rng = rng.numpy_rng
        
        # Generate latent factor(s) for each sample
        # Shape: [n, latent_dim]
        Z = np_rng.standard_normal((n, self.params.latent_dim))
        
        # Generate loadings for how latent affects values
        # Shape: [latent_dim, d]
        value_loadings = np_rng.standard_normal((self.params.latent_dim, d))
        value_loadings *= self.params.latent_effect_on_values
        
        # Generate base values (independent of latent)
        # Shape: [n, d]
        X_base = np_rng.standard_normal((n, d))
        
        # Add latent effect to values
        # X = X_base + Z @ value_loadings
        # Shape: [n, d]
        X = X_base + Z @ value_loadings
        
        # Initialize missingness mask (all observed)
        R = np.ones((n, d), dtype=bool)
        
        # Determine which columns are affected by latent missingness
        n_affected = max(1, int(d * self.params.affected_columns))
        affected_cols = np_rng.choice(d, n_affected, replace=False)
        
        # Generate loadings for how latent affects missingness
        # Shape: [latent_dim, n_affected]
        missing_loadings = np_rng.standard_normal((self.params.latent_dim, n_affected))
        missing_loadings *= self.params.latent_effect_on_missing
        
        # Compute latent effect on missingness for each affected column
        # Shape: [n, n_affected]
        latent_effect = Z @ missing_loadings
        
        # Apply latent-driven missingness to affected columns
        for idx, j in enumerate(affected_cols):
            # Probability of missing based on latent
            # Use tanh to map latent effect to [-1, 1], then scale
            logit = latent_effect[:, idx]
            p_missing = self.params.base_missing_rate + 0.3 * np.tanh(logit)
            p_missing = np.clip(p_missing, 0.01, 0.99)
            
            # Sample missingness
            missing = np_rng.random(n) < p_missing
            R[:, j] = ~missing
        
        # Create observed data (NaN for missing)
        X_obs = X.copy()
        X_obs[~R] = np.nan
        
        return ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id=dataset_id or f"mnar_lat_{self.generator_id}",
            n_original=n,
            d_original=d,
        )


# =============================================================================
# MNAR Mixture Generator
# =============================================================================

@dataclass
class MNARMixtureParams(GeneratorParams):
    """Parameters for mixture MNAR mechanism.
    
    Combines multiple MNAR patterns within a single dataset,
    simulating heterogeneous missingness across subpopulations.
    
    Attributes:
        n_components: Number of mixture components (subpopulations).
        component_weights: Mixing weights (default: uniform).
        variant_types: Which MNAR variants to mix ("self_censoring", "threshold", "latent").
    """
    n_components: int = 2
    component_weights: Optional[List[float]] = None
    variant_types: Optional[List[str]] = None


class MNARMixtureGenerator(BaseGenerator):
    """
    Generator for mixture MNAR mechanism.
    
    Different rows (samples) follow different MNAR patterns,
    simulating heterogeneous missingness in real data. This is
    useful for testing whether the model can detect mechanism
    heterogeneity.
    
    Example scenarios:
    - Different hospitals have different data collection practices
    - Different patient subgroups have different dropout patterns
    - Time-varying missingness mechanisms
    
    Attributes:
        params: MNARMixtureParams configuration.
        variant_id: Uses the ID of the primary variant.
        sub_generators: List of MNAR generators to mix.
    """
    
    def __init__(
        self,
        generator_id: int,
        params: Optional[MNARMixtureParams] = None,
    ):
        super().__init__(
            generator_id=generator_id,
            class_id=MNAR,
            name="mnar_mixture",
        )
        self.params = params or MNARMixtureParams()
        
        # Default variant types if not specified
        if self.params.variant_types is None:
            self.params.variant_types = ["self_censoring", "threshold"]
        
        # Default uniform weights if not specified
        if self.params.component_weights is None:
            n = len(self.params.variant_types)
            self.params.component_weights = [1.0 / n] * n
        
        # Use first variant's ID as primary
        self.variant_id = MNAR_SELF_CENSORING
        
        # Create sub-generators for each variant
        self.sub_generators = []
        for i, variant_type in enumerate(self.params.variant_types):
            if variant_type == "self_censoring":
                sub_gen = MNARSelfCensoringGenerator(
                    generator_id=generator_id * 100 + i,
                    params=MNARSelfCensoringParams(),
                )
            elif variant_type == "threshold":
                sub_gen = MNARThresholdGenerator(
                    generator_id=generator_id * 100 + i,
                    params=MNARThresholdParams(),
                )
            elif variant_type == "latent":
                sub_gen = MNARLatentGenerator(
                    generator_id=generator_id * 100 + i,
                    params=MNARLatentParams(),
                )
            else:
                raise ValueError(f"Unknown MNAR variant type: {variant_type}")
            
            self.sub_generators.append(sub_gen)
    
    def sample_observed(
        self,
        rng: RNGState,
        n: int,
        d: int,
        dataset_id: Optional[str] = None,
    ) -> ObservedDataset:
        """Generate dataset with mixture of MNAR patterns.
        
        Args:
            rng: RNG state for reproducibility.
            n: Number of rows (samples).
            d: Number of columns (features).
            dataset_id: Optional identifier for the dataset.
        
        Returns:
            ObservedDataset with mixed MNAR patterns.
        """
        np_rng = rng.numpy_rng
        
        # Assign each row to a component
        weights = np.array(self.params.component_weights)
        weights = weights / weights.sum()  # Normalize
        component_assignments = np_rng.choice(
            len(self.sub_generators),
            size=n,
            p=weights,
        )
        
        # Generate complete data (shared across components for consistency)
        X = np_rng.standard_normal((n, d))
        
        # Initialize missingness mask
        R = np.ones((n, d), dtype=bool)
        
        # Apply each component's missingness pattern to its assigned rows
        for comp_idx, sub_gen in enumerate(self.sub_generators):
            # Get rows assigned to this component
            row_mask = component_assignments == comp_idx
            n_comp = row_mask.sum()
            
            if n_comp == 0:
                continue
            
            # Generate missingness pattern for this component
            sub_dataset = sub_gen.sample_observed(
                rng=rng.spawn(),
                n=n_comp,
                d=d,
                dataset_id=f"{dataset_id}_comp{comp_idx}",
            )
            
            # Extract just the missingness pattern (not the values)
            # and apply it to the corresponding rows
            R[row_mask] = sub_dataset.R
        
        # Create observed data
        X_obs = X.copy()
        X_obs[~R] = np.nan
        
        return ObservedDataset(
            X_obs=X_obs,
            R=R,
            dataset_id=dataset_id or f"mnar_mix_{self.generator_id}",
            n_original=n,
            d_original=d,
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_mnar_self_censoring(
    generator_id: int,
    base_missing_rate: float = 0.2,
    censoring_direction: str = "high",
    censoring_strength: float = 0.5,
    affected_columns: float = 0.5,
) -> MNARSelfCensoringGenerator:
    """
    Factory function to create self-censoring MNAR generator.
    
    Args:
        generator_id: Unique identifier for this generator.
        base_missing_rate: Baseline probability of missingness.
        censoring_direction: "high" or "low".
        censoring_strength: Strength of value-dependent censoring.
        affected_columns: Fraction of columns affected.
    
    Returns:
        Configured MNARSelfCensoringGenerator.
    """
    params = MNARSelfCensoringParams(
        base_missing_rate=base_missing_rate,
        censoring_direction=censoring_direction,
        censoring_strength=censoring_strength,
        affected_columns=affected_columns,
    )
    return MNARSelfCensoringGenerator(generator_id=generator_id, params=params)


def create_mnar_threshold(
    generator_id: int,
    threshold_type: str = "lower",
    threshold_percentile: float = 0.1,
    affected_columns: float = 0.3,
    noise: float = 0.1,
) -> MNARThresholdGenerator:
    """
    Factory function to create threshold MNAR generator.
    
    Args:
        generator_id: Unique identifier for this generator.
        threshold_type: "lower", "upper", or "both".
        threshold_percentile: Percentile for threshold placement.
        affected_columns: Fraction of columns affected.
        noise: Noise in threshold application.
    
    Returns:
        Configured MNARThresholdGenerator.
    """
    params = MNARThresholdParams(
        threshold_type=threshold_type,
        threshold_percentile=threshold_percentile,
        affected_columns=affected_columns,
        noise=noise,
    )
    return MNARThresholdGenerator(generator_id=generator_id, params=params)


def create_mnar_latent(
    generator_id: int,
    latent_dim: int = 1,
    latent_effect_on_values: float = 0.5,
    latent_effect_on_missing: float = 0.5,
    base_missing_rate: float = 0.2,
    affected_columns: float = 0.5,
) -> MNARLatentGenerator:
    """
    Factory function to create latent MNAR generator.
    
    Args:
        generator_id: Unique identifier for this generator.
        latent_dim: Dimension of latent factor.
        latent_effect_on_values: Strength of latent effect on values.
        latent_effect_on_missing: Strength of latent effect on missingness.
        base_missing_rate: Baseline missingness rate.
        affected_columns: Fraction of columns affected.
    
    Returns:
        Configured MNARLatentGenerator.
    """
    params = MNARLatentParams(
        latent_dim=latent_dim,
        latent_effect_on_values=latent_effect_on_values,
        latent_effect_on_missing=latent_effect_on_missing,
        base_missing_rate=base_missing_rate,
        affected_columns=affected_columns,
    )
    return MNARLatentGenerator(generator_id=generator_id, params=params)


def create_mnar_mixture(
    generator_id: int,
    variant_types: Optional[List[str]] = None,
    component_weights: Optional[List[float]] = None,
) -> MNARMixtureGenerator:
    """
    Factory function to create mixture MNAR generator.
    
    Args:
        generator_id: Unique identifier for this generator.
        variant_types: List of variant types to mix.
        component_weights: Mixing weights for each variant.
    
    Returns:
        Configured MNARMixtureGenerator.
    """
    params = MNARMixtureParams(
        variant_types=variant_types,
        component_weights=component_weights,
    )
    return MNARMixtureGenerator(generator_id=generator_id, params=params)


def create_all_mnar_generators(
    start_id: int = 4,
) -> List[BaseGenerator]:
    """
    Create one generator for each MNAR variant.
    
    Useful for building a complete generator registry with
    coverage of all MNAR patterns.
    
    Args:
        start_id: Starting generator ID (MCAR/MAR typically use 0-3).
    
    Returns:
        List of MNAR generators [self_censoring, threshold, latent].
    """
    return [
        create_mnar_self_censoring(generator_id=start_id),
        create_mnar_threshold(generator_id=start_id + 1),
        create_mnar_latent(generator_id=start_id + 2),
    ]