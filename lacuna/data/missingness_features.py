"""
lacuna.data.missingness_features

Explicit missingness pattern features for mechanism classification.

The Problem:
    Reconstruction errors alone cannot distinguish MCAR from MAR effectively
    because cross-attention helps prediction under BOTH mechanisms (just less
    so under MCAR). The gap between MCAR and MAR reconstruction ratios is tiny.

The Solution:
    Add explicit statistical features that capture the STRUCTURE of missingness:
    
    1. MCAR Signature: Random scatter
       - Uniform missingness across columns (low variance in missing rates)
       - No correlation between missingness and observed values
       - Missingness indicators are independent across columns
    
    2. MAR Signature: Structured cross-column dependency
       - Missingness in column j correlates with observed values in column k
       - Predictable missingness patterns from observed data
       - Non-uniform missingness (some columns more affected)
    
    3. MNAR Signature: Self-dependency
       - Distributional distortions within columns (skewness, kurtosis shifts)
       - Missingness correlates with the (unobserved) value itself
       - Truncation/censoring patterns

Features Extracted:
    - Missing rate statistics (mean, variance, range across columns)
    - Point-biserial correlations (missingness vs observed values)
    - Cross-column missingness correlations
    - Little's MCAR test approximation
    - Distributional statistics of observed values (for MNAR detection)

Usage:
    >>> from lacuna.data.missingness_features import extract_missingness_features
    >>> features = extract_missingness_features(tokens, row_mask, col_mask)
    >>> # features: [B, n_features] tensor ready for MoE gating
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from lacuna.data.tokenization import IDX_VALUE, IDX_OBSERVED


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MissingnessFeatureConfig:
    """Configuration for missingness feature extraction."""
    
    # Which feature groups to include
    include_missing_rate_stats: bool = True      # Mean, var, range of missing rates
    include_pointbiserial: bool = True           # Correlation: missingness vs values
    include_cross_column_corr: bool = True       # Correlation: missingness across columns
    include_distributional: bool = True          # Skewness, kurtosis of observed values
    include_littles_approx: bool = True          # Approximate Little's test statistic
    
    # Numerical stability
    eps: float = 1e-8
    
    @property
    def n_features(self) -> int:
        """Total number of features extracted."""
        n = 0
        if self.include_missing_rate_stats:
            n += 4  # mean, var, range, max
        if self.include_pointbiserial:
            n += 3  # mean, max, std of correlations
        if self.include_cross_column_corr:
            n += 3  # mean, max of cross-column missingness correlation
        if self.include_distributional:
            n += 4  # mean skewness, mean kurtosis, skew variance, kurt variance
        if self.include_littles_approx:
            n += 2  # approximate chi-squared statistic, p-value proxy
        return n


# Default configuration
DEFAULT_CONFIG = MissingnessFeatureConfig()


# =============================================================================
# Core Feature Extraction Functions
# =============================================================================

def compute_missing_rate_stats(
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute statistics of missing rates across columns.
    
    MCAR Signature: Low variance in missing rates (uniform randomness)
    MAR/MNAR Signature: High variance (some columns more affected)
    
    Returns:
        features: [B, 4] tensor with [mean_rate, var_rate, range_rate, max_rate]
    """
    B, max_rows, max_cols = is_observed.shape
    device = is_observed.device
    
    # Expand masks for broadcasting
    # row_mask: [B, max_rows] -> [B, max_rows, 1]
    # col_mask: [B, max_cols] -> [B, 1, max_cols]
    row_mask_exp = row_mask.unsqueeze(-1).float()
    col_mask_exp = col_mask.unsqueeze(1).float()
    
    # Valid cell mask: [B, max_rows, max_cols]
    valid_mask = row_mask_exp * col_mask_exp
    
    # Compute missing rate per column
    # Sum of missing indicators per column / number of valid rows
    is_missing = 1.0 - is_observed.float()
    
    # Per-column: sum missing, sum valid rows
    missing_per_col = (is_missing * valid_mask).sum(dim=1)  # [B, max_cols]
    valid_rows_per_col = valid_mask.sum(dim=1)  # [B, max_cols]
    
    # Missing rate per column (avoid division by zero)
    missing_rate = missing_per_col / valid_rows_per_col.clamp(min=1)  # [B, max_cols]
    
    # Mask out invalid columns
    col_mask_float = col_mask.float()
    n_valid_cols = col_mask_float.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    
    # Compute statistics over valid columns only
    # Mean missing rate
    mean_rate = (missing_rate * col_mask_float).sum(dim=1) / n_valid_cols.squeeze(-1)
    
    # Variance of missing rates
    rate_diff = (missing_rate - mean_rate.unsqueeze(-1)) * col_mask_float
    var_rate = (rate_diff ** 2).sum(dim=1) / n_valid_cols.squeeze(-1).clamp(min=1)
    
    # Range of missing rates (max - min over valid columns)
    # Set invalid columns to extreme values for min/max computation
    rate_for_max = missing_rate.clone()
    rate_for_max[~col_mask] = -float('inf')
    max_rate = rate_for_max.max(dim=1).values
    
    rate_for_min = missing_rate.clone()
    rate_for_min[~col_mask] = float('inf')
    min_rate = rate_for_min.min(dim=1).values
    
    range_rate = max_rate - min_rate
    
    # Handle edge cases (single column or all same rate)
    range_rate = torch.where(
        torch.isinf(range_rate) | torch.isnan(range_rate),
        torch.zeros_like(range_rate),
        range_rate
    )
    max_rate = torch.where(
        torch.isinf(max_rate),
        mean_rate,
        max_rate
    )
    
    # Stack features: [B, 4]
    features = torch.stack([mean_rate, var_rate, range_rate, max_rate], dim=-1)
    
    # Clamp to reasonable range
    features = features.clamp(min=0.0, max=1.0)
    
    return features


def compute_pointbiserial_correlations(
    values: torch.Tensor,       # [B, max_rows, max_cols]
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute point-biserial correlations between missingness and observed values.
    
    For each column j, compute correlation between:
        - Missingness indicator in column j
        - Observed values in OTHER columns
    
    MAR Signature: High correlations (missingness depends on observed values)
    MCAR Signature: Low correlations (missingness is random)
    MNAR Signature: Variable (depends on specific mechanism)
    
    Returns:
        features: [B, 3] tensor with [mean_corr, max_corr, std_corr]
    """
    B, max_rows, max_cols = values.shape
    device = values.device
    
    # Valid cell mask
    row_mask_exp = row_mask.unsqueeze(-1).float()
    col_mask_exp = col_mask.unsqueeze(1).float()
    valid_mask = row_mask_exp * col_mask_exp  # [B, max_rows, max_cols]
    
    # Missingness indicator (1 = missing, 0 = observed)
    is_missing = (1.0 - is_observed.float()) * valid_mask
    
    # Observed values (masked)
    obs_values = values * is_observed.float() * valid_mask
    
    # For each pair of columns (j, k), compute correlation between:
    # - missingness in column j
    # - observed values in column k (where k != j and value is observed)
    
    # This is O(d^2) which could be expensive, so we approximate:
    # Compute correlation between missingness in each column and the
    # MEAN of observed values across all other columns in that row
    
    # Step 1: Compute row-wise mean of observed values (excluding each column)
    # Total observed sum per row
    row_obs_sum = (obs_values * is_observed.float()).sum(dim=2)  # [B, max_rows]
    row_obs_count = (is_observed.float() * valid_mask).sum(dim=2)  # [B, max_rows]
    
    # For column j: mean of OTHER columns = (total - col_j) / (count - 1)
    # This is an approximation; we'll use total mean as proxy
    row_mean = row_obs_sum / row_obs_count.clamp(min=1)  # [B, max_rows]
    
    # Step 2: For each column, compute correlation between missingness and row_mean
    # Expand row_mean: [B, max_rows] -> [B, max_rows, 1] -> broadcast to [B, max_rows, max_cols]
    row_mean_exp = row_mean.unsqueeze(-1).expand_as(values)
    
    # Compute per-column correlation
    # corr(is_missing[:, j], row_mean) for each column j
    
    # Center the variables
    miss_mean = is_missing.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # [B, max_cols]
    miss_centered = is_missing - miss_mean.unsqueeze(1) * valid_mask
    
    rowmean_mean = (row_mean_exp * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    rowmean_centered = (row_mean_exp - rowmean_mean.unsqueeze(1)) * valid_mask
    
    # Covariance: sum of products
    cov = (miss_centered * rowmean_centered).sum(dim=1)  # [B, max_cols]
    
    # Standard deviations
    miss_std = (miss_centered ** 2).sum(dim=1).sqrt().clamp(min=eps)  # [B, max_cols]
    rowmean_std = (rowmean_centered ** 2).sum(dim=1).sqrt().clamp(min=eps)  # [B, max_cols]
    
    # Correlation
    corr = cov / (miss_std * rowmean_std)  # [B, max_cols]
    
    # Mask invalid columns
    corr = corr * col_mask.float()
    corr = torch.where(torch.isnan(corr) | torch.isinf(corr), torch.zeros_like(corr), corr)
    
    # Aggregate statistics over columns
    n_valid = col_mask.float().sum(dim=1).clamp(min=1)
    
    # Mean absolute correlation
    mean_corr = corr.abs().sum(dim=1) / n_valid
    
    # Max absolute correlation
    corr_for_max = corr.abs().clone()
    corr_for_max[~col_mask] = -float('inf')
    max_corr = corr_for_max.max(dim=1).values
    max_corr = torch.where(torch.isinf(max_corr), torch.zeros_like(max_corr), max_corr)
    
    # Std of correlations
    corr_diff = (corr.abs() - mean_corr.unsqueeze(-1)) * col_mask.float()
    std_corr = ((corr_diff ** 2).sum(dim=1) / n_valid.clamp(min=1)).sqrt()
    
    # Stack features: [B, 3]
    features = torch.stack([mean_corr, max_corr, std_corr], dim=-1)
    
    # Clamp correlations to valid range
    features = features.clamp(min=0.0, max=1.0)
    
    return features


def compute_cross_column_missingness_correlation(
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute correlations between missingness indicators across columns.
    
    MCAR Signature: Low cross-column correlation (independent missingness)
    MAR Signature: High cross-column correlation (shared predictor drives missingness)
    MNAR Signature: Variable
    
    Returns:
        features: [B, 3] tensor with [mean_cross_corr, max_cross_corr, n_high_corr_pairs]
    """
    B, max_rows, max_cols = is_observed.shape
    device = is_observed.device
    
    # Valid mask
    row_mask_exp = row_mask.unsqueeze(-1).float()
    valid_rows = row_mask_exp  # [B, max_rows, 1]
    
    # Missingness indicators
    is_missing = (1.0 - is_observed.float()) * valid_rows  # [B, max_rows, max_cols]
    
    # Number of valid rows
    n_rows = row_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
    
    # Compute correlation matrix between columns
    # Center the missingness indicators
    miss_mean = is_missing.sum(dim=1, keepdim=True) / n_rows.unsqueeze(-1)  # [B, 1, max_cols]
    miss_centered = (is_missing - miss_mean) * valid_rows  # [B, max_rows, max_cols]
    
    # Covariance matrix: [B, max_cols, max_cols]
    # cov[i,j] = sum_rows(miss_centered[:, i] * miss_centered[:, j])
    cov_matrix = torch.bmm(
        miss_centered.transpose(1, 2),  # [B, max_cols, max_rows]
        miss_centered                    # [B, max_rows, max_cols]
    )  # [B, max_cols, max_cols]
    
    # Standard deviations
    variances = cov_matrix.diagonal(dim1=1, dim2=2)  # [B, max_cols]
    stds = variances.sqrt().clamp(min=eps)  # [B, max_cols]
    
    # Correlation matrix
    # corr[i,j] = cov[i,j] / (std[i] * std[j])
    std_outer = stds.unsqueeze(-1) * stds.unsqueeze(-2)  # [B, max_cols, max_cols]
    corr_matrix = cov_matrix / std_outer.clamp(min=eps)  # [B, max_cols, max_cols]
    
    # Mask out invalid columns and diagonal
    col_mask_2d = col_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)  # [B, max_cols, max_cols]
    diag_mask = ~torch.eye(max_cols, dtype=torch.bool, device=device).unsqueeze(0)
    valid_pairs = col_mask_2d & diag_mask
    
    # Extract off-diagonal correlations
    corr_matrix = corr_matrix * valid_pairs.float()
    corr_matrix = torch.where(
        torch.isnan(corr_matrix) | torch.isinf(corr_matrix),
        torch.zeros_like(corr_matrix),
        corr_matrix
    )
    
    # Number of valid pairs
    n_pairs = valid_pairs.float().sum(dim=(1, 2)).clamp(min=1)  # [B]
    
    # Mean absolute cross-correlation
    mean_cross_corr = corr_matrix.abs().sum(dim=(1, 2)) / n_pairs
    
    # Max absolute cross-correlation
    corr_for_max = corr_matrix.abs().clone()
    corr_for_max[~valid_pairs] = -float('inf')
    max_cross_corr = corr_for_max.view(B, -1).max(dim=1).values
    max_cross_corr = torch.where(
        torch.isinf(max_cross_corr),
        torch.zeros_like(max_cross_corr),
        max_cross_corr
    )
    
    # Count of high-correlation pairs (|corr| > 0.3)
    high_corr_mask = (corr_matrix.abs() > 0.3) & valid_pairs
    n_high_corr = high_corr_mask.float().sum(dim=(1, 2)) / n_pairs  # Normalized
    
    # Stack features: [B, 3]
    features = torch.stack([mean_cross_corr, max_cross_corr, n_high_corr], dim=-1)
    
    # Clamp to valid range
    features = features.clamp(min=0.0, max=1.0)
    
    return features


def compute_distributional_stats(
    values: torch.Tensor,       # [B, max_rows, max_cols]
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute distributional statistics of observed values.
    
    MNAR Signature: Distributional distortions (high skewness, unusual kurtosis)
                    due to self-censoring or truncation
    MCAR/MAR: More normal-looking distributions (assuming original data is normal-ish)
    
    Returns:
        features: [B, 4] tensor with [mean_skew, mean_kurt, var_skew, var_kurt]
    """
    B, max_rows, max_cols = values.shape
    device = values.device
    
    # Valid observed mask
    row_mask_exp = row_mask.unsqueeze(-1).float()
    col_mask_exp = col_mask.unsqueeze(1).float()
    valid_mask = row_mask_exp * col_mask_exp * is_observed.float()  # [B, max_rows, max_cols]
    
    # Per-column statistics
    n_obs = valid_mask.sum(dim=1).clamp(min=1)  # [B, max_cols]
    
    # Mean per column
    obs_sum = (values * valid_mask).sum(dim=1)  # [B, max_cols]
    col_mean = obs_sum / n_obs  # [B, max_cols]
    
    # Variance per column
    centered = (values - col_mean.unsqueeze(1)) * valid_mask
    col_var = (centered ** 2).sum(dim=1) / n_obs.clamp(min=1)  # [B, max_cols]
    col_std = col_var.sqrt().clamp(min=eps)
    
    # Standardized values
    standardized = centered / col_std.unsqueeze(1).clamp(min=eps)
    standardized = standardized * valid_mask
    
    # Skewness per column: E[(X - mu)^3] / sigma^3
    skewness = (standardized ** 3).sum(dim=1) / n_obs.clamp(min=1)  # [B, max_cols]
    
    # Kurtosis per column: E[(X - mu)^4] / sigma^4 - 3 (excess kurtosis)
    kurtosis = (standardized ** 4).sum(dim=1) / n_obs.clamp(min=1) - 3  # [B, max_cols]
    
    # Handle invalid columns
    skewness = skewness * col_mask.float()
    kurtosis = kurtosis * col_mask.float()
    skewness = torch.where(torch.isnan(skewness) | torch.isinf(skewness), torch.zeros_like(skewness), skewness)
    kurtosis = torch.where(torch.isnan(kurtosis) | torch.isinf(kurtosis), torch.zeros_like(kurtosis), kurtosis)
    
    # Aggregate across columns
    n_valid_cols = col_mask.float().sum(dim=1).clamp(min=1)
    
    mean_skew = skewness.abs().sum(dim=1) / n_valid_cols
    mean_kurt = kurtosis.abs().sum(dim=1) / n_valid_cols
    
    # Variance of skewness/kurtosis (high variance suggests MNAR affecting some columns)
    skew_centered = (skewness.abs() - mean_skew.unsqueeze(-1)) * col_mask.float()
    var_skew = (skew_centered ** 2).sum(dim=1) / n_valid_cols.clamp(min=1)
    
    kurt_centered = (kurtosis.abs() - mean_kurt.unsqueeze(-1)) * col_mask.float()
    var_kurt = (kurt_centered ** 2).sum(dim=1) / n_valid_cols.clamp(min=1)
    
    # Stack features: [B, 4]
    features = torch.stack([mean_skew, mean_kurt, var_skew, var_kurt], dim=-1)
    
    # Clamp to reasonable range (skewness/kurtosis can be large but shouldn't be extreme)
    features = features.clamp(min=0.0, max=10.0)
    
    return features


def compute_littles_test_approx(
    values: torch.Tensor,       # [B, max_rows, max_cols]
    is_observed: torch.Tensor,  # [B, max_rows, max_cols]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute an approximation to Little's MCAR test statistic.
    
    Little's test: Under MCAR, the means of observed values should be equal
    across different missingness patterns. Large deviations suggest non-MCAR.
    
    Approximation: Compare mean of observed values for each column between
    rows with different missingness patterns.
    
    High statistic -> Evidence against MCAR (likely MAR or MNAR)
    Low statistic -> Consistent with MCAR
    
    Returns:
        features: [B, 2] tensor with [test_statistic, significance_proxy]
    """
    B, max_rows, max_cols = values.shape
    device = values.device
    
    # For a simplified approximation:
    # 1. Partition rows into "low missingness" and "high missingness" groups
    # 2. Compare column means between groups
    # 3. Large differences suggest non-MCAR
    
    # Valid mask
    row_mask_exp = row_mask.unsqueeze(-1).float()
    col_mask_exp = col_mask.unsqueeze(1).float()
    valid_mask = row_mask_exp * col_mask_exp
    
    # Count missing per row
    is_missing = 1.0 - is_observed.float()
    missing_per_row = (is_missing * valid_mask).sum(dim=2)  # [B, max_rows]
    
    # Median split: rows with below-median vs above-median missingness
    # (Use mean as approximation to avoid sorting)
    mean_missing = (missing_per_row * row_mask.float()).sum(dim=1) / row_mask.float().sum(dim=1).clamp(min=1)
    
    # Group masks
    low_miss_rows = (missing_per_row <= mean_missing.unsqueeze(-1)) & row_mask  # [B, max_rows]
    high_miss_rows = (missing_per_row > mean_missing.unsqueeze(-1)) & row_mask  # [B, max_rows]
    
    # Expand for column operations
    low_mask = low_miss_rows.unsqueeze(-1).float() * col_mask_exp * is_observed.float()
    high_mask = high_miss_rows.unsqueeze(-1).float() * col_mask_exp * is_observed.float()
    
    # Column means for each group
    low_sum = (values * low_mask).sum(dim=1)
    low_count = low_mask.sum(dim=1).clamp(min=1)
    low_mean = low_sum / low_count  # [B, max_cols]
    
    high_sum = (values * high_mask).sum(dim=1)
    high_count = high_mask.sum(dim=1).clamp(min=1)
    high_mean = high_sum / high_count  # [B, max_cols]
    
    # Pooled standard deviation
    pooled_var = (
        ((values - low_mean.unsqueeze(1)) ** 2 * low_mask).sum(dim=1) +
        ((values - high_mean.unsqueeze(1)) ** 2 * high_mask).sum(dim=1)
    ) / (low_count + high_count - 2).clamp(min=1)
    pooled_std = pooled_var.sqrt().clamp(min=eps)
    
    # Standardized mean difference per column
    mean_diff = (high_mean - low_mean).abs() / pooled_std
    mean_diff = mean_diff * col_mask.float()
    mean_diff = torch.where(torch.isnan(mean_diff) | torch.isinf(mean_diff), torch.zeros_like(mean_diff), mean_diff)
    
    # Aggregate into test statistic
    n_valid_cols = col_mask.float().sum(dim=1).clamp(min=1)
    test_stat = (mean_diff ** 2).sum(dim=1) / n_valid_cols  # Chi-squared-like
    
    # Significance proxy: fraction of columns with |d| > 0.5 (medium effect)
    sig_proxy = (mean_diff > 0.5).float().sum(dim=1) / n_valid_cols
    
    # Stack features: [B, 2]
    features = torch.stack([test_stat, sig_proxy], dim=-1)
    
    # Clamp to reasonable range
    features = features.clamp(min=0.0, max=10.0)
    
    return features


# =============================================================================
# Main Feature Extraction Function
# =============================================================================

def extract_missingness_features(
    tokens: torch.Tensor,       # [B, max_rows, max_cols, TOKEN_DIM]
    row_mask: torch.Tensor,     # [B, max_rows]
    col_mask: torch.Tensor,     # [B, max_cols]
    config: Optional[MissingnessFeatureConfig] = None,
) -> torch.Tensor:
    """
    Extract all missingness pattern features from tokenized batch.
    
    This is the main entry point for feature extraction. Call this from
    the model's forward pass to get features for the MoE gating network.
    
    Args:
        tokens: Tokenized batch from data loader.
        row_mask: Boolean mask for valid rows.
        col_mask: Boolean mask for valid columns.
        config: Feature extraction configuration.
    
    Returns:
        features: [B, n_features] tensor of missingness pattern features.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    # Extract values and observation mask from tokens
    values = tokens[..., IDX_VALUE]        # [B, max_rows, max_cols]
    is_observed = tokens[..., IDX_OBSERVED]  # [B, max_rows, max_cols]
    
    # Convert is_observed to boolean for masking operations
    is_observed_bool = is_observed > 0.5
    
    feature_list = []
    
    # 1. Missing rate statistics
    if config.include_missing_rate_stats:
        rate_features = compute_missing_rate_stats(
            is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(rate_features)
    
    # 2. Point-biserial correlations
    if config.include_pointbiserial:
        pb_features = compute_pointbiserial_correlations(
            values, is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(pb_features)
    
    # 3. Cross-column missingness correlations
    if config.include_cross_column_corr:
        cc_features = compute_cross_column_missingness_correlation(
            is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(cc_features)
    
    # 4. Distributional statistics
    if config.include_distributional:
        dist_features = compute_distributional_stats(
            values, is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(dist_features)
    
    # 5. Little's test approximation
    if config.include_littles_approx:
        littles_features = compute_littles_test_approx(
            values, is_observed_bool, row_mask, col_mask, config.eps
        )
        feature_list.append(littles_features)
    
    # Concatenate all features
    if feature_list:
        features = torch.cat(feature_list, dim=-1)
    else:
        # Return empty tensor if no features configured
        B = tokens.shape[0]
        features = torch.empty(B, 0, device=tokens.device)
    
    # Final cleanup: replace any remaining NaN/Inf and clamp to reasonable range
    features = torch.where(
        torch.isnan(features) | torch.isinf(features),
        torch.zeros_like(features),
        features
    )
    
    # Clamp to prevent extreme values that cause NaN in downstream computations
    features = features.clamp(min=-10.0, max=10.0)
    
    return features


# =============================================================================
# Feature Extractor Module (for integration into model)
# =============================================================================

class MissingnessFeatureExtractor(torch.nn.Module):
    """
    PyTorch module wrapper for missingness feature extraction.
    
    This can be added to the model architecture and will be included
    in the forward pass.
    
    Usage:
        >>> extractor = MissingnessFeatureExtractor()
        >>> features = extractor(tokens, row_mask, col_mask)
        >>> # Concatenate with evidence for MoE gating
        >>> gate_input = torch.cat([evidence, recon_errors, features], dim=-1)
    """
    
    def __init__(self, config: Optional[MissingnessFeatureConfig] = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
    
    @property
    def n_features(self) -> int:
        """Number of features extracted."""
        return self.config.n_features
    
    def forward(
        self,
        tokens: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract missingness features from batch."""
        return extract_missingness_features(tokens, row_mask, col_mask, self.config)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_feature_names(config: Optional[MissingnessFeatureConfig] = None) -> list:
    """
    Get human-readable names for each feature.
    
    Useful for debugging and analysis.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    names = []
    
    if config.include_missing_rate_stats:
        names.extend([
            "miss_rate_mean",
            "miss_rate_var",
            "miss_rate_range",
            "miss_rate_max",
        ])
    
    if config.include_pointbiserial:
        names.extend([
            "pointbiserial_mean",
            "pointbiserial_max",
            "pointbiserial_std",
        ])
    
    if config.include_cross_column_corr:
        names.extend([
            "cross_col_corr_mean",
            "cross_col_corr_max",
            "cross_col_high_corr_frac",
        ])
    
    if config.include_distributional:
        names.extend([
            "skewness_mean",
            "kurtosis_mean",
            "skewness_var",
            "kurtosis_var",
        ])
    
    if config.include_littles_approx:
        names.extend([
            "littles_stat",
            "littles_sig_proxy",
        ])
    
    return names