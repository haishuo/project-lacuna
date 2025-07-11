"""MCAR Detection using Little's test and pattern analysis"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import scipy.stats as stats
from scipy.linalg import inv
import warnings
from itertools import combinations

class MCARDetector:
    """Pure statistical MCAR detection - no ML/LLM needed"""
    
    def __init__(self, alpha: float = 0.05, min_pattern_freq: int = 3):
        """
        Initialize MCAR detector
        
        Args:
            alpha: Significance level for Little's test
            min_pattern_freq: Minimum frequency for pattern analysis (lowered from 5 to 3)
        """
        self.alpha = alpha
        self.min_pattern_freq = min_pattern_freq
    
    def test(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Little's MCAR test and pattern analysis
        
        Args:
            data: DataFrame with potential missing values
            
        Returns:
            Dictionary with test results and analysis
        """
        # Input validation
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        # Check if there's any missing data
        if not data.isnull().any().any():
            return {
                'is_mcar_plausible': True,
                'test_statistic': 0.0,
                'p_value': 1.0,
                'degrees_freedom': 0,
                'missing_percentage': 0.0,
                'pattern_analysis': {'total_patterns': 1, 'complete_cases': len(data)},
                'recommendation': 'No missing data detected - MCAR assumption satisfied',
                'confidence': 'high'
            }
        
        # Run Little's MCAR test
        littles_result = self._littles_test(data)
        
        # Analyze missing data patterns
        pattern_analysis = self._analyze_patterns(data)
        
        # Determine overall assessment
        is_mcar_plausible = bool(littles_result['p_value'] > self.alpha)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            littles_result, pattern_analysis, is_mcar_plausible
        )
        
        return {
            'is_mcar_plausible': is_mcar_plausible,
            'test_statistic': littles_result['test_statistic'],
            'p_value': littles_result['p_value'],
            'degrees_freedom': littles_result['degrees_freedom'],
            'missing_percentage': (data.isnull().sum().sum() / data.size) * 100,
            'pattern_analysis': pattern_analysis,
            'recommendation': recommendation,
            'confidence': self._assess_confidence(littles_result, pattern_analysis)
        }
    
    def _littles_test(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Implement Little's MCAR test
        
        The test statistic follows a chi-square distribution under the null hypothesis
        that data are MCAR.
        """
        try:
            # Convert to numpy array for computation
            X = data.values.astype(float)
            n, p = X.shape
            
            # Get missing data indicator matrix
            R = ~np.isnan(X)  # True where data is observed
            
            # Find unique missing patterns
            patterns = np.unique(R, axis=0)
            num_patterns = len(patterns)
            
            if num_patterns == 1:
                # Only one pattern (either all missing or all observed)
                return {
                    'test_statistic': 0.0,
                    'p_value': 1.0,
                    'degrees_freedom': 0,
                    'warning': 'Only one missing pattern found'
                }
            
            # Calculate pattern frequencies
            pattern_counts = []
            for pattern in patterns:
                count = np.sum(np.all(R == pattern, axis=1))
                pattern_counts.append(count)
            
            pattern_counts = np.array(pattern_counts)
            
            # Remove patterns with too few observations
            valid_patterns_mask = pattern_counts >= self.min_pattern_freq
            if not np.any(valid_patterns_mask):
                return {
                    'test_statistic': np.nan,
                    'p_value': np.nan,
                    'degrees_freedom': 0,
                    'warning': f'No patterns with >= {self.min_pattern_freq} observations'
                }
            
            patterns = patterns[valid_patterns_mask]
            pattern_counts = pattern_counts[valid_patterns_mask]
            num_patterns = len(patterns)
            
            # Calculate sufficient statistics for each pattern
            means_by_pattern = []
            covariances_by_pattern = []
            
            for i, pattern in enumerate(patterns):
                # Get observed variables for this pattern
                obs_vars = np.where(pattern)[0]
                
                if len(obs_vars) == 0:
                    continue
                
                # Get data for this pattern
                pattern_mask = np.all(R == pattern, axis=1)
                pattern_data = X[pattern_mask][:, obs_vars]
                
                if len(pattern_data) > 1:
                    means_by_pattern.append(np.mean(pattern_data, axis=0))
                    if len(obs_vars) > 1:
                        covariances_by_pattern.append(np.cov(pattern_data, rowvar=False))
                    else:
                        covariances_by_pattern.append(np.var(pattern_data))
                else:
                    means_by_pattern.append(pattern_data[0] if len(pattern_data) > 0 else np.array([]))
                    covariances_by_pattern.append(None)
            
            # Simplified test statistic calculation
            # This is a computational approximation of Little's test
            test_statistic = self._compute_test_statistic(
                X, R, patterns, pattern_counts, means_by_pattern
            )
            
            # Degrees of freedom approximation
            # This is simplified - full calculation is quite complex
            df = self._compute_degrees_freedom(patterns, p)
            
            # P-value from chi-square distribution
            if df > 0 and not np.isnan(test_statistic):
                p_value = 1 - stats.chi2.cdf(test_statistic, df)
            else:
                p_value = np.nan
            
            return {
                'test_statistic': test_statistic,
                'p_value': p_value,
                'degrees_freedom': df
            }
            
        except Exception as e:
            warnings.warn(f"Little's test computation failed: {str(e)}")
            return {
                'test_statistic': np.nan,
                'p_value': np.nan,
                'degrees_freedom': 0,
                'error': str(e)
            }
    
    def _compute_test_statistic(self, X, R, patterns, pattern_counts, means_by_pattern):
        """Compute simplified test statistic for Little's test"""
        try:
            n, p = X.shape
            test_stat = 0.0
            
            # Overall mean (using available data)
            overall_means = []
            for j in range(p):
                col_data = X[:, j]
                valid_data = col_data[~np.isnan(col_data)]
                if len(valid_data) > 0:
                    overall_means.append(np.mean(valid_data))
                else:
                    overall_means.append(0.0)
            overall_means = np.array(overall_means)
            
            # Compare pattern means to overall means
            for i, (pattern, count, pattern_mean) in enumerate(zip(patterns, pattern_counts, means_by_pattern)):
                if pattern_mean is None or len(pattern_mean) == 0:
                    continue
                
                obs_vars = np.where(pattern)[0]
                if len(obs_vars) == 0:
                    continue
                
                # Difference between pattern mean and overall mean
                mean_diff = pattern_mean - overall_means[obs_vars]
                
                # Weight by pattern frequency
                contribution = count * np.sum(mean_diff ** 2)
                test_stat += contribution
            
            return test_stat
            
        except Exception:
            return np.nan
    
    def _compute_degrees_freedom(self, patterns, p):
        """Compute degrees of freedom (simplified approximation)"""
        try:
            # Count number of free parameters
            total_params = 0
            for pattern in patterns:
                obs_vars = np.sum(pattern)
                if obs_vars > 0:
                    # Parameters for mean of observed variables
                    total_params += obs_vars
            
            # Subtract constraints
            # This is a simplified calculation
            df = max(0, total_params - p)
            return df
            
        except Exception:
            return 0
    
    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        # Missing data indicator matrix
        missing_matrix = data.isnull()
        
        # Convert patterns to tuples for counting
        pattern_tuples = [tuple(row) for row in missing_matrix.values]
        pattern_counts = pd.Series(pattern_tuples).value_counts()
        
        # Total number of patterns
        total_patterns = len(pattern_counts)
        
        # Complete cases (no missing data)
        complete_pattern = tuple([False] * len(data.columns))
        complete_cases = pattern_counts.get(complete_pattern, 0)
        
        # Cases with any missing data
        incomplete_cases = len(data) - complete_cases
        
        # Most common patterns
        top_patterns = pattern_counts.head(5).to_dict()
        
        # Pattern analysis
        monotone_missing = self._check_monotone_pattern(missing_matrix)
        
        # Missing percentages by variable
        missing_by_var = (missing_matrix.sum() / len(data) * 100).to_dict()
        
        return {
            'total_patterns': total_patterns,
            'complete_cases': complete_cases,
            'incomplete_cases': incomplete_cases,
            'completion_rate': (complete_cases / len(data)) * 100,
            'top_patterns': top_patterns,
            'is_monotone': monotone_missing,
            'missing_by_variable': missing_by_var,
            'total_missing_cells': missing_matrix.sum().sum(),
            'sparsity': (missing_matrix.sum().sum() / missing_matrix.size) * 100
        }
    
    def _check_monotone_pattern(self, missing_matrix: pd.DataFrame) -> bool:
        """
        Check if missing data follows a monotone pattern
        
        Monotone means if variable j is missing, then all variables j+1, j+2, ... 
        are also missing (for some ordering of variables)
        """
        try:
            # Try different orderings of variables to see if any gives monotone pattern
            for perm in combinations(range(len(missing_matrix.columns)), len(missing_matrix.columns)):
                reordered = missing_matrix.iloc[:, list(perm)]
                
                # Check if this ordering gives monotone pattern
                is_monotone = True
                for _, row in reordered.iterrows():
                    # Find first missing value
                    missing_indices = np.where(row)[0]
                    if len(missing_indices) > 0:
                        first_missing = missing_indices[0]
                        # Check if all subsequent values are also missing
                        if not all(row.iloc[first_missing:]):
                            is_monotone = False
                            break
                
                if is_monotone:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _generate_recommendation(self, littles_result: Dict, pattern_analysis: Dict, 
                               is_mcar_plausible: bool) -> str:
        """Generate recommendation based on test results"""
        
        if 'error' in littles_result:
            return "Little's test failed to compute. Consider manual inspection of missing data patterns."
        
        if 'warning' in littles_result:
            return f"Limited test reliability: {littles_result['warning']}. Consider descriptive analysis."
        
        p_value = littles_result['p_value']
        total_patterns = pattern_analysis['total_patterns']
        completion_rate = pattern_analysis['completion_rate']
        
        if is_mcar_plausible:
            if p_value > 0.1:
                confidence = "strong"
            else:
                confidence = "moderate"
            
            recommendation = f"MCAR assumption is plausible (p={p_value:.3f}). {confidence.capitalize()} evidence for completely random missingness."
        else:
            recommendation = f"MCAR assumption is questionable (p={p_value:.3f}). Consider MAR vs MNAR analysis."
        
        # Add pattern-based insights
        if total_patterns == 1:
            recommendation += " Single missing pattern detected."
        elif total_patterns > 10:
            recommendation += f" Complex missing pattern ({total_patterns} patterns) suggests systematic missingness."
        
        if completion_rate < 50:
            recommendation += " High missingness rate - exercise caution in interpretation."
        
        return recommendation
    
    def _assess_confidence(self, littles_result: Dict, pattern_analysis: Dict) -> str:
        """Assess confidence in the MCAR test result"""
        
        if 'error' in littles_result or np.isnan(littles_result.get('p_value', np.nan)):
            return 'low'
        
        completion_rate = pattern_analysis['completion_rate']
        total_patterns = pattern_analysis['total_patterns']
        
        # High confidence conditions
        if (completion_rate > 70 and 
            total_patterns <= 5 and 
            not np.isnan(littles_result['p_value'])):
            return 'high'
        
        # Medium confidence conditions  
        if (completion_rate > 50 and 
            total_patterns <= 10):
            return 'medium'
        
        # Low confidence otherwise
        return 'low'