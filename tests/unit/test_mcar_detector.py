"""Unit tests for MCAR detection"""

import pytest
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/projects/project_lacuna')

from lacuna.core.mcar_detector import MCARDetector

class TestMCARDetector:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = MCARDetector()
        
    def test_mcar_detector_initialization(self):
        """Test MCARDetector can be initialized with parameters"""
        detector = MCARDetector(alpha=0.01, min_pattern_freq=10)
        assert detector.alpha == 0.01
        assert detector.min_pattern_freq == 10
    
    def test_no_missing_data(self):
        """Test behavior when no data is missing"""
        # Complete data
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = self.detector.test(data)
        
        assert result['is_mcar_plausible'] == True
        assert result['missing_percentage'] == 0.0
        assert result['p_value'] == 1.0
        assert result['test_statistic'] == 0.0
        assert 'No missing data' in result['recommendation']
    
    def test_completely_missing_column(self):
        """Test behavior with completely missing column"""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'C': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = self.detector.test(data)
        
        # Should be able to handle this case
        assert isinstance(result['is_mcar_plausible'], (bool, np.bool_))
        assert result['missing_percentage'] > 0
        assert 'pattern_analysis' in result
    
    def test_simple_mcar_pattern(self):
        """Test with simple MCAR-like pattern"""
        np.random.seed(42)
        
        # Create data with random missingness (MCAR)
        data = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100),
            'C': np.random.normal(0, 1, 100)
        })
        
        # Introduce random missingness (20% missing)
        missing_mask = np.random.random((100, 3)) < 0.2
        data = data.mask(missing_mask)
        
        result = self.detector.test(data)
        
        # Should detect this as plausibly MCAR
        assert isinstance(result['is_mcar_plausible'], (bool, np.bool_))
        assert result['missing_percentage'] > 0
        assert result['missing_percentage'] < 30  # Should be around 20%
        assert isinstance(result['p_value'], (float, type(np.nan)))
    
    def test_systematic_missing_pattern(self):
        """Test with clearly non-MCAR pattern"""
        # Create data where missingness depends on values (MNAR-like)
        data = pd.DataFrame({
            'income': [10000, 50000, 100000, 150000, 200000] * 20,
            'age': np.random.randint(20, 70, 100),
            'response': np.random.choice([1, 2, 3, 4, 5], 100)
        })
        
        # High earners don't report income (systematic missingness)
        high_income_mask = data['income'] > 100000
        data.loc[high_income_mask, 'income'] = np.nan
        
        result = self.detector.test(data)
        
        # Should detect this as questionable for MCAR
        assert isinstance(result['is_mcar_plausible'], (bool, np.bool_))
        assert result['missing_percentage'] > 0
        assert 'pattern_analysis' in result
    
    def test_pattern_analysis(self):
        """Test missing data pattern analysis"""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, np.nan, 40, 50],
            'C': [0.1, 0.2, 0.3, np.nan, 0.5]
        })
        
        result = self.detector.test(data)
        pattern_analysis = result['pattern_analysis']
        
        assert 'total_patterns' in pattern_analysis
        assert 'complete_cases' in pattern_analysis
        assert 'incomplete_cases' in pattern_analysis
        assert 'missing_by_variable' in pattern_analysis
        assert pattern_analysis['total_patterns'] > 1
        assert pattern_analysis['complete_cases'] >= 0
        assert pattern_analysis['incomplete_cases'] >= 0
    
    def test_monotone_pattern_detection(self):
        """Test monotone missing pattern detection"""
        # Create monotone missing pattern
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 6],
            'B': [10, 20, np.nan, np.nan, np.nan, np.nan],
            'C': [0.1, 0.2, np.nan, np.nan, np.nan, np.nan]
        })
        
        result = self.detector.test(data)
        pattern_analysis = result['pattern_analysis']
        
        # Should detect monotone pattern
        assert 'is_monotone' in pattern_analysis
        assert isinstance(pattern_analysis['is_monotone'], bool)
    
    def test_empty_dataframe(self):
        """Test behavior with empty dataframe"""
        data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            self.detector.test(data)
    
    def test_single_row_data(self):
        """Test with single row of data"""
        data = pd.DataFrame({
            'A': [1],
            'B': [np.nan],
            'C': [3]
        })
        
        result = self.detector.test(data)
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert isinstance(result['is_mcar_plausible'], (bool, np.bool_))
        assert 'pattern_analysis' in result
    
    def test_high_dimensional_data(self):
        """Test with higher dimensional data"""
        np.random.seed(42)
        
        # Create 20-dimensional data
        data = pd.DataFrame(np.random.normal(0, 1, (50, 20)))
        
        # Add some random missingness
        missing_mask = np.random.random((50, 20)) < 0.1
        data = data.mask(missing_mask)
        
        result = self.detector.test(data)
        
        # Should handle high-dimensional case
        assert isinstance(result['is_mcar_plausible'], (bool, np.bool_))
        assert result['missing_percentage'] >= 0
    
    def test_confidence_assessment(self):
        """Test confidence assessment in results"""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5] * 20,  # 100 observations
            'B': [10, 20, 30, 40, 50] * 20,
            'C': [0.1, 0.2, 0.3, 0.4, 0.5] * 20
        })
        
        # Add minimal missingness
        data.iloc[0, 0] = np.nan
        data.iloc[1, 1] = np.nan
        
        result = self.detector.test(data)
        
        assert 'confidence' in result
        assert result['confidence'] in ['low', 'medium', 'high']
    
    def test_recommendation_generation(self):
        """Test that meaningful recommendations are generated"""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, np.nan, 50],
            'C': [0.1, 0.2, 0.3, 0.4, np.nan]
        })
        
        result = self.detector.test(data)
        
        assert 'recommendation' in result
        assert isinstance(result['recommendation'], str)
        assert len(result['recommendation']) > 10  # Should be meaningful text
    
    def test_degrees_of_freedom_calculation(self):
        """Test degrees of freedom calculation"""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, np.nan, 50]
        })
        
        result = self.detector.test(data)
        
        assert 'degrees_freedom' in result
        assert isinstance(result['degrees_freedom'], (int, float))
        assert result['degrees_freedom'] >= 0

class TestMCARDetectorEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.detector = MCARDetector()
    
    def test_all_nan_data(self):
        """Test with completely missing data"""
        data = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [np.nan, np.nan, np.nan],
            'C': [np.nan, np.nan, np.nan]
        })
        
        result = self.detector.test(data)
        
        # Should handle gracefully without crashing
        assert isinstance(result, dict)
        assert isinstance(result['is_mcar_plausible'], (bool, np.bool_))
    
    def test_single_column_data(self):
        """Test with single column"""
        data = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5]
        })
        
        result = self.detector.test(data)
        
        assert isinstance(result, dict)
        assert 'pattern_analysis' in result
    
    def test_non_numeric_data_handling(self):
        """Test behavior with non-numeric data"""
        data = pd.DataFrame({
            'A': ['a', 'b', np.nan, 'd'],
            'B': [1, 2, 3, np.nan],
            'C': [True, False, True, np.nan]
        })
        
        # Should handle mixed types gracefully
        result = self.detector.test(data)
        assert isinstance(result, dict)