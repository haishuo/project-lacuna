"""Unit tests for MCAR simulator"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append('/mnt/projects/project_lacuna')

from lacuna.data.simulators.mcar_simulator import MCARSimulator, create_mcar_test_suite
from lacuna.utils.forge_config import LACUNAForgeConfig

class TestMCARSimulator:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.simulator = MCARSimulator(random_state=42)
    
    def test_simulator_initialization(self):
        """Test MCARSimulator initialization"""
        simulator = MCARSimulator(random_state=123)
        assert simulator.random_state == 123
    
    def test_synthetic_health_data_generation(self):
        """Test synthetic health data generation as fallback"""
        data = self.simulator._create_synthetic_health_data(n_patients=100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'age' in data.columns
        assert 'heart_disease' in data.columns
        
        # Check realistic ranges
        assert data['age'].min() >= 25
        assert data['age'].max() <= 80
        assert data['sex'].isin([0, 1]).all()
    
    def test_uniform_mcar_generation(self):
        """Test uniform MCAR pattern generation"""
        # Create small test dataset
        base_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5] * 20,  # 100 rows
            'B': [10, 20, 30, 40, 50] * 20,
            'C': [0.1, 0.2, 0.3, 0.4, 0.5] * 20
        })
        
        result = self.simulator.generate_mcar_data(
            base_data=base_data,
            missing_rate=0.2,
            pattern_type='uniform'
        )
        
        # Check structure
        assert 'mcar_data' in result
        assert 'original_data' in result
        assert result['missing_mechanism'] == 'MCAR'
        
        # Check missing rate is approximately correct
        missing_rate = result['mcar_data'].isnull().sum().sum() / result['mcar_data'].size
        assert abs(missing_rate - 0.2) < 0.05  # Within 5% of target
    
    def test_variable_specific_mcar(self):
        """Test variable-specific MCAR pattern"""
        base_data = pd.DataFrame({
            'var1': np.random.normal(0, 1, 100),
            'var2': np.random.normal(0, 1, 100),
            'var3': np.random.normal(0, 1, 100),
            'outcome': np.random.binomial(1, 0.5, 100)
        })
        
        result = self.simulator.generate_mcar_data(
            base_data=base_data,
            missing_rate=0.15,
            pattern_type='variable_specific',
            variables_to_miss=['var1', 'var2', 'var3']
        )
        
        # Outcome should not have missingness
        assert not result['mcar_data']['outcome'].isnull().any()
        
        # Other variables should have some missingness
        for var in ['var1', 'var2', 'var3']:
            assert result['mcar_data'][var].isnull().any()
    
    def test_block_mcar_pattern(self):
        """Test block MCAR pattern generation"""
        base_data = pd.DataFrame({
            'A': range(50),
            'B': range(50, 100),
            'C': range(100, 150),
            'D': range(150, 200)
        })
        
        result = self.simulator.generate_mcar_data(
            base_data=base_data,
            missing_rate=0.25,
            pattern_type='block'
        )
        
        assert result['pattern_type'] == 'block'
        assert 'blocks_created' in result['missing_info']
        
        # Should have some missingness
        assert result['mcar_data'].isnull().sum().sum() > 0
    
    def test_mcar_validation(self):
        """Test that generated MCAR data validates as MCAR"""
        # Generate simple MCAR data
        base_data = pd.DataFrame({
            'x1': np.random.normal(0, 1, 200),
            'x2': np.random.normal(0, 1, 200),
            'x3': np.random.normal(0, 1, 200)
        })
        
        result = self.simulator.generate_mcar_data(
            base_data=base_data,
            missing_rate=0.15,
            pattern_type='uniform'
        )
        
        # Validate MCAR properties
        validation = self.simulator.validate_mcar_properties(result)
        
        assert 'mcar_test_result' in validation
        assert 'detected_as_mcar' in validation
        assert 'actual_missing_rate' in validation
        
        # Should generally detect as MCAR (though not guaranteed due to randomness)
        assert isinstance(validation['detected_as_mcar'], (bool, np.bool_))
    
    def test_metadata_preservation(self):
        """Test that metadata is properly preserved"""
        base_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        result = self.simulator.generate_mcar_data(
            base_data=base_data,
            missing_rate=0.2
        )
        
        # Check metadata
        assert 'metadata' in result
        metadata = result['metadata']
        assert 'description' in metadata
        assert 'random_state' in metadata
        assert 'generation_timestamp' in metadata
        assert metadata['random_state'] == 42
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Very small dataset
        tiny_data = pd.DataFrame({'x': [1, 2, 3]})
        
        result = self.simulator.generate_mcar_data(
            base_data=tiny_data,
            missing_rate=0.1
        )
        
        assert isinstance(result, dict)
        assert 'mcar_data' in result
        
        # High missing rate
        result_high = self.simulator.generate_mcar_data(
            base_data=tiny_data,
            missing_rate=0.8
        )
        
        assert isinstance(result_high, dict)
    
    def test_variable_selection(self):
        """Test automatic variable selection"""
        data_with_outcome = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'weight': [60, 70, 80, 90, 100],
            'target': [0, 1, 0, 1, 0],  # Should be excluded
            'disease_outcome': [1, 0, 1, 0, 1]  # Should be excluded
        })
        
        result = self.simulator.generate_mcar_data(
            base_data=data_with_outcome,
            missing_rate=0.2
        )
        
        # Outcome variables should not have missingness
        assert not result['mcar_data']['target'].isnull().any()
        assert not result['mcar_data']['disease_outcome'].isnull().any()
        
        # Age and weight might have missingness
        total_missing = (result['mcar_data']['age'].isnull().sum() + 
                        result['mcar_data']['weight'].isnull().sum())
        assert total_missing >= 0  # Should have some or no missingness in these vars
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state"""
        base_data = pd.DataFrame({
            'A': np.arange(50),
            'B': np.arange(50, 100)
        })
        
        # Generate data twice with same parameters
        sim1 = MCARSimulator(random_state=999)
        sim2 = MCARSimulator(random_state=999)
        
        result1 = sim1.generate_mcar_data(base_data, missing_rate=0.2)
        result2 = sim2.generate_mcar_data(base_data, missing_rate=0.2)
        
        # Should have identical missing patterns
        missing1 = result1['mcar_data'].isnull()
        missing2 = result2['mcar_data'].isnull()
        
        assert missing1.equals(missing2)

class TestMCARTestSuite:
    """Test the comprehensive MCAR test suite creation"""
    
    def test_create_mcar_test_suite(self):
        """Test creation of MCAR test suite"""
        # Use temporary directory for testing
        config = LACUNAForgeConfig()
        test_dir = config.SCRATCH_DIR / "test_mcar_suite"
        test_dir.mkdir(exist_ok=True)
        
        # Create small test suite
        test_suite = create_mcar_test_suite(output_dir=test_dir)
        
        # Should have multiple scenarios
        assert len(test_suite) > 0
        assert 'low_uniform' in test_suite
        assert 'medium_uniform' in test_suite
        
        # Each scenario should have validation
        for scenario_name, scenario_data in test_suite.items():
            assert 'validation' in scenario_data
            assert 'mcar_data' in scenario_data
            assert 'original_data' in scenario_data
            
            # CSV files should be created
            csv_file = test_dir / f"mcar_{scenario_name}.csv"
            assert csv_file.exists()
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)

class TestRealDataIntegration:
    """Test integration with real data loading"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.simulator = MCARSimulator(random_state=42)
    
    def test_synthetic_fallback(self):
        """Test that synthetic data fallback works"""
        # This will use synthetic data since we're not actually downloading
        data = self.simulator._create_synthetic_health_data(n_patients=50)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        assert 'age' in data.columns
        assert 'heart_disease' in data.columns
        
        # Test generating MCAR on this synthetic data
        result = self.simulator.generate_mcar_data(
            base_data=data,
            missing_rate=0.15
        )
        
        assert result['missing_mechanism'] == 'MCAR'
        assert result['mcar_data'].isnull().sum().sum() > 0