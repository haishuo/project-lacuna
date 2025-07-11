"""MCAR data generation using real health datasets as foundation"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Union, List
import urllib.request
import os
from pathlib import Path
import sys

# Add project to path for imports
sys.path.append('/mnt/projects/project_lacuna')
from lacuna.utils.forge_config import LACUNAForgeConfig

class MCARSimulator:
    """Generate datasets with MCAR missingness patterns from complete health data"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize MCAR simulator
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.config = LACUNAForgeConfig()
        np.random.seed(random_state)
        
    def load_heart_disease_data(self, cache_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Load UCI Heart Disease dataset as base complete data
        
        Args:
            cache_dir: Directory to cache downloaded data
            
        Returns:
            Complete dataset ready for MCAR simulation
        """
        if cache_dir is None:
            cache_dir = self.config.REAL_WORLD_DATA / "benchmark_datasets"
            cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / "heart_disease_complete.csv"
        
        if cache_file.exists():
            print(f"Loading cached heart disease data from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("Downloading UCI Heart Disease dataset...")
        
        # UCI Heart Disease dataset URL (Cleveland data)
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        # Column names for heart disease dataset
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]
        
        try:
            # Download and load data
            data = pd.read_csv(url, names=column_names, na_values='?')
            
            # Clean the data - remove any existing missing values for clean baseline
            data_complete = data.dropna().copy()  # Add .copy() to avoid warnings
            
            # Convert appropriate columns to numeric
            numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
            for col in numeric_cols:
                data_complete.loc[:, col] = pd.to_numeric(data_complete[col], errors='coerce')
            
            # Remove any rows that couldn't be converted
            data_complete = data_complete.dropna()
            
            # Add meaningful column descriptions as metadata
            data_complete.attrs['column_descriptions'] = {
                'age': 'Age in years',
                'sex': 'Sex (1=male, 0=female)', 
                'cp': 'Chest pain type (1-4)',
                'trestbps': 'Resting blood pressure (mm Hg)',
                'chol': 'Serum cholesterol (mg/dl)',
                'fbs': 'Fasting blood sugar > 120 mg/dl (1=true, 0=false)',
                'restecg': 'Resting ECG results (0-2)',
                'thalach': 'Maximum heart rate achieved',
                'exang': 'Exercise induced angina (1=yes, 0=no)',
                'oldpeak': 'ST depression induced by exercise',
                'slope': 'Slope of peak exercise ST segment (1-3)',
                'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
                'thal': 'Thalassemia (3=normal, 6=fixed defect, 7=reversible)',
                'target': 'Heart disease presence (0=no, 1=yes)'
            }
            
            # Cache for future use
            data_complete.to_csv(cache_file, index=False)
            print(f"Heart disease data cached to {cache_file}")
            
            return data_complete
            
        except Exception as e:
            print(f"Failed to download heart disease data: {e}")
            # Fallback: create synthetic health-like data
            return self._create_synthetic_health_data()
    
    def _create_synthetic_health_data(self, n_patients: int = 300) -> pd.DataFrame:
        """
        Create synthetic health data as fallback if download fails
        
        Args:
            n_patients: Number of patients to simulate
            
        Returns:
            Synthetic complete health dataset
        """
        print(f"Creating synthetic health data with {n_patients} patients...")
        
        np.random.seed(self.random_state)
        
        # Generate realistic health measurements
        data = pd.DataFrame({
            'age': np.random.normal(54, 9, n_patients).astype(int),
            'sex': np.random.binomial(1, 0.68, n_patients),  # ~68% male (realistic for heart disease)
            'systolic_bp': np.random.normal(131, 17, n_patients),
            'diastolic_bp': np.random.normal(72, 12, n_patients), 
            'cholesterol': np.random.normal(246, 51, n_patients),
            'bmi': np.random.normal(25.8, 4.3, n_patients),
            'heart_rate': np.random.normal(150, 23, n_patients),
            'glucose': np.random.normal(120, 30, n_patients),
            'smoking': np.random.binomial(1, 0.25, n_patients),
            'family_history': np.random.binomial(1, 0.4, n_patients),
        })
        
        # Ensure realistic ranges
        data['age'] = np.clip(data['age'], 25, 80)
        data['systolic_bp'] = np.clip(data['systolic_bp'], 90, 200)
        data['diastolic_bp'] = np.clip(data['diastolic_bp'], 50, 120)
        data['cholesterol'] = np.clip(data['cholesterol'], 100, 400)
        data['bmi'] = np.clip(data['bmi'], 15, 45)
        data['heart_rate'] = np.clip(data['heart_rate'], 60, 220)
        data['glucose'] = np.clip(data['glucose'], 70, 300)
        
        # Add outcome variable (simplified risk model)
        risk_score = (
            0.02 * data['age'] + 
            0.3 * data['sex'] +
            0.01 * data['systolic_bp'] +
            0.002 * data['cholesterol'] +
            0.1 * data['smoking'] +
            0.2 * data['family_history'] +
            np.random.normal(0, 0.5, n_patients)
        )
        data['heart_disease'] = (risk_score > np.percentile(risk_score, 70)).astype(int)
        
        return data
    
    def generate_mcar_data(self, 
                          base_data: Optional[pd.DataFrame] = None,
                          missing_rate: float = 0.2,
                          variables_to_miss: Optional[List[str]] = None,
                          pattern_type: str = 'uniform') -> Dict[str, Any]:
        """
        Generate MCAR missingness on complete dataset
        
        Args:
            base_data: Complete dataset (if None, loads heart disease data)
            missing_rate: Overall proportion of values to make missing
            variables_to_miss: Specific variables to introduce missingness (if None, all numeric)
            pattern_type: 'uniform', 'variable_specific', or 'block'
            
        Returns:
            Dictionary with original data, MCAR data, and metadata
        """
        # Load base data if not provided
        if base_data is None:
            base_data = self.load_heart_disease_data()
        
        # Make a copy for modification
        mcar_data = base_data.copy()
        
        # Determine which variables can have missingness
        if variables_to_miss is None:
            # Default: all numeric columns except primary outcome
            numeric_cols = mcar_data.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude obvious outcome variables
            outcome_patterns = ['target', 'outcome', 'disease', 'death', 'survival']
            variables_to_miss = [col for col in numeric_cols 
                               if not any(pattern in col.lower() for pattern in outcome_patterns)]
        
        # Generate MCAR missingness based on pattern type
        if pattern_type == 'uniform':
            mcar_data, missing_info = self._uniform_mcar(mcar_data, variables_to_miss, missing_rate)
        elif pattern_type == 'variable_specific':
            mcar_data, missing_info = self._variable_specific_mcar(mcar_data, variables_to_miss, missing_rate)
        elif pattern_type == 'block':
            mcar_data, missing_info = self._block_mcar(mcar_data, variables_to_miss, missing_rate)
        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")
        
        return {
            'original_data': base_data,
            'mcar_data': mcar_data,
            'missing_mechanism': 'MCAR',
            'missing_rate': missing_rate,
            'pattern_type': pattern_type,
            'variables_affected': variables_to_miss,
            'missing_info': missing_info,
            'metadata': {
                'description': f'MCAR simulation with {missing_rate*100:.1f}% missingness',
                'pattern': pattern_type,
                'random_state': self.random_state,
                'generation_timestamp': pd.Timestamp.now(),
            }
        }
    
    def _uniform_mcar(self, data: pd.DataFrame, variables: List[str], missing_rate: float) -> tuple:
        """Apply uniform random missingness across specified variables"""
        # Use instance random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        missing_info = {'pattern': 'uniform', 'cells_made_missing': 0}
        
        # Calculate total cells that could be missing
        total_cells = len(data) * len(variables)
        cells_to_miss = int(total_cells * missing_rate)
        
        # Create flat list of (row, col) indices
        all_indices = [(i, var) for i in range(len(data)) for var in variables]
        
        # Randomly select cells to make missing
        rng.shuffle(all_indices)
        indices_to_miss = all_indices[:cells_to_miss]
        
        # Apply missingness
        for row_idx, col_name in indices_to_miss:
            data.iloc[row_idx, data.columns.get_loc(col_name)] = np.nan
            missing_info['cells_made_missing'] += 1
        
        return data, missing_info
    
    def _variable_specific_mcar(self, data: pd.DataFrame, variables: List[str], missing_rate: float) -> tuple:
        """Apply different missing rates to different variables (but still MCAR within each)"""
        # Use instance random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        missing_info = {'pattern': 'variable_specific', 'by_variable': {}}
        
        # Assign different missing rates to variables (but average to target rate)
        variable_rates = rng.uniform(0.05, 0.35, len(variables))
        # Scale to achieve target overall rate
        variable_rates = variable_rates * (missing_rate / np.mean(variable_rates))
        
        for var, var_rate in zip(variables, variable_rates):
            # For each variable, randomly select rows to make missing
            n_to_miss = int(len(data) * var_rate)
            rows_to_miss = rng.choice(len(data), size=n_to_miss, replace=False)
            
            data.loc[rows_to_miss, var] = np.nan
            missing_info['by_variable'][var] = {
                'rate': var_rate,
                'count': n_to_miss
            }
        
        return data, missing_info
    
    def _block_mcar(self, data: pd.DataFrame, variables: List[str], missing_rate: float) -> tuple:
        """Create block patterns of missingness (multiple variables missing together)"""
        # Use instance random state for reproducibility
        rng = np.random.RandomState(self.random_state)
        
        missing_info = {'pattern': 'block', 'blocks_created': 0}
        
        # Create blocks of 2-3 variables
        block_size = min(3, len(variables))
        n_rows_to_affect = int(len(data) * missing_rate)
        
        # Randomly select rows and variable combinations
        rows_to_miss = rng.choice(len(data), size=n_rows_to_affect, replace=False)
        
        for row_idx in rows_to_miss:
            # Randomly select a block of variables
            block_vars = rng.choice(variables, size=block_size, replace=False)
            
            # Make this block missing for this row
            for var in block_vars:
                data.iloc[row_idx, data.columns.get_loc(var)] = np.nan
            
            missing_info['blocks_created'] += 1
        
        return data, missing_info
    
    def validate_mcar_properties(self, mcar_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that generated data has MCAR properties
        
        Args:
            mcar_result: Result from generate_mcar_data()
            
        Returns:
            Validation results
        """
        from lacuna.core.mcar_detector import MCARDetector
        
        # Run our MCAR detector on the generated data
        detector = MCARDetector()
        detection_result = detector.test(mcar_result['mcar_data'])
        
        # Calculate actual missing rates
        mcar_data = mcar_result['mcar_data']
        actual_missing_rate = mcar_data.isnull().sum().sum() / mcar_data.size
        
        # Check missing patterns
        missing_matrix = mcar_data.isnull()
        pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size()
        
        validation = {
            'mcar_test_result': detection_result,
            'expected_mcar': True,
            'detected_as_mcar': detection_result['is_mcar_plausible'],
            'test_passes': detection_result['is_mcar_plausible'],
            'actual_missing_rate': actual_missing_rate,
            'target_missing_rate': mcar_result['missing_rate'],
            'rate_accuracy': abs(actual_missing_rate - mcar_result['missing_rate']) < 0.05,
            'num_missing_patterns': len(pattern_counts),
            'littles_test_pvalue': detection_result['p_value'],
            'recommendation': detection_result['recommendation']
        }
        
        return validation

# Example usage functions
def create_mcar_test_suite(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Create a comprehensive test suite of MCAR datasets
    
    Args:
        output_dir: Directory to save datasets
        
    Returns:
        Dictionary of generated datasets
    """
    if output_dir is None:
        config = LACUNAForgeConfig()
        output_dir = config.SYNTHETIC_DATA / "mcar_datasets"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    simulator = MCARSimulator(random_state=42)
    test_suite = {}
    
    # Different MCAR scenarios
    scenarios = [
        {'name': 'low_uniform', 'missing_rate': 0.1, 'pattern_type': 'uniform'},
        {'name': 'medium_uniform', 'missing_rate': 0.2, 'pattern_type': 'uniform'}, 
        {'name': 'high_uniform', 'missing_rate': 0.35, 'pattern_type': 'uniform'},
        {'name': 'variable_specific', 'missing_rate': 0.2, 'pattern_type': 'variable_specific'},
        {'name': 'block_pattern', 'missing_rate': 0.2, 'pattern_type': 'block'},
    ]
    
    for scenario in scenarios:
        print(f"Generating MCAR scenario: {scenario['name']}")
        
        result = simulator.generate_mcar_data(
            missing_rate=scenario['missing_rate'],
            pattern_type=scenario['pattern_type']
        )
        
        # Validate MCAR properties
        validation = simulator.validate_mcar_properties(result)
        result['validation'] = validation
        
        # Save dataset
        output_file = output_dir / f"mcar_{scenario['name']}.csv"
        result['mcar_data'].to_csv(output_file, index=False)
        
        test_suite[scenario['name']] = result
        
        print(f"  - Missing rate: {validation['actual_missing_rate']:.3f}")
        print(f"  - MCAR detected: {validation['detected_as_mcar']}")
        print(f"  - Little's p-value: {validation['littles_test_pvalue']:.3f}")
    
    return test_suite