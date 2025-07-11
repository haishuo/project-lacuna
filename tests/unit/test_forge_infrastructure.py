"""Unit tests for Forge infrastructure validation"""

import pytest
import sys
from pathlib import Path

# Add project to path
sys.path.append('/mnt/projects/project_lacuna')

from lacuna.utils.forge_config import LACUNAForgeConfig

class TestForgeInfrastructure:
    """Test that required Forge directories exist"""
    
    @classmethod
    def setup_class(cls):
        """Setup test fixtures"""
        cls.config = LACUNAForgeConfig()
    
    def test_base_forge_mounts_exist(self):
        """Test that Forge three-drive structure exists"""
        assert Path('/mnt/projects').exists(), "Projects drive missing"
        assert Path('/mnt/data').exists(), "Data drive missing" 
        assert Path('/mnt/artifacts').exists(), "Artifacts drive missing"
    
    def test_project_directories_exist(self):
        """Test that LACUNA project directories exist"""
        assert self.config.PROJECT_ROOT.exists(), f"Project root missing: {self.config.PROJECT_ROOT}"
        assert self.config.DATA_ROOT.exists(), f"Data root missing: {self.config.DATA_ROOT}"
        assert self.config.ARTIFACTS_ROOT.exists(), f"Artifacts root missing: {self.config.ARTIFACTS_ROOT}"
    
    def test_source_structure_exists(self):
        """Test that source code structure exists"""
        assert self.config.LACUNA_PACKAGE.exists(), f"LACUNA package missing: {self.config.LACUNA_PACKAGE}"
        assert self.config.CONFIGS_DIR.exists(), f"Configs directory missing: {self.config.CONFIGS_DIR}"
        assert self.config.NOTEBOOKS_DIR.exists(), f"Notebooks directory missing: {self.config.NOTEBOOKS_DIR}"
        assert self.config.SCRIPTS_DIR.exists(), f"Scripts directory missing: {self.config.SCRIPTS_DIR}"
    
    def test_data_directories_exist(self):
        """Test that data directories exist"""
        assert self.config.SYNTHETIC_DATA.exists(), f"Synthetic data dir missing: {self.config.SYNTHETIC_DATA}"
        assert self.config.REAL_WORLD_DATA.exists(), f"Real world data dir missing: {self.config.REAL_WORLD_DATA}"
        assert self.config.PREPROCESSED_DATA.exists(), f"Preprocessed data dir missing: {self.config.PREPROCESSED_DATA}"
        assert self.config.SCRATCH_DIR.exists(), f"Scratch dir missing: {self.config.SCRATCH_DIR}"
    
    def test_model_directories_exist(self):
        """Test that model artifact directories exist"""
        assert self.config.MODELS_DIR.exists(), f"Models dir missing: {self.config.MODELS_DIR}"
        assert self.config.CHECKPOINTS_DIR.exists(), f"Checkpoints dir missing: {self.config.CHECKPOINTS_DIR}"
        assert self.config.PRODUCTION_MODELS.exists(), f"Production models dir missing: {self.config.PRODUCTION_MODELS}"
    
    def test_output_directories_exist(self):
        """Test that output directories exist"""
        assert self.config.EVALUATIONS_DIR.exists(), f"Evaluations dir missing: {self.config.EVALUATIONS_DIR}"
        assert self.config.REPORTS_DIR.exists(), f"Reports dir missing: {self.config.REPORTS_DIR}"
        assert self.config.VISUALIZATIONS_DIR.exists(), f"Visualizations dir missing: {self.config.VISUALIZATIONS_DIR}"
    
    def test_python_environment_ready(self):
        """Test that Python environment has required packages"""
        try:
            import torch, transformers, numpy, pandas, scipy, statsmodels
            assert torch.cuda.is_available(), "CUDA not available"
            
            # Test package import
            from lacuna import MCARDetector
            
        except ImportError as e:
            pytest.fail(f"Missing required package: {e}")