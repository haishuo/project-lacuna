"""Forge-specific path management for Project LACUNA"""

from pathlib import Path

class LACUNAForgeConfig:
    """Forge-specific path management for Project LACUNA"""
    
    # Base directories (following Forge conventions)
    PROJECT_ROOT = Path("/mnt/projects/project_lacuna")
    DATA_ROOT = Path("/mnt/data/project_lacuna") 
    ARTIFACTS_ROOT = Path("/mnt/artifacts/project_lacuna")
    
    # Source code paths
    LACUNA_PACKAGE = PROJECT_ROOT / "lacuna"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    SCRIPTS_DIR = PROJECT_ROOT / "scripts"
    
    # Data paths
    SYNTHETIC_DATA = DATA_ROOT / "synthetic"
    REAL_WORLD_DATA = DATA_ROOT / "real_world"
    PREPROCESSED_DATA = DATA_ROOT / "preprocessed"
    SCRATCH_DIR = DATA_ROOT / "scratch"
    
    # Model artifacts
    MODELS_DIR = ARTIFACTS_ROOT / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    PRODUCTION_MODELS = MODELS_DIR / "production"
    
    # Outputs
    EVALUATIONS_DIR = ARTIFACTS_ROOT / "evaluations"
    REPORTS_DIR = ARTIFACTS_ROOT / "reports"
    VISUALIZATIONS_DIR = ARTIFACTS_ROOT / "visualizations"
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories"""
        for attr_name in dir(cls):
            if attr_name.endswith('_DIR') or attr_name.endswith('_ROOT'):
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(parents=True, exist_ok=True)
                    
    @classmethod
    def get_synthetic_data_path(cls, mechanism, domain):
        """Get path for specific synthetic dataset"""
        return cls.SYNTHETIC_DATA / f"{mechanism}_datasets" / domain
        
    @classmethod
    def get_model_checkpoint_path(cls, model_name, epoch=None):
        """Get path for model checkpoint"""
        if epoch:
            return cls.CHECKPOINTS_DIR / f"{model_name}_epoch_{epoch:02d}"
        return cls.CHECKPOINTS_DIR / f"{model_name}_final"
