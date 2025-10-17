"""
lacuna.utils.forge_config

Purpose: Forge-specific path configuration

Design Principles:
- UNIX Philosophy: Do ONE thing well
- No defaults (except top-level config)
- Trust neighbors (no redundant validation)
- Fail fast and loud
- Target: <150 lines

Spec Reference: Section Appendix
"""

from pathlib import Path


class ForgeConfig:
    """Forge-specific path management"""
    
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

