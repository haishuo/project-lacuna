#!/bin/bash
# Setup Lacuna MVP directory structure
# Run from /mnt/projects/project_lacuna

set -e

echo "Setting up Lacuna MVP..."

# Create directories
mkdir -p lacuna/data/simulators
mkdir -p lacuna/models
mkdir -p lacuna/training
mkdir -p lacuna/baselines
mkdir -p lacuna/evaluation
mkdir -p scripts

# Create __init__.py files
cat > lacuna/__init__.py << 'EOF'
"""
Project Lacuna: Systematic Missing Data Mechanism Classification
"""
__version__ = "0.1.0"
EOF

cat > lacuna/data/__init__.py << 'EOF'
"""Data processing modules."""
from .feature_extractor import ColumnFeatureExtractor
from .dataset import LacunaDataset, create_dataloaders
EOF

cat > lacuna/data/simulators/__init__.py << 'EOF'
"""Synthetic data generators."""
from .generator import SyntheticGenerator, GeneratorConfig
EOF

cat > lacuna/models/__init__.py << 'EOF'
"""Model architectures."""
from .lacuna_mini import LacunaModel, LacunaMini, LacunaMiniConfig, Classifier
EOF

cat > lacuna/training/__init__.py << 'EOF'
"""Training utilities."""
from .trainer import Trainer, TrainerConfig
EOF

cat > lacuna/baselines/__init__.py << 'EOF'
"""Baseline models for comparison."""
from .logistic import LogisticBaseline
EOF

cat > lacuna/evaluation/__init__.py << 'EOF'
"""Evaluation metrics."""
EOF

# Create output directory
mkdir -p /mnt/artifacts/project_lacuna/mvp

echo "Done! Directory structure:"
find lacuna -name "*.py" | head -20

echo ""
echo "Next steps:"
echo "  1. Copy the module files from the artifacts"
echo "  2. Run: python scripts/verify_mvp.py"
echo "  3. Run: python scripts/train_mvp.py --n-train 10000 --epochs 20"