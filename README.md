# Project Lacuna

Systematic missing data mechanism classification using transformer-based pattern recognition.

## Installation

```bash
cd /mnt/projects/project_lacuna
conda create -n lacuna python=3.10
conda activate lacuna
pip install -e .
```

## Quick Start

```python
from lacuna import LacunaPipeline
from lacuna.config import LacunaConfig

# Load pipeline
config = LacunaConfig.default()
pipeline = LacunaPipeline.load("path/to/model", config)

# Analyze data
result = pipeline.analyze(data, metadata={'domain': 'clinical_trials'})
print(result)
```

## Development

See technical specification document for architecture details.
