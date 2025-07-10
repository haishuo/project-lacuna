# Project LACUNA

**L**LM-**A**ugmented **C**lassification of **UN**observed **A**ssumptions

A framework for quantifying uncertainty in missing data mechanisms (MCAR/MAR/MNAR) in biostatistics and social sciences.

## Overview

Project LACUNA addresses the gap in systematic tools for assessing missing data mechanism plausibility. Rather than relying on informal intuition, LACUNA provides:

- **Classical Diagnostics**: Little's MCAR test, pattern analysis
- **ML Classification**: Mixture of Experts trained on domain-specific patterns  
- **LLM Reasoning**: Structured reasoning for edge cases and domain knowledge integration
- **Uncertainty Quantification**: Explicit uncertainty bounds rather than categorical declarations

## Installation

```bash
cd /mnt/projects/project_lacuna
pip install -e .
```

## Quick Start

```python
from lacuna import LACUNAInferencePipeline

# Initialize pipeline
pipeline = LACUNAInferencePipeline.from_pretrained("lacuna-moe-v1")

# Analyze missing data mechanism
result = pipeline.analyze_missingness(
    data=your_dataframe,
    study_description="Longitudinal cohort study of cardiovascular outcomes",
    metadata={"domain": "clinical_trials", "design": "longitudinal"}
)

print(f"MAR probability: {result['mar_probability']:.2f}")
print(f"MNAR risk: {result['mnar_risk']:.2f}")
print(f"Recommended sensitivity analysis: {result['recommendations']}")
```

## Architecture

```
MCAR Detection (Statistical) → MAR vs MNAR (ML+LLM Hybrid) → Uncertainty Quantification
```

## Development

See `docs/development.md` for development setup and contribution guidelines.

## Citation

If you use LACUNA in your research, please cite:

```bibtex
@software{lacuna2024,
  title={LACUNA: LLM-Augmented Classification of UNobserved Assumptions},
  author={Hai-Shuo Shu},
  year={2025},
  url={https://github.com/username/project-lacuna}
}
```
