# LACUNA Architecture

## System Overview

LACUNA uses a hierarchical approach to missing data mechanism assessment:

### Stage 1: MCAR Detection
- Pure statistical automation using Little's test
- Missing data pattern analysis  
- No ML/LLM required for this stage

### Stage 2: MAR vs MNAR Classification  
- Hybrid ML + LLM approach
- Mixture of Experts for domain-specific pattern recognition
- LLM fallback for edge cases and novel scenarios

### Stage 3: Evidence Synthesis
- Uncertainty quantification across all sources
- Structured reporting with confidence bounds
- Actionable sensitivity analysis recommendations

## Component Details

### Classical Diagnostics Engine
- Little's MCAR test with power analysis
- Missing data pattern visualization
- Logistic regression diagnostics

### Mixture of Experts Model
- Domain-specific experts: Clinical, Survey, Longitudinal, Observational
- Smart gating network for automatic expert selection
- Built on PubMedBERT for biomedical domain knowledge

### LLM Interpretive Layer
- Structured reasoning prompts for domain expertise
- Edge case handling and novel scenario analysis
- Transparent reasoning with uncertainty acknowledgment

## Training Strategy

### Synthetic Data Generation
- Domain-specific realistic scenarios
- Known MAR/MNAR mechanisms with ground truth labels
- Balanced representation across domains and complexity levels

### Model Training
- Multi-task learning: classification + uncertainty estimation
- Expert utilization balancing
- Uncertainty calibration loss functions

## Deployment

### Local Inference
- Fast MoE inference for common cases (sub-second)
- Local model deployment without API dependencies

### LLM Integration
- Confidence-based routing to LLM services
- Cost-efficient fallback strategy
- API-agnostic LLM interface
