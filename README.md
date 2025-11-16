# Bias Activation Patching Implementation

This repository implements activation patching methods for mechanistic interpretability of bias in language models, specifically focusing on GPT-2 Medium with StereoSet and WinoGender datasets.

## Overview

This implementation focuses on Activation Patching methods:
- **Attribution Patching**: Efficient gradient-based approximation of edge importance (Nanda 2023)
- **Head/MLP Ablations**: Zero out entire attention heads or MLPs to measure bias impact (Yang et al. 2025)

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

### Supported Datasets

- **StereoSet**: Gender and racial bias benchmark (Nadeem et al. 2021)
  - 2,106 contexts (6,318 examples) from validation split (used as test)
  - Measures stereotype vs antistereotype completions
  
- **WinoGender**: Gender bias in coreference resolution (Rudinger et al. 2018)
  - 240 examples from test split
  - Measures pronoun prediction accuracy for gender-neutral professions

### Dataset Setup

Before running experiments, download the datasets:

```bash
python download_datasets.py
```

This downloads and saves datasets to `data/` directory:
- `data/stereoset_test.json`
- `data/winogender_test.json`

## Models

### Supported Models

- **GPT-2 Medium** (350M parameters)
  - Decoder-only transformer architecture
  - Loaded via TransformerLens (HookedTransformer)
  - Enables activation hooking and patching

### Model Configuration

Models are automatically loaded and configured via `src/model_setup.py`:
- Model weights from HuggingFace
- Tokenizer configuration

## Evaluation and Metrics

### Bias Metrics

**StereoSet Score (SS)**: 
- Computes log probability difference between stereotype and antistereotype completions
- SS = mean(log P(stereotype_token | context) - log P(antistereotype_token | context))
- Higher scores indicate more bias toward stereotypes

**WinoGender Score**:
- Measures pronoun prediction accuracy difference
- Compares model's preference for correct pronouns given profession
- Positive scores indicate male bias, negative scores indicate female bias

### Metric Implementation

Bias metrics are implemented in `src/bias_metrics.py`:
- `compute_stereoset_score()`: StereoSet SS calculation
- `compute_winogender_score()`: WinoGender accuracy difference
- `compute_bias_metric()`: Dispatcher function

## Methods

### Activation Patching Methods

All activation patching methods are located in `src/methods/activation_patching/`:

#### Attribution Patching
- **File**: `src/methods/activation_patching/attribution_patching.py`
- **Method**: Gradient-based edge importance estimation
- **Reference**: Nanda (2023)
- **Approach**: Computes `grad(bias_metric) @ activation` for each edge
- **Efficiency**: Single backward pass per example, averages over all examples

#### Head/MLP Ablations
- **File**: `src/methods/activation_patching/ablations.py`
- **Method**: Zero out attention heads or MLPs and measure bias change
- **Reference**: Yang et al. (2025)
- **Approach**: 
  - Ablate individual heads: zero out specific head output
  - Ablate MLPs: zero out entire MLP output
  - Measure change in bias metric: `bias(ablated) - bias(original)`
- **Averaging**: Results averaged over all examples

## Usage

### Running Experiments

Run the main experiment script:

```bash
python -m src.experiments
```

This will:
1. Load GPT-2 Medium model with TransformerLens
2. Load StereoSet and WinoGender datasets
3. Compute baseline bias metrics
4. Run attribution patching experiments
5. Run head/MLP ablation experiments
6. Save results to `results/` directory

### Analysis and Visualization

Generate summary reports and visualizations:

```bash
python analysis.py
```

This creates:
- Text summary report (`results/summary_report.txt`)
- Visualization plots (`results/visualizations/`)

## Project Structure

```
fmga/
├── data/                                    # Dataset storage
│   ├── stereoset_test.json                  # StereoSet dataset
│   └── winogender_test.json                 # WinoGender dataset
├── src/
│   ├── data_loader.py                       # Dataset loading utilities
│   ├── model_setup.py                       # Model loading and setup
│   ├── bias_metrics.py                      # Bias score computation
│   ├── experiments.py                       # Main experiment orchestration
│   └── methods/
│       └── activation_patching/
│           ├── __init__.py
│           ├── attribution_patching.py      # Attribution patching
│           ├── ablations.py                 # Head/MLP ablations
│           └── hook_points.py               # Hook point utilities
├── download_datasets.py                     # Dataset download script
├── analysis.py                              # Analysis and visualization
├── requirements.txt                         # Dependencies
└── README.md
```

## Results

Results are saved in the `results/` directory:
- `summary.json`: Complete results summary
- `attribution_patching_*.json`: Attribution scores for edges
- `head_ablations_*.json`: Head ablation impacts
- `mlp_ablations_*.json`: MLP ablation impacts

## Notes

- This implementation uses TransformerLens for activation hooking and patching
- Experiments are designed to fail fast with clear error messages
- Code follows research code principles: no placeholders, direct imports, minimal error handling
- All activation patching methods from Section 4.1 are implemented and organized in `src/methods/activation_patching/`