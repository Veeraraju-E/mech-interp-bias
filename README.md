# Bias Activation Patching Implementation

This repository implements activation patching methods for mechanistic interpretability of bias in language models, specifically focusing on GPT-2 Medium with StereoSet and WinoGender datasets.

## Overview

This implementation focuses on Section 4.1 (Activation Patching) methods:
- **Causal Tracing/Activation Patching**: Measure importance of each model component by patching activations
- **Attribution Patching**: Efficient gradient-based approximation of edge importance
- **Head/MLP Ablations**: Zero out entire attention heads or MLPs to measure bias impact

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

Before running experiments, download the datasets:

```bash
python download_datasets.py
```

This will download and save:
- **StereoSet**: 2,106 contexts (6,318 examples) from validation split (used as test)
- **WinoGender**: 240 examples from test split

The datasets are saved to `data/` directory as JSON files:
- `data/stereoset_test.json`
- `data/winogender_test.json`

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
4. Run causal activation patching experiments
5. Run attribution patching experiments
6. Run head/MLP ablation experiments
7. Save results to `results/` directory

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
├── data/                      # Dataset storage (created after download_datasets.py)
│   ├── stereoset_test.json    # StereoSet dataset
│   └── winogender_test.json   # WinoGender dataset
├── src/
│   ├── data_loader.py        # Dataset loading (StereoSet, WinoGender)
│   ├── model_setup.py         # GPT-2 Medium setup with TransformerLens
│   ├── bias_metrics.py        # Bias score computation
│   ├── activation_patching.py # Causal activation patching
│   ├── attribution_patching.py # Gradient-based attribution patching
│   ├── ablations.py          # Head/MLP ablation functions
│   └── experiments.py        # Main experiment orchestration
├── download_datasets.py       # Dataset download script
├── analysis.py                # Analysis and visualization
├── requirements.txt           # Dependencies
└── README.md
```

## Results

Results are saved in the `results/` directory:
- `summary.json`: Complete results summary
- `causal_patching_*.json`: Edge impact scores from causal patching
- `attribution_patching_*.json`: Attribution scores
- `head_ablations_*.json`: Head ablation impacts
- `mlp_ablations_*.json`: MLP ablation impacts
- `*_ranked.csv`: Ranked component lists

## References

- **Conmy et al. (2024)**: Edge Attribution Patching methodology
- **Chandna et al. (2025)**: Edge-level bias localization
- **Yang et al. (2025)**: Head-level ablation approach
- **Nanda (2023)**: Attribution patching via gradients

## Notes

- This implementation uses TransformerLens for activation hooking and patching
- Experiments are designed to fail fast with clear error messages
- Code follows research code principles: no placeholders, direct imports, minimal error handling

