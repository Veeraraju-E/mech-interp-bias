# Fantastic Biases and Where to Find Them

We present our work on mechanistically interpreting mitigate racial and gender bias in LLMs. We specifically focus on OSS models like GPT-2 (Medium/Large) and Eleuther AI's GPT-Neo on StereoSet (gender and racial bias) and WinoGender (gender bias) datasets.

---

## 1. Repository Layout

```
root/
├── requirements.txt                # Python dependencies
├── download_datasets.py            # StereoSet + WinoGender downloader preprocessor
├── src/
│   ├── __init__.py
│   ├── bias_metrics.py             # Bias metrics + closures for patching
│   ├── cache_utils.py              # On-disk caching helpers (scores, activations, etc.)
│   ├── data_loader.py              # Dataset normalization + prompt builders
│   ├── model_setup.py              # HookedTransformer + tokenizer initialization
│   └── methods/
│       ├── patching/
│       │   ├── ablations.py        # Head/MLP ablation scans
│       │   ├── attribution_patching.py
│       │   └── experiments.py      # End-to-end activation patching pipeline
│       ├── linear_probing/
│       │   └── probe.py            # Layer-wise logistic probes + CLI
│       └── steering/
│           ├── experiments.py      # SAE steering experiments
│           ├── dataset.py           # Activation dataset collection
│           ├── train_sae.py         # SAE training
│           └── evaluation.py       # Bias reduction evaluation
├── runs/                           # Cache directories (created on demand)
└── results/                        # JSON + figure outputs (per model)
```

---

## 2. Environment Setup

1. **Python environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Model + tokenizer loading**  
   `src/model_setup.py` wraps `HookedTransformer.from_pretrained` and always mirrors the tokenizer (padding token set to EOS). Supported model identifiers:
   - `gpt2-medium`
   - `gpt2-large`
   - `gpt-neo-125M` (loaded as `EleutherAI/gpt-neo-125M`)

---

## 3. Data Preparation

```bash
python download_datasets.py
```

This script fetches and normalizes:

| Dataset      | File                                    | Notes |
|--------------|-----------------------------------------|-------|
| StereoSet    | `data/stereoset_test.json`              | Uses validation split, retains bias type metadata |
| WinoGender   | `data/winogender_test.json`             | Reconstructs templates, pronouns, and example IDs |

Additional experiment code expects preprocessed ACDC pairs:

```
data/stereoset_gender_acdc_pairs.json
data/stereoset_race_acdc_pairs.json
```

These files should contain the `{"clean": {"tokens": [...], "metadata": {...}}, ...}` format described in `src/data_loader.load_stereoset_acdc_pairs`.

---

## 4. Activation Patching Pipeline

The main orchestration lives in `src/methods/patching/experiments.py`. It runs:
   - **Attribution patching** (`attribution_patching.py`): gradient-based edge scoring with hook names such as `blocks.5.attn.hook_z`.
   - **Head/MLP ablation scans** (`ablations.py`): zeroes each head or MLP output and measures ΔB.

### Run it

```bash
python -m src.methods.patching.experiments \
  --model gpt2-medium \
  --output-dir results \
  --cache-dir runs/activation_patching/cache
```

Outputs per dataset (`stereoset_race`, `stereoset_gender`, `winogender`):

| File | Description |
|------|-------------|
| `baseline` entry in `summary.json` | Scalar bias score |
| `attribution_patching_<dataset>.json` | Hook name → attribution score |
| `head_ablations_<dataset>.json` | `"layer_head"` → Δbias |
| `mlp_ablations_<dataset>.json` | `layer` → Δbias |

---

## 6. Linear Probing

`src/methods/linear_probing/probe.py` trains logistic probes on residual stream activations to see where bias becomes linearly decodable.

```bash
python -m src.methods.linear_probing.probe \
  --model gpt2-medium \
  --output-dir results \
  --cache-dir runs/linear_probing/cache \
  --position last
```

---

## 7. SAE Steering Experiments

`src/methods/steering/experiments.py` implements Sparse Autoencoder (SAE) based bias suppression experiments. This method:

1. Trains SAEs on model activations to learn sparse latent representations
2. Identifies bias-correlated latents through correlation analysis
3. Suppresses identified latents to reduce bias while maintaining model performance
4. Evaluates on bias metrics, perplexity, and MMLU accuracy

### Run it

```bash
python -m src.methods.steering.experiments \
  --model gpt2-medium \
  --dataset stereoset_gender \
  --layer 0 \
  --k-sparse 64 \
  --n-examples 2000 \
  --output-dir results
```

Outputs:
- `sae_bias_suppression_k<k>.json`: Results for each k-sparse value
- `sae_bias_suppression_summary.json`: Aggregated summary
- `activation_profiles/`: Activation profile data
- `latent_profiles/`: Latent profile visualizations

---