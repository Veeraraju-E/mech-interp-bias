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
│   ├── visualization.py            # Plot factory for summaries/heatmaps
│   └── methods/
│       ├── activation_patching/
│       │   ├── ablations.py        # Head/MLP ablation scans
│       │   ├── attribution_patching.py
│       │   ├── experiments.py      # End-to-end activation patching pipeline
│       │   └── hook_points.py
│       └── linear_probing/
│           └── probe.py            # Layer-wise logistic probes + CLI
├── runs/                           # Cache directories (created on demand)
└── results/                        # JSON + figure outputs (per model)
```

All experimental entry points live inside `src/methods`, which keeps datasets, metrics, and device utilities reusable across analyses.

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

Additional experiment code expects optional preprocessed ACDC pairs:

```
data/stereoset_gender_acdc_pairs.json
data/stereoset_race_acdc_pairs.json
```

These files should contain the `{"clean": {"tokens": [...], "metadata": {...}}, ...}` format described in `src/data_loader.load_stereoset_acdc_pairs`.

---

## 4. Baseline Metrics

`src/bias_metrics.py` provides reproducible scores used across all methods:

- `compute_stereoset_score`: log-probability gap between stereotype vs. antistereotype tokens.
- `compute_winogender_score`: log-probability gap between male vs. female pronouns in paired templates.
- `build_bias_metric_fn`: closure that maps logits + metadata → differentiable scalar; this powers attribution patching and ablation comparisons.
---

## 5. Activation Patching Pipeline

The main orchestration lives in `src/methods/activation_patching/experiments.py`. It:

1. Loads the requested model + tokenizer (`--model {gpt2-medium,gpt2-large,gpt-neo-125M}`).
2. Loads datasets + (optionally) cached baselines from `runs/activation_patching/cache`.
3. Prepares prompts for each dataset (`StereoSet` race/gender ACDC pairs, `WinoGender` pronoun contexts).
4. Builds a dataset-specific bias metric closure.
5. Runs:
   - **Attribution patching** (`attribution_patching.py`): gradient-based edge scoring with hook names such as `blocks.5.attn.hook_z`.
   - **Head/MLP ablation scans** (`ablations.py`): zeroes each head or MLP output and measures Δbias.
6. Persists JSON artifacts in `results/<model>/`.

### Run it

```bash
python -m src.methods.activation_patching.experiments \
  --model gpt2-medium \
  --output-dir results \
  --cache-dir runs/activation_patching/cache
```

Useful flags:
- `--no-cache`: recompute everything (ignores disk cache).
- `--output-dir`: change the root results directory (defaults to `results`).
- `--cache-dir`: pick a different cache root.

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

Pipeline highlights:

- Builds balanced biased/neutral prompts (`prepare_biased_neutral_pairs`) for each dataset.
- Captures residual activations for every layer (including layer 0) via TransformerLens hooks.
- Trains `sklearn` logistic regressions with an 80/20 split per layer.
- Saves per-layer accuracy/AUC to `results/<model>/linear_probing_<dataset>.json` and an aggregated `linear_probing_summary.json`.
- Caches activations in `.npz` files so subsequent runs avoid re-encoding prompts.

Interpretation tip: higher probe accuracy at later layers usually signals that bias representations sharpen as the model processes the context, aligning with the hypotheses in `requirements.md`.

---

## 7. Contributing / Extending

- Add new datasets by extending `src/data_loader` (normalize JSON, supply prompt builders, wire metadata into `build_bias_metric_fn`).
- To analyze additional model families, extend `model_setup.load_model` / `get_tokenizer` to include validated checkpoints.
- When adding new experiments, keep caching + visualization hooks consistent so the reporting pipeline stays uniform.

For questions or reproducibility issues, open an issue with your environment details, the exact command, and relevant log excerpts.