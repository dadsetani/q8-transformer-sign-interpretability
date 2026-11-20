# Q8 Transformer: Position-Dependent Sign Readout and MLP-Mediated Control

Reproducibility materials for an exploratory mechanistic analysis of a **three-block causal Transformer** trained on quaternion-group (`Q8`) trajectories. This repository focuses on **one conditional-head checkpoint (model seed 11, epoch 197)** and **512 matched validation pairs**. It does not establish cross-seed generalization or full retraining reproducibility.

## What is included

| Analysis | Command | Main output |
| --- | --- | --- |
| Baseline validation accuracy and checkpoint verification | `python verify.py` | `recomputed_audit.json` |
| Paired representation distances | `python verify_geometry.py` | geometry audit |
| Exact final-block residual decomposition | `python verify_decomposition.py` | decomposition audit |
| Quotient-specific sign-head alignment and projections | `python verify_original_sign_direction.py` | sign-direction audit |
| Attention ablation, own/opposite MLP restoration and 5,000 paired bootstrap replicates | `python verify_interventions.py` | intervention audit and bootstrap arrays |

Scripts use the frozen checkpoint and validation data in the included ZIP archives. They do not open the reserved test split.

## Quick start

Python 3.10+ is recommended for trying the scripts (the recorded source runtime used Python 3.13.15). Install compatible PyTorch and NumPy versions for your platform:

```bash
python -m pip install -r requirements.txt
python verify.py
python verify_geometry.py
python verify_decomposition.py
python verify_original_sign_direction.py
python verify_interventions.py
```

On CPU, small numerical differences in floating-point quantities are possible. The recorded environment used a Tesla T4, PyTorch 2.11.0+cu128 and NumPy 2.1.3. The original environment is not guaranteed to be reproduced bit-for-bit.

## Data and provenance

- `checkpoint_and_config.zip`: conditional-head epoch-197 weights and configuration.
- `validation_data.zip`: validation trajectories only, paired by `pair_id`.
- `validation_activations.npz`: saved validation activations for geometry verification.
- `source_notebook.ipynb`: recovered historical source notebook, retained for provenance; it contains exploratory material beyond the reported analyses.
- `model_extracted.py`: model definitions recovered from that notebook.
- `original_cell_83.py`, `original_cell_85.py`, `original_cell_87.py`: preserved original notebook cell excerpts for intervention traceability.
- `*_audit.json` and `intervention_bootstrap_arrays.npz`: recorded audit artifacts.
- `manifest_sha256.json`: original V5 bundle hashes. **Note:** `README.md` was replaced with this repository-facing guide; the original is preserved as `README_V5_ORIGINAL.md`. Use `verify_source_manifest.py` to check the remaining original files.

## Scientific scope

- Quotient prediction remains near-perfect on action positions 1–12; conditional sign prediction weakens with position.
- The final-block MLP output shows a prominent early-position projection onto the trained sign-head direction.
- Under final-block attention ablation, restoring the target MLP output versus its matched opposite-sign counterpart changes correct-sign probability substantially at early positions. The reported mean contrast across positions 1–5 is approximately **0.8841**, with a **95% paired-bootstrap interval [0.8756, 0.8922]**.
- The intervention replaces a **whole activation vector**. It does **not** identify a unique or independent sign circuit, demonstrate erasure of late-position sign information, or recover baseline full-state accuracy in all conditions.
- Bootstrap intervals are conditional on the checkpoint and exploratory validation sample. The reserved test split has not been evaluated in this package.

## Important limitations

This is a **checkpoint-evaluation and intervention-replay package**, not an end-to-end training pipeline. The recovered notebook is provided for traceability, not as a validated clean training script. Do not claim independent test confirmation, training-from-scratch reproducibility, or cross-seed robustness based on this repository alone.

See [`README_V5_ORIGINAL.md`](README_V5_ORIGINAL.md) for the detailed provenance notes of the recovered V5 artifact.

## Authors and affiliation

1. **M.K. Dadsetani** — Department of Computer, Faculty of Engineering, Shahed University, Tehran, Iran. Email: m.dadsetani@shahed.ac.ir
2. **Hamid H.S. Javadi** (**corresponding author**) — Department of Computer, Faculty of Engineering, Shahed University, Tehran, Iran. Email: h.s.javadi@shahed.ac.ir

## License and citation

Citation metadata is supplied in [`CITATION.cff`](CITATION.cff) using the manuscript title and author details. The manuscript is not yet assigned a publication DOI. No software license is assigned yet; select a license before public release. No Zenodo DOI or public repository URL is asserted here.
