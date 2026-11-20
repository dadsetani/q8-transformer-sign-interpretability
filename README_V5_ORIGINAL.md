# Q8 Transformer: checkpoint evaluation reproducibility bundle

Scope: **one checkpoint, model seed 11, epoch 197, conditional head**. This bundle is intended to reproduce the held-out *validation* metrics reported for the position-dependent quotient/sign dissociation. It is **not** a full independent retraining package and does **not** rerun causal interventions.

## Files
- `verify.py`: loads the exact recovered notebook model classes, checks the validation data hash, loads the epoch-197 weights with strict state-dict matching, and recomputes loss and per-position accuracies.
- `source_notebook.ipynb`: recovered original notebook used to extract model classes (cell 16); includes other historical experiments and may contain unrelated exploratory code.
- `model_extracted.py`: the two model classes extracted verbatim from that notebook, for inspection.
- `checkpoint_and_config.zip`: `conditional/best.pt`, `conditional/run_config.json`, `conditional/summary.json`; excludes dual-head and other checkpoints.
- `validation_data.zip`: the original `q8_pilot_v1/validation.npz` only; training and sealed test data are not distributed in this minimal evaluation bundle.
- `original_audit.json`: previously recomputed evaluation report.
- `manifest_sha256.json`: checksums of bundled files.

## Run

```bash
python -m pip install numpy torch
python verify.py
```

Run in the directory containing these files. The script writes `recomputed_audit.json`. A CPU run is supported; small floating-point differences in loss are expected across hardware or PyTorch versions. The source runtime reported PyTorch `2.11.0+cu128`, CUDA `12.8`, Tesla T4, and NumPy `2.1.3` in the associated audit; the exact original environment has not been fully reconstructed.

## Scientific and provenance cautions
- `run_config.json` declares model seed 11, shuffle seed 1011, batch size 256, 200 planned epochs, AdamW-style parameters, and `test_evaluated: false`. The checkpoint selected for this article is epoch 197.
- Position 0 was **excluded from the training objective**; report action positions 1–12 as the primary evaluation window.
- The reported validation results are exploratory, not an independent sealed-test confirmation.
- The original training source archive hash is recorded in `run_config.json`, but this minimal bundle does not establish that the recovered notebook reproduces training from initialization bit-for-bit.
- The full-vector causal patching and bootstrap analyses must be independently packaged and verified before claiming end-to-end mechanistic reproducibility.
- Do not interpret the quotient/sign dissociation as proof of information erasure or a unique MLP circuit.

## Data handling
This package intentionally omits `test.npz`, training data, optimizer states, and unrelated exploratory archives. Confirm data rights and repository licensing before public release. Checkpoint loading uses PyTorch's `weights_only=True` and requires trusted input files.

## Addendum v2: saved-activation geometry audit
Run `python verify_geometry.py` to independently recompute the mean Euclidean distance between 512 matched opposite-sign validation trajectories at each position from the included `validation_activations.npz`. This uses *saved activations*, not a fresh forward-hook capture; it does not reproduce the sign-direction projections, MLP decomposition, bootstrap or interventions. These remain pending. The original v1 `verify.py` remains unchanged. The sealed test split is excluded.

## Addendum v3: exact final-block decomposition (newly recomputed)
Run `python verify_decomposition.py`. It executes the recovered checkpoint on the validation tokens, explicitly computes `pre`, `attention`, `mlp`, and `output` in the final Transformer block, and verifies `output = pre + attention + mlp` against the original block forward pass. This audit gives exact numerical agreement (maximum observed error 0.0 on CPU). It computes **exploratory** paired differences using a clearly stated *new* direction definition (normalized mean output difference per position). **Do not compare the directional projections numerically with those in the article:** the article's sign-associated direction is defined differently and has not been reconstructed in this audit. The causal restoration/ablation experiment, bootstrap CIs, and original sign-direction calculation remain unverified. The saved-activation geometry in v2 is also a different representation (likely normalized hidden state) from the raw pre-final-norm residual stream in v3.

## Addendum v4: recovered original conditional sign-head direction
Run `python verify_original_sign_direction.py` to recompute Experiments 14/19 style sign-head alignment from the frozen epoch-197 model and 512 matched validation pairs. The quotient-specific direction is `W[q,1]-W[q,0]` from the trained conditional sign head; pair differences are oriented at each position as `h(sign=1)-h(sign=0)`. The script reports both the **mean per-pair parallel-energy fraction** and the **ratio of summed energies**, which are different statistics. Recomputed final-normalized signed cosine at positions 1/4/12: 0.87905/0.72685/0.00682; mean per-pair parallel fraction: 0.77489/0.55471/0.01836. Recomputed MLP signed projections at positions 1/4/12: 28.4653/7.6603/0.0643. The maximum discrepancy from saved final-normalized validation activations is approximately 1.43e-5. This is a descriptive readout-direction audit, not a causal intervention. The sealed test split remains unopened.

## V5: Independent replay of Experiments 23–24

Run `python verify_interventions.py` from this directory (requires PyTorch and NumPy).
The script loads the conditional checkpoint at epoch 197, reads only validation
trajectories, and replays final-block attention-output zeroing with the MLP
output restored from either the target trajectory or its paired opposite-sign
trajectory. It computes true-quotient-conditioned sign probability, quotient
and full-state accuracy, and 5,000 pair-level bootstrap replicates with
NumPy seed 20261119. It saves `intervention_audit.json` and
`intervention_bootstrap_arrays.npz`. The sealed test set is not opened.

This verifies the particular full-vector intervention in a single checkpoint.
It does not prove an independent or minimal MLP circuit, sign-information
erasure, generalization across training seeds, or full training reproducibility.
Original notebook cell excerpts 83, 85 and 87 are included for traceability.
