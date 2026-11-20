# Additional robustness controls for the Q8 Transformer manuscript

**Purpose:** Add-on for the existing `dadsetani/q8-transformer-sign-interpretability` GitHub repository. These controls are exploratory and do not use the sealed test split.

## Upload instructions

Upload the **`robustness_checks/` folder** into the root of the existing repository. Do not upload the ZIP itself. The original repository root must already contain `model_extracted.py`, `checkpoint_and_config.zip`, and `validation_data.zip`. Scripts import the model from the repository root and read the existing ZIP archives there. They write validation-control results inside `robustness_checks/`.

Run from the repository root:

```bash
python robustness_checks/run_matched_donor_control.py
python robustness_checks/run_factorial_donor_control.py
python robustness_checks/run_permutation_control.py
```

**Verification:** Scripts passed Python syntax checks. Execution against the full original repository, its archives, and checkpoints has **not** been verified in this package. Do not claim full end-to-end reproducibility until those runs succeed.

## Included controls and outputs

| Script | Existing result | Description |
|---|---|---|
| `run_matched_donor_control.py` | `matched_donor_control_results.json`, `matched_donor_control_arrays.npz` | Same quotient/sign independent donor and norm-matched random vector controls |
| `run_factorial_donor_control.py` | `factorial_donor_control_results.json`, `factorial_donor_control_arrays.npz` | Donor matching controls varying sign and quotient |
| `run_permutation_control.py` | `permutation_control_results.json` | Five action-order permutations with recomputed labels |
| `run_loss_weight_control.py` | **Not included: full 200-epoch Kaggle outputs** | Original versus symmetric objective retraining |

The first three result files were produced using the original conditional-head model checkpoint, seed 11, epoch 197, and validation trajectories (512 matched pairs / 1024 trajectories). They are not multi-seed replication studies. The full training-loss control was run separately in Kaggle using the original training archive, and its complete outputs have **not** been supplied to this package. Do not claim the GitHub repository reproduces the published loss-weight curves until the actual full-run files have been uploaded.

## Run the validation-only controls

Requires Python, NumPy and PyTorch. From the repository root:

```bash
python robustness_checks/run_matched_donor_control.py
python robustness_checks/run_factorial_donor_control.py
python robustness_checks/run_permutation_control.py
```

These scripts overwrite their own named JSON/NPZ result files on rerun; use a clean checkout if preserving prior output is necessary.

## Loss-weight sensitivity: complete the evidence record

`run_loss_weight_control.py` requires the **original** `q8_pilot_backup_20261005_102000.zip` training archive, which is intentionally **not** bundled here. It verifies the original archive and the train/validation split hashes and does not open `test.npz`.

```bash
python robustness_checks/run_loss_weight_control.py --archive /path/to/q8_pilot_backup_20261005_102000.zip --epochs 200 --out loss_weight_control_results
```

For the completed Kaggle run, upload these files from `/kaggle/working/q8_loss_control_results/` into a GitHub folder named `loss_weight_control_results/`:

- `config.json`
- `original_history.json`
- `symmetric_history.json`
- `original_best.pt`
- `symmetric_best.pt`

**Do not upload the original training archive or sealed test split without first reviewing distribution rights and repository size.** For readers without the training archive, the histories and checkpoints document the completed run but do not alone enable retraining from scratch.

Reported from the Kaggle user's logs (not independently re-evaluated from the checkpoint files in this package): original minimum validation full NLL at epoch 200 = 0.380321; symmetric minimum validation full NLL at epoch 109 = 0.476030. Conditional sign accuracy at position 6 was 80.47% versus 52.34%; final-position quotient accuracy was 99.32% versus 99.22%. These are single-seed exploratory validation findings.

## Scope and caveats

- The control interventions use full MLP activation vectors, not a uniquely isolated sign circuit.
- Donor changes may introduce out-of-distribution activation combinations.
- Permutation invariance of performance does not identify a specific learned shortcut.
- Loss-weight training used one seed per objective; hardware/library versions may affect exact retraining reproducibility.
- The original checkpoint (epoch 197) and the retrained original-objective checkpoint (epoch 200) are **not the same artifact**.
- No sealed test results are claimed.

After publishing an updated Zenodo version, cite its **version-specific DOI** in the final article and submission form if exact version reproducibility is the goal. Keep the GitHub link as a convenient living repository reference.
