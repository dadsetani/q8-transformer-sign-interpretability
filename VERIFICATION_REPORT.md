# Verification report — 2026-11-20

Executed in the available Python environment against the included frozen epoch-197 checkpoint and validation data:

| Script | Result |
|---|---|
| `verify_source_manifest.py` | PASS: all preserved original V5 files matched their SHA-256 hashes |
| `verify.py` | PASS (exit 0) |
| `verify_geometry.py` | PASS (exit 0); position-12 paired distance 2.5290613 |
| `verify_decomposition.py` | PASS (exit 0); residual reconstruction error 0.0 |
| `verify_original_sign_direction.py` | PASS (exit 0); position-12 mean parallel-energy fraction 0.01836 |
| `verify_interventions.py` | PASS (exit 0); positions 1–5 mean restoration contrast 0.8840724, 95% paired bootstrap CI [0.8756388, 0.8922297] |

These checks are for stored-checkpoint validation replay, not full training-from-scratch reproducibility, independent held-out test confirmation, or independent code review. The recovered notebook should be reviewed manually before publication. No GitHub repository or Zenodo deposit was created by this packaging step.
