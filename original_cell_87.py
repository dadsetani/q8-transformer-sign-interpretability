
# ============================================================
# EXPERIMENT 24 — Paired Bootstrap and Full-State Verification
# Reuses Experiment 23 results; no model forward required
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

EXP24_DIR = Path("/kaggle/working/q8_diagnostic24")
EXP24_DIR.mkdir(parents=True, exist_ok=True)

pairs = np.asarray(exp03_valid_pairs)
q_true = np.asarray(exp03_valid["quotient_state"])
s_true = np.asarray(exp03_valid["nuisance"])

assert pairs.shape == (512, 2)

conditions = [
    "baseline",
    "attention_ablation",
    "own_mlp_restore",
    "opposite_sign_mlp_restore"
]

# ------------------------------------------------------------
# 1. Verify the Experiment 23 data
# ------------------------------------------------------------

for condition in conditions:
    data = results23[condition]
    assert data["prob"].shape == (1024, 13)
    assert data["sign_pred"].shape == (1024, 13)
    assert data["q_pred"].shape == (1024, 13)

# ------------------------------------------------------------
# 2. Full-state correctness
# ------------------------------------------------------------

full_correct = {}

for condition in conditions:
    data = results23[condition]

    quotient_correct = data["q_pred"] == q_true
    conditional_sign_correct = data["sign_pred"] == s_true

    full_correct[condition] = (
        quotient_correct & conditional_sign_correct
    )

full_rows = []

for position in range(13):
    row = {"position": position}

    for condition in conditions:
        row[condition] = float(
            full_correct[condition][:, position].mean()
        )

    full_rows.append(row)

exp24_full_df = pd.DataFrame(full_rows)

print("FULL-STATE ACCURACY")
display(exp24_full_df.round(4))

# ------------------------------------------------------------
# 3. Pair-level contrast for probability
# ------------------------------------------------------------

own_prob = results23["own_mlp_restore"]["prob"]
opp_prob = results23["opposite_sign_mlp_restore"]["prob"]

trajectory_difference = own_prob - opp_prob

# Each pair is the bootstrap unit.
pair_difference = trajectory_difference[pairs].mean(axis=1)

assert pair_difference.shape == (512, 13)

# ------------------------------------------------------------
# 4. Bootstrap confidence intervals
# ------------------------------------------------------------

rng = np.random.default_rng(20261119)

n_pairs = len(pairs)
n_boot = 5000

bootstrap_indices = rng.integers(
    low=0,
    high=n_pairs,
    size=(n_boot, n_pairs)
)

bootstrap_means = pair_difference[
    bootstrap_indices
].mean(axis=1)

observed_means = pair_difference.mean(axis=0)

ci_low = np.percentile(
    bootstrap_means, 2.5, axis=0
)
ci_high = np.percentile(
    bootstrap_means, 97.5, axis=0
)

bootstrap_rows = []

for position in range(13):
    bootstrap_rows.append({
        "position": position,
        "own_minus_opposite_mean": float(
            observed_means[position]
        ),
        "ci95_low": float(ci_low[position]),
        "ci95_high": float(ci_high[position])
    })

exp24_bootstrap_df = pd.DataFrame(bootstrap_rows)

print("\nPAIR-LEVEL BOOTSTRAP: SIGN PROBABILITY CONTRAST")
display(exp24_bootstrap_df.round(5))

# ------------------------------------------------------------
# 5. Descriptive early-position summary
# ------------------------------------------------------------

early_positions = np.arange(1, 6)

early_pair_effect = pair_difference[
    :, early_positions
].mean(axis=1)

early_boot = early_pair_effect[
    bootstrap_indices
].mean(axis=1)

print("\nEARLY POSITIONS 1–5")
print(
    "Mean own-minus-opposite effect:",
    round(float(early_pair_effect.mean()), 6)
)
print(
    "95% pair-bootstrap CI:",
    np.percentile(early_boot, [2.5, 97.5]).round(6)
)

# ------------------------------------------------------------
# 6. Full-state accuracy contrast with paired uncertainty
# ------------------------------------------------------------

full_own = full_correct["own_mlp_restore"].astype(float)
full_opp = full_correct[
    "opposite_sign_mlp_restore"
].astype(float)

full_pair_difference = (
    full_own[pairs] - full_opp[pairs]
).mean(axis=1)

full_bootstrap_means = full_pair_difference[
    bootstrap_indices
].mean(axis=1)

full_contrast_rows = []

for position in range(13):
    full_contrast_rows.append({
        "position": position,
        "full_accuracy_difference": float(
            full_pair_difference[:, position].mean()
        ),
        "ci95_low": float(
            np.percentile(
                full_bootstrap_means[:, position], 2.5
            )
        ),
        "ci95_high": float(
            np.percentile(
                full_bootstrap_means[:, position], 97.5
            )
        )
    })

exp24_full_contrast = pd.DataFrame(full_contrast_rows)

print("\nFULL-STATE ACCURACY: OWN MINUS OPPOSITE")
display(exp24_full_contrast.round(5))

# ------------------------------------------------------------
# 7. Save
# ------------------------------------------------------------

exp24_full_df.to_csv(
    EXP24_DIR / "full_state_accuracy.csv",
    index=False
)

exp24_bootstrap_df.to_csv(
    EXP24_DIR / "sign_probability_bootstrap.csv",
    index=False
)

exp24_full_contrast.to_csv(
    EXP24_DIR / "full_state_accuracy_contrast.csv",
    index=False
)

print("\nExperiment 24 completed.")
print("Bootstrap unit: 512 validation pairs")
print("Test split: NOT ACCESSED")
