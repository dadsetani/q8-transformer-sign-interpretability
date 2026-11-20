
# ============================================================
# EXPERIMENT 23 — Sign-Specific MLP Restoration
# Validation only; paired replacement
# ============================================================

import numpy as np
import pandas as pd
import torch
from pathlib import Path

EXP23_DIR = Path("/kaggle/working/q8_diagnostic23")
EXP23_DIR.mkdir(parents=True, exist_ok=True)

model = exp03_model
model.eval()
block = model.blocks[2]

tokens_np = np.asarray(exp03_valid["tokens"])
q_np = np.asarray(exp03_valid["quotient_state"])
s_np = np.asarray(exp03_valid["nuisance"])

pairs = np.asarray(exp03_valid_pairs)

n, seq_len = tokens_np.shape
assert n == 1024 and seq_len == 13
assert pairs.shape == (512, 2)

# Map each validation trajectory to its paired counterpart.
partner = np.empty(n, dtype=np.int64)

partner[pairs[:, 0]] = pairs[:, 1]
partner[pairs[:, 1]] = pairs[:, 0]

assert np.array_equal(partner[partner], np.arange(n))
assert np.array_equal(q_np, q_np[partner])
assert np.all(s_np != s_np[partner])

def zero_output(module, inputs, output):
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]), *output[1:])
    return torch.zeros_like(output)

def metrics(outputs, q_true, s_true):

    logits = outputs["sign_logits"]

    selected = logits.gather(
        2,
        q_true[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)

    prob = selected.softmax(-1).gather(
        -1, s_true[..., None]
    ).squeeze(-1)

    return (
        prob.cpu().numpy(),
        selected.argmax(-1).cpu().numpy(),
        outputs["quotient_logits"].argmax(-1).cpu().numpy()
    )

conditions = [
    "baseline",
    "attention_ablation",
    "own_mlp_restore",
    "opposite_sign_mlp_restore"
]

collected = {
    condition: {"prob": [], "sign_pred": [], "q_pred": []}
    for condition in conditions
}

# Process complete pairs together to preserve partner mapping.
batch_pairs = 64

with torch.inference_mode():

    for start in range(0, len(pairs), batch_pairs):

        selected_pairs = pairs[start:start + batch_pairs]
        indices = selected_pairs.reshape(-1)

        x = torch.as_tensor(
            tokens_np[indices],
            dtype=torch.long,
            device=exp03_device
        )

        q_true = torch.as_tensor(
            q_np[indices],
            dtype=torch.long,
            device=exp03_device
        )

        s_true = torch.as_tensor(
            s_np[indices],
            dtype=torch.long,
            device=exp03_device
        )

        saved = []

        def save_mlp(module, inputs, output):
            saved.append(output.detach().clone())

        h = block.feedforward.register_forward_hook(save_mlp)

        try:
            baseline_outputs = model(x)
        finally:
            h.remove()

        assert len(saved) == 1

        own_mlp = saved[0]

        # Pair rows are consecutive:
        # [pair0_a, pair0_b, pair1_a, pair1_b, ...]
        opposite_mlp = own_mlp.reshape(
            -1, 2, seq_len, own_mlp.shape[-1]
        ).flip(1).reshape_as(own_mlp)

        def run_intervention(restore_tensor=None):

            handles = [
                block.attention.register_forward_hook(
                    zero_output
                )
            ]

            if restore_tensor is not None:

                def restore_hook(module, inputs, output):
                    assert output.shape == restore_tensor.shape
                    return restore_tensor

                handles.append(
                    block.feedforward.register_forward_hook(
                        restore_hook
                    )
                )

            try:
                return model(x)
            finally:
                for handle in reversed(handles):
                    handle.remove()

        output_map = {
            "baseline": baseline_outputs,
            "attention_ablation": run_intervention(),
            "own_mlp_restore": run_intervention(own_mlp),
            "opposite_sign_mlp_restore": run_intervention(
                opposite_mlp
            )
        }

        for condition, outputs in output_map.items():

            prob, sign_pred, q_pred = metrics(
                outputs, q_true, s_true
            )

            collected[condition]["prob"].append(prob)
            collected[condition]["sign_pred"].append(sign_pred)
            collected[condition]["q_pred"].append(q_pred)

results23 = {
    condition: {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in data.items()
    }
    for condition, data in collected.items()
}

# ------------------------------------------------------------
# Consistency checks against Experiment 22
# ------------------------------------------------------------

for condition23, condition22 in [
    ("baseline", "baseline"),
    ("attention_ablation", "attention_ablation"),
    ("own_mlp_restore", "attention_ablation_mlp_restore")
]:
    assert np.allclose(
        results23[condition23]["prob"],
        results22[condition22]["prob"],
        atol=1e-5
    ), f"Consistency check failed: {condition23}"

# ------------------------------------------------------------
# Summary tables
# ------------------------------------------------------------

rows = []

for condition in conditions:

    data = results23[condition]

    for position in range(seq_len):

        rows.append({
            "condition": condition,
            "position": position,

            "mean_true_sign_probability": float(
                data["prob"][:, position].mean()
            ),

            "conditional_sign_accuracy": float(
                np.mean(
                    data["sign_pred"][:, position]
                    == s_np[:, position]
                )
            ),

            "quotient_accuracy": float(
                np.mean(
                    data["q_pred"][:, position]
                    == q_np[:, position]
                )
            )
        })

exp23_df = pd.DataFrame(rows)

exp23_df.to_csv(
    EXP23_DIR / "sign_specific_restoration.csv",
    index=False
)

print("\nTRUE SIGN PROBABILITY")

display(
    exp23_df.pivot(
        index="position",
        columns="condition",
        values="mean_true_sign_probability"
    )[conditions].round(4)
)

print("\nCONDITIONAL SIGN ACCURACY")

display(
    exp23_df.pivot(
        index="position",
        columns="condition",
        values="conditional_sign_accuracy"
    )[conditions].round(4)
)

print("\nQUOTIENT ACCURACY")

display(
    exp23_df.pivot(
        index="position",
        columns="condition",
        values="quotient_accuracy"
    )[conditions].round(4)
)

# ------------------------------------------------------------
# Own vs opposite-sign restoration contrast
# ------------------------------------------------------------

own = results23["own_mlp_restore"]["prob"]
opposite = results23["opposite_sign_mlp_restore"]["prob"]

contrast_rows = []

for position in range(seq_len):

    difference = own[:, position] - opposite[:, position]

    contrast_rows.append({
        "position": position,
        "mean_own_minus_opposite": float(
            difference.mean()
        ),
        "fraction_own_better": float(
            np.mean(difference > 0)
        )
    })

exp23_contrast = pd.DataFrame(contrast_rows)

exp23_contrast.to_csv(
    EXP23_DIR / "own_vs_opposite_contrast.csv",
    index=False
)

print("\nOWN VS OPPOSITE-SIGN RESTORATION")

display(exp23_contrast.round(4))

print("\nExperiment 23 completed.")
print("Test split: NOT ACCESSED")
