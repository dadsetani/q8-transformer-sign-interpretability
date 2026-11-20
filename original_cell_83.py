
# ============================================================
# EXPERIMENT 22 — Attention Ablation + MLP Restoration
# Validation only, batched, no retraining
# ============================================================

import numpy as np
import pandas as pd
import torch
from pathlib import Path

EXP22_DIR = Path("/kaggle/working/q8_diagnostic22")
EXP22_DIR.mkdir(parents=True, exist_ok=True)

model = exp03_model
model.eval()
block = model.blocks[2]

tokens_np = np.asarray(exp03_valid["tokens"])
q_np = np.asarray(exp03_valid["quotient_state"])
s_np = np.asarray(exp03_valid["nuisance"])

assert tokens_np.shape == q_np.shape == s_np.shape
assert tokens_np.shape == (1024, 13)

def zero_output(module, inputs, output):
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]), *output[1:])
    return torch.zeros_like(output)

def evaluate_logits(outputs, q_true, s_true):
    logits = outputs["sign_logits"]
    selected = logits.gather(
        2, q_true[..., None, None].expand(-1, -1, 1, 2)
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
    "mlp_ablation",
    "attention_ablation_mlp_restore"
]

collected = {
    name: {"prob": [], "sign_pred": [], "q_pred": []}
    for name in conditions
}

with torch.inference_mode():
    for start in range(0, len(tokens_np), 128):
        end = min(start + 128, len(tokens_np))

        x = torch.as_tensor(
            tokens_np[start:end],
            dtype=torch.long,
            device=exp03_device
        )
        q_true = torch.as_tensor(
            q_np[start:end],
            dtype=torch.long,
            device=exp03_device
        )
        s_true = torch.as_tensor(
            s_np[start:end],
            dtype=torch.long,
            device=exp03_device
        )

        baseline_mlp = []

        def save_mlp(module, inputs, output):
            baseline_mlp.append(output.detach().clone())

        handle = block.feedforward.register_forward_hook(save_mlp)
        try:
            baseline_output = model(x)
        finally:
            handle.remove()

        assert len(baseline_mlp) == 1
        original_mlp = baseline_mlp[0]

        outputs_by_condition = {
            "baseline": baseline_output
        }

        # Attention zeroed; MLP recomputed normally.
        handle = block.attention.register_forward_hook(zero_output)
        try:
            outputs_by_condition["attention_ablation"] = model(x)
        finally:
            handle.remove()

        # MLP zeroed; Attention remains natural.
        handle = block.feedforward.register_forward_hook(zero_output)
        try:
            outputs_by_condition["mlp_ablation"] = model(x)
        finally:
            handle.remove()

        # Attention zeroed; baseline MLP output restored.
        def restore_mlp(module, inputs, output):
            assert output.shape == original_mlp.shape
            return original_mlp

        h_attn = block.attention.register_forward_hook(zero_output)
        h_mlp = block.feedforward.register_forward_hook(restore_mlp)

        try:
            outputs_by_condition[
                "attention_ablation_mlp_restore"
            ] = model(x)
        finally:
            h_mlp.remove()
            h_attn.remove()

        for condition, outputs in outputs_by_condition.items():
            prob, sign_pred, q_pred = evaluate_logits(
                outputs, q_true, s_true
            )
            collected[condition]["prob"].append(prob)
            collected[condition]["sign_pred"].append(sign_pred)
            collected[condition]["q_pred"].append(q_pred)

results22 = {
    name: {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in values.items()
    }
    for name, values in collected.items()
}

# Verify baseline against Experiment 21.
for key, previous_key in [
    ("prob", "true_sign_prob"),
    ("sign_pred", "sign_pred"),
    ("q_pred", "quotient_pred")
]:
    assert np.allclose(
        results22["baseline"][key],
        results["baseline"][previous_key],
        atol=1e-5
    ), f"Baseline mismatch: {key}"

rows = []
for condition in conditions:
    data = results22[condition]

    for pos in range(13):
        rows.append({
            "condition": condition,
            "position": pos,
            "mean_true_sign_prob": float(
                data["prob"][:, pos].mean()
            ),
            "sign_accuracy": float(
                np.mean(data["sign_pred"][:, pos] == s_np[:, pos])
            ),
            "quotient_accuracy": float(
                np.mean(data["q_pred"][:, pos] == q_np[:, pos])
            )
        })

exp22_df = pd.DataFrame(rows)
exp22_df.to_csv(
    EXP22_DIR / "restoration_results.csv",
    index=False
)

print("TRUE SIGN PROBABILITY")
display(
    exp22_df.pivot(
        index="position",
        columns="condition",
        values="mean_true_sign_prob"
    )[conditions].round(4)
)

print("QUOTIENT ACCURACY — SELECTED POSITIONS")
display(
    exp22_df[
        exp22_df.position.isin([1, 4, 6, 8, 12])
    ][["condition", "position", "quotient_accuracy"]].round(4)
)

# Paired, per-trajectory restoration improvement.
restore = results22[
    "attention_ablation_mlp_restore"
]["prob"]

ablated = results22["attention_ablation"]["prob"]
baseline = results22["baseline"]["prob"]

recovery_rows = []
for pos in range(13):
    recovery_rows.append({
        "position": pos,
        "mean_restoration_gain": float(
            (restore[:, pos] - ablated[:, pos]).mean()
        ),
        "mean_remaining_baseline_gap": float(
            (baseline[:, pos] - restore[:, pos]).mean()
        )
    })

exp22_recovery = pd.DataFrame(recovery_rows)
exp22_recovery.to_csv(
    EXP22_DIR / "restoration_gain.csv",
    index=False
)

print("RESTORATION GAIN")
display(exp22_recovery.round(4))

print("Experiment 22 completed.")
print("Test split: NOT ACCESSED")
