import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_CONFIG = {'vocab_size': 12, 'd_model': 64, 'n_heads': 4, 'n_layers': 3, 'd_ff': 128, 'max_tokens': 65, 'dropout': 0.0}

class CausalBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, hidden, attention_mask):
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.feedforward(
            self.feedforward_norm(hidden)
        )
        return hidden

class HeadComparisonQ8Transformer(nn.Module):
    def __init__(self, head_type):
        super().__init__()
        if head_type not in ("dual", "conditional"):
            raise ValueError(f"Unknown head type: {head_type}")

        d_model = MODEL_CONFIG["d_model"]
        self.head_type = head_type
        self.token_embedding = nn.Embedding(
            MODEL_CONFIG["vocab_size"], d_model
        )

        positions = torch.arange(
            MODEL_CONFIG["max_tokens"], dtype=torch.float32
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        positional_encoding = torch.zeros(
            MODEL_CONFIG["max_tokens"], d_model
        )
        positional_encoding[:, 0::2] = torch.sin(positions * frequencies)
        positional_encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("positional_encoding", positional_encoding)

        self.blocks = nn.ModuleList([
            CausalBlock(
                d_model,
                MODEL_CONFIG["n_heads"],
                MODEL_CONFIG["d_ff"],
                MODEL_CONFIG["dropout"],
            )
            for _ in range(MODEL_CONFIG["n_layers"])
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Both designs have 12 output values per hidden unit.
        self.quotient_head = nn.Linear(d_model, 4)
        if head_type == "dual":
            self.full_head = nn.Linear(d_model, 8)
        else:
            self.conditional_sign_head = nn.Linear(d_model, 8)

    def forward(self, tokens):
        sequence_length = tokens.shape[1]
        if sequence_length > self.positional_encoding.shape[0]:
            raise ValueError("Sequence is longer than configured maximum.")

        hidden = (
            self.token_embedding(tokens)
            + self.positional_encoding[:sequence_length].unsqueeze(0)
        )

        attention_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            hidden = block(hidden, attention_mask)

        hidden = self.final_norm(hidden)
        quotient_logits = self.quotient_head(hidden)

        if self.head_type == "dual":
            full_scores = self.full_head(hidden)
            sign_logits = None
        else:
            # P(full state) = P(quotient) * P(sign | quotient)
            sign_logits = self.conditional_sign_head(hidden).reshape(
                *hidden.shape[:-1], 4, 2
            )
            full_scores = (
                F.log_softmax(quotient_logits, dim=-1).unsqueeze(-1)
                + F.log_softmax(sign_logits, dim=-1)
            ).flatten(start_dim=-2)

        return {
            "full_scores": full_scores,
            "quotient_logits": quotient_logits,
            "sign_logits": sign_logits,
        }
