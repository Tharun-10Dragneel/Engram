"""
ttt.py — In-Place Test-Time Training for AssocTTT

After each conversation turn the web deems strong enough,
we run a few gradient steps on LoRA delta weights attached
to Gemma 4 E2B's MLP layers (gate_proj, up_proj, down_proj).

The delta weights start at zero. They grow only where real
patterns land. They persist across sessions. Over time the
model stops retrieving and starts knowing.

No retraining. No separate database. The weights ARE the memory.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# ── config ─────────────────────────────────────────────────────────────────────
ADAPTER_PATH   = Path("ttt_adapters.npz")   # persisted delta weights
TTT_LR         = 5e-5                        # small — we don't want to blast old memories
TTT_STEPS      = 3                           # gradient steps per turn
TTT_RANK       = 8                           # LoRA rank for delta weights
MLP_TARGETS    = {"gate_proj", "up_proj", "down_proj"}   # which MLP weights we write


# ── LoRA delta layer ───────────────────────────────────────────────────────────
class LoRADelta(nn.Module):
    """
    Wraps a frozen QuantizedLinear with a trainable low-rank delta.
    forward(x) = frozen_output(x) + lora_B(lora_A(x))
    lora_A: [in_features, rank]
    lora_B: [rank, out_features]  — init to zero so delta starts at 0
    """
    def __init__(self, frozen: nn.Module, rank: int = TTT_RANK):
        super().__init__()
        self.frozen = frozen
        # infer dims from frozen weight
        # QuantizedLinear stores weight as (out, in) after dequant
        w = mx.dequantize(
            frozen.weight,
            frozen.scales,
            frozen.biases,
            frozen.group_size,
            frozen.bits,
        )
        out_features, in_features = w.shape
        self.lora_A = mx.zeros((in_features, rank), dtype=mx.float16)
        self.lora_B = mx.zeros((rank, out_features), dtype=mx.float16)

    def __call__(self, x: mx.array) -> mx.array:
        base = self.frozen(x)
        delta = (x.astype(mx.float16) @ self.lora_A) @ self.lora_B
        return base + delta.astype(base.dtype)

    @property
    def trainable_params(self):
        return {"lora_A": self.lora_A, "lora_B": self.lora_B}


# ── inject / remove deltas ─────────────────────────────────────────────────────
def inject_deltas(model: nn.Module, rank: int = TTT_RANK) -> dict[str, LoRADelta]:
    """
    Walk model layers, replace target MLP QuantizedLinear layers with LoRADelta.
    Returns a flat dict of path → LoRADelta for easy gradient access.
    """
    deltas: dict[str, LoRADelta] = {}
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj_name in MLP_TARGETS:
            if not hasattr(mlp, proj_name):
                continue
            proj = getattr(mlp, proj_name)
            if not isinstance(proj, nn.QuantizedLinear):
                continue
            delta = LoRADelta(proj, rank=rank)
            setattr(mlp, proj_name, delta)
            deltas[f"layer_{i}_{proj_name}"] = delta
    return deltas


def load_adapters(deltas: dict[str, LoRADelta]) -> None:
    """Load persisted LoRA weights into the injected delta layers."""
    if not ADAPTER_PATH.exists():
        return
    saved = dict(mx.load(str(ADAPTER_PATH)))
    for key, delta in deltas.items():
        a_key = f"{key}_A"
        b_key = f"{key}_B"
        if a_key in saved:
            delta.lora_A = saved[a_key]
        if b_key in saved:
            delta.lora_B = saved[b_key]
    mx.eval([d.lora_A for d in deltas.values()] + [d.lora_B for d in deltas.values()])
    print(f"[ttt] Loaded adapters from {ADAPTER_PATH}")


def save_adapters(deltas: dict[str, LoRADelta]) -> None:
    """Persist LoRA weights to disk."""
    flat = {}
    for key, delta in deltas.items():
        flat[f"{key}_A"] = delta.lora_A
        flat[f"{key}_B"] = delta.lora_B
    mx.savez(str(ADAPTER_PATH), **flat)


# ── TTT update ─────────────────────────────────────────────────────────────────
def update(
    model: nn.Module,
    tokenizer,
    deltas: dict[str, LoRADelta],
    turn_text: str,
    web_confidence: float = 1.0,
    lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> float:
    """
    Run TTT gradient steps on this turn's text.
    Only called when web spreading activation fires with high confidence.

    web_confidence scales the effective learning rate:
      high confidence pattern → bigger update
      weak pattern → smaller update

    Returns final loss value.
    """
    effective_lr = (lr or TTT_LR) * web_confidence
    n_steps = steps or TTT_STEPS

    tokens = tokenizer.encode(turn_text, add_special_tokens=False)
    if len(tokens) < 2:
        return 0.0

    ids = mx.array(tokens)
    x   = ids[:-1]   # input
    y   = ids[1:]     # target (next token)

    # collect trainable params from all delta layers
    trainable = {}
    for key, delta in deltas.items():
        trainable[f"{key}_A"] = delta.lora_A
        trainable[f"{key}_B"] = delta.lora_B

    def loss_fn(params):
        # write current params back into delta layers
        for key, delta in deltas.items():
            delta.lora_A = params[f"{key}_A"]
            delta.lora_B = params[f"{key}_B"]
        logits = model(x[None])       # [1, seq_len, vocab]
        loss   = nn.losses.cross_entropy(logits[0], y, reduction="mean")
        return loss

    final_loss = 0.0
    for step in range(n_steps):
        loss_val, grads = mx.value_and_grad(loss_fn)(trainable)
        mx.eval(loss_val)
        final_loss = loss_val.item()

        # SGD step
        for key in trainable:
            trainable[key] = trainable[key] - effective_lr * grads[key]
        mx.eval(list(trainable.values()))

    # write updated params back into delta layers
    for key, delta in deltas.items():
        delta.lora_A = trainable[f"{key}_A"]
        delta.lora_B = trainable[f"{key}_B"]

    save_adapters(deltas)
    print(f"[ttt] Updated weights | steps={n_steps} | lr={effective_lr:.2e} | loss={final_loss:.4f}")
    return final_loss


# ── confidence threshold ────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.6   # web_signal confidence below this → skip TTT, just update web graph

def should_write(web_signal: dict) -> bool:
    """Only write to weights when the web is confident about the pattern."""
    return web_signal.get("confidence", 0.0) >= MIN_CONFIDENCE


# ── standalone test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mlx_lm import load
    from inference import _get_model_path

    print("[ttt] Loading model...")
    model, tokenizer = load(_get_model_path())

    print("[ttt] Injecting LoRA deltas into MLP layers...")
    deltas = inject_deltas(model)
    print(f"[ttt] {len(deltas)} delta layers injected: {list(deltas.keys())[:4]} ...")

    load_adapters(deltas)

    # simulate one high-confidence TTT write
    test_turn = "I'm dead 💀 that was actually hilarious"
    web_signal = {"emotion": "humor", "confidence": 0.91, "pattern": "skull+humor"}

    if should_write(web_signal):
        loss = update(model, tokenizer, deltas, test_turn, web_confidence=web_signal["confidence"])
        print(f"[ttt] Done. Final loss: {loss:.4f}")
    else:
        print("[ttt] Confidence too low — skipping weight write")
