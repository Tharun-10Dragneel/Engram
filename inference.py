"""
inference.py — Gemma 4 E2B inference, clean.

Text input → streamed text output.
No web_signal injection. No prompt engineering.
Memory lives in weights (TTT), not in the system prompt.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

_model     = None
_tokenizer = None


def _get_model_path() -> str:
    cfg = Path(__file__).parent / "model_path.txt"
    if cfg.exists():
        path = cfg.read_text().strip()
        if path and Path(path).exists():
            return path
    return "unsloth/gemma-4-E2B-it-UD-MLX-4bit"


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from mlx_lm import load
    model_path = _get_model_path()
    print(f"[inference] loading: {model_path}")
    _model, _tokenizer = load(model_path)
    print("[inference] ready.\n")
    return _model, _tokenizer


SYSTEM = "You are IRA, a voice companion. Respond naturally and warmly."


def generate_stream(
    message: str,
    history: list | None = None,
    max_tokens: int = 256,
    temperature: float = 0.85,
    top_p: float = 0.92,
) -> Iterator[str]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tokenizer = _load_model()
    messages = [{"role": "system", "content": SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    sampler = make_sampler(temp=temperature, top_p=top_p)
    thinking = False
    buf = ""
    for chunk in stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens, sampler=sampler,
    ):
        buf += chunk.text

        if not thinking and "<|channel>" in buf:
            # yield anything that came before the thinking block
            pre = buf.split("<|channel>", 1)[0]
            if pre:
                yield pre
            buf = ""
            thinking = True
            continue

        if thinking:
            if "<channel|>" in buf:
                after = buf.split("<channel|>", 1)[-1]
                buf = after
                thinking = False
                # fall through to yield below
            else:
                buf = ""
                continue

        if buf:
            yield buf
            buf = ""


def generate(message: str, **kwargs) -> str:
    return "".join(generate_stream(message, **kwargs))


if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) or "hey"
    print("IRA: ", end="", flush=True)
    for tok in generate_stream(msg):
        print(tok, end="", flush=True)
    print()
