"""
mic.py — Mic input → Gemma 4 E2B → text response.

Hold ENTER to start recording, release to stop.
E2B receives raw audio directly — no STT middleware.

Run:
    uv run python mic.py
"""

from __future__ import annotations
import tempfile
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from pathlib import Path
from inference import _load_model, _get_model_path

SAMPLE_RATE = 16_000
CHANNELS    = 1


def record_until_enter() -> np.ndarray:
    """Record mic until user presses ENTER. Returns float32 audio array."""
    chunks = []
    stop   = threading.Event()

    def callback(indata, frames, time, status):
        if not stop.is_set():
            chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )

    with stream:
        input("  [mic] press ENTER to start... ")
        print("  [mic] recording — press ENTER to stop")
        input()
        stop.set()

    audio = np.concatenate(chunks, axis=0).squeeze()
    print(f"  [mic] captured {len(audio)/SAMPLE_RATE:.1f}s of audio")
    return audio


def audio_to_response(audio: np.ndarray) -> None:
    """Pass raw audio to Gemma 4 E2B via mlx-vlm and stream response."""
    from mlx_vlm import load as vlm_load
    from mlx_vlm.generate import generate as vlm_generate

    model_path = _get_model_path()

    # save audio to temp wav — mlx-vlm reads from file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    wav.write(wav_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print("[IRA] loading model for audio input...")
    model, processor = vlm_load(model_path)

    print("IRA: ", end="", flush=True)
    output = vlm_generate(
        model,
        processor,
        prompt="Respond to what you just heard naturally.",
        image=wav_path,      # mlx-vlm uses 'image' arg for any media
        max_tokens=256,
        temperature=0.85,
        verbose=False,
    )
    print(output)
    Path(wav_path).unlink(missing_ok=True)


if __name__ == "__main__":
    print("\nAssocTTT — mic test")
    print("=" * 40)
    audio = record_until_enter()
    audio_to_response(audio)
