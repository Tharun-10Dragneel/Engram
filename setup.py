"""
setup.py — AssocTTT Gemma 4 E2B environment setup.

Run with:
    uv run python setup.py
"""

import subprocess
import sys
import os

DEPS = [
    "mlx-lm>=0.22.0",
    "mlx-vlm>=0.1.0",
    "huggingface-hub",
    "numpy",
    "sounddevice",
    "scipy",
]

MODEL_ID = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
LOCAL_DIR = os.path.expanduser("~/.cache/assocttt/gemma4-e2b-4bit")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "model_path.txt")


def install_deps():
    print("[setup] Installing dependencies via uv pip ...")
    uv_pip = ["uv", "pip", "install", "--upgrade"] + DEPS
    try:
        subprocess.run(uv_pip, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[setup] uv not found, falling back to pip ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + DEPS, check=True)
    print("[setup] Dependencies installed.\n")


def download_model():
    print(f"[setup] Downloading model: {MODEL_ID}")
    print(f"        Destination  : {LOCAL_DIR}\n")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        ignore_patterns=["*.bin", "*.pt"],
    )
    print(f"\n[setup] Model ready at: {LOCAL_DIR}")


def write_config():
    with open(CONFIG_PATH, "w") as f:
        f.write(LOCAL_DIR + "\n")
    print(f"[setup] Model path written to {CONFIG_PATH}")


if __name__ == "__main__":
    install_deps()
    download_model()
    write_config()
    print("\n[setup] Done. Run:  uv run python inference.py")
