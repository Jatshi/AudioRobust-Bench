#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${AUDIO_ROBUST_VENV:-/root/autodl-tmp/portfolio-v3/envs/audio-robust}"
PYTHON_BIN="${AUDIO_ROBUST_PYTHON:-python3}"
CACHE_ROOT="${HF_HOME:-/root/autodl-tmp/portfolio-v3/cache/huggingface}"
export HF_HOME="$CACHE_ROOT"
mkdir -p "$(dirname "$VENV")" "$CACHE_ROOT" "$ROOT/artifacts/system"
"$PYTHON_BIN" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.0 torchaudio==2.9.0
python -m pip install -e "$ROOT" pytest ruff \
  faster-whisper==1.2.1 "transformers>=4.57,<5" soundfile==0.13.1 scipy==1.15.3
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python -m pip freeze > "$ROOT/artifacts/system/environment.freeze.txt"
