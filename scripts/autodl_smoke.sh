#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${AUDIO_ROBUST_VENV:-/root/autodl-tmp/portfolio-v3/envs/audio-robust}"
OUTPUT="${AUDIO_ROBUST_OUTPUT:-$ROOT/artifacts/smoke/run_manifest.json}"
source "$VENV/bin/activate"
cd "$ROOT"
python -m pytest -q
python scripts/download_public_sample.py data/public/jfk.wav
python scripts/run_hf_gpu_smoke.py --audio data/public/jfk.wav --output "$OUTPUT"
python -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["status"]=="passed"' "$OUTPUT"
