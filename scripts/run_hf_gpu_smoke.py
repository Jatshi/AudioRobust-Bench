"""Run three real Hugging Face audio models on clean and corrupted speech.

ASR uses a transcript reference. Speaker verification and audio classification use
clean-model outputs as anchors, so their smoke scores measure corruption consistency,
not task accuracy. The distinction is recorded in the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audio_robust_bench.adapters import (
    ASRAdapter,
    SoundEventAdapter,
    SpeakerCloningAdapter,
)
from audio_robust_bench.audio import apply_corruptions
from audio_robust_bench.core import (
    BenchmarkCase,
    ReliabilityEvaluator,
    TaskPrediction,
)

REFERENCE = (
    "And so my fellow Americans ask not what your country can do for you ask what you "
    "can do for your country"
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-model", default="Systran/faster-whisper-tiny.en")
    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--event-model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))


def main() -> int:
    args = _args()
    import soundfile as sf
    import torch
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        AutoModelForAudioXVector,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not run this GPU smoke locally")
    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != 16_000:
        waveform = resample_poly(waveform, 16_000, sample_rate).astype(np.float32)
        sample_rate = 16_000

    levels: list[float | None] = [None, 10.0, 0.0]
    level_names = ["clean" if level is None else f"{level:g}" for level in levels]
    cases = [
        BenchmarkCase(
            case_id=f"jfk-snr-{level_name}",
            source_id="jfk-public",
            corruptions={"snr_db": 100.0 if level is None else level},
            seed=20260802 + index,
        )
        for index, (level, level_name) in enumerate(zip(levels, level_names, strict=True))
    ]
    waves = [
        waveform if level is None else apply_corruptions(waveform, sample_rate, case)
        for level, case in zip(levels, cases, strict=True)
    ]

    asr_model = WhisperModel(args.asr_model, device="cuda", compute_type="float16")
    asr_adapter = ASRAdapter()
    asr_predictions: list[TaskPrediction] = []
    transcripts: list[str] = []
    for case, audio in zip(cases, waves, strict=True):
        segments, info = asr_model.transcribe(audio, language="en", beam_size=1)
        hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
        transcripts.append(hypothesis)
        asr_predictions.append(
            asr_adapter.prediction(
                REFERENCE,
                hypothesis,
                case_id=case.case_id,
                confidence=min(1.0, max(0.0, float(info.language_probability))),
            )
        )

    speaker_extractor = AutoFeatureExtractor.from_pretrained(args.speaker_model)
    speaker_model = AutoModelForAudioXVector.from_pretrained(args.speaker_model).cuda().eval()
    speaker_vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for audio in waves:
            batch = speaker_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
            embedding = speaker_model(**{key: value.cuda() for key, value in batch.items()}).embeddings
            speaker_vectors.append(embedding[0].float().cpu().numpy())
    speaker_adapter = SpeakerCloningAdapter()
    speaker_predictions = [
        speaker_adapter.prediction(
            _cosine(speaker_vectors[0], vector), case_id=case.case_id
        )
        for case, vector in zip(cases, speaker_vectors, strict=True)
    ]

    event_extractor = AutoFeatureExtractor.from_pretrained(args.event_model)
    event_model = AutoModelForAudioClassification.from_pretrained(args.event_model).cuda().eval()
    event_labels: list[set[str]] = []
    event_confidences: list[float] = []
    with torch.inference_mode():
        for audio in waves:
            batch = event_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
            logits = event_model(**{key: value.cuda() for key, value in batch.items()}).logits[0]
            probabilities = torch.sigmoid(logits)
            values, indices = torch.topk(probabilities, k=5)
            event_labels.append({event_model.config.id2label[int(index)] for index in indices})
            event_confidences.append(float(values.mean().cpu()))
    event_adapter = SoundEventAdapter()
    event_predictions = [
        event_adapter.prediction(
            expected=event_labels[0],
            predicted=labels,
            case_id=case.case_id,
            confidence=confidence,
        )
        for case, labels, confidence in zip(
            cases, event_labels, event_confidences, strict=True
        )
    ]

    evaluator = ReliabilityEvaluator(ece_bins=5)
    reports = {
        "asr": asdict(evaluator.evaluate(cases, asr_predictions)),
        "speaker_consistency": asdict(evaluator.evaluate(cases, speaker_predictions)),
        "event_consistency": asdict(evaluator.evaluate(cases, event_predictions)),
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "measurement_scope": {
            "asr": "reference-based task quality",
            "speaker_consistency": "clean-embedding anchor; not cloning naturalness",
            "event_consistency": "clean top-5 pseudo-label anchor; not ground-truth accuracy",
        },
        "models": {
            "asr": args.asr_model,
            "speaker": args.speaker_model,
            "event": args.event_model,
        },
        "input": {"path": str(args.audio), "sha256": _sha256(args.audio)},
        "cuda": {
            "torch": torch.__version__,
            "runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
        },
        "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
        "transcripts": dict(zip((case.case_id for case in cases), transcripts, strict=True)),
        "event_labels": {
            case.case_id: sorted(labels) for case, labels in zip(cases, event_labels, strict=True)
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
