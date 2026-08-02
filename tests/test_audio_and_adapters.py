from __future__ import annotations

import numpy as np

from audio_robust_bench import BenchmarkCase
from audio_robust_bench.adapters import (
    ASRAdapter,
    SoundEventAdapter,
    SpeakerCloningAdapter,
)
from audio_robust_bench.audio import apply_corruptions


def test_audio_corruption_is_deterministic_for_case_seed() -> None:
    source = np.sin(np.linspace(0, 20, 1600, dtype=np.float32)) * 0.1
    case = BenchmarkCase(
        case_id="case",
        source_id="utt",
        corruptions={"snr_db": 5.0, "clip_threshold": 0.08, "packet_loss": 0.1},
        seed=123,
    )
    first = apply_corruptions(source, 16000, case)
    second = apply_corruptions(source, 16000, case)
    np.testing.assert_allclose(first, second)
    assert float(np.max(np.abs(first))) <= 0.080001


def test_task_adapters_normalize_scores_to_unit_interval() -> None:
    asr = ASRAdapter().prediction("你好世界", "你好世", case_id="asr")
    speaker = SpeakerCloningAdapter().prediction(0.83, case_id="speaker")
    detection = SoundEventAdapter().prediction(
        expected={"gunshot", "baby_cry"},
        predicted={"gunshot"},
        case_id="sed",
    )
    assert asr.score == 0.75
    assert speaker.score == 0.83
    assert round(detection.score, 6) == round(2 / 3, 6)
