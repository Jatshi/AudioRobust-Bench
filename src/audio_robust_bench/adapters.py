"""Task-specific metrics normalized to the common reliability contract."""

from __future__ import annotations

from audio_robust_bench.core import TaskPrediction


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


class ASRAdapter:
    def prediction(
        self,
        reference: str,
        hypothesis: str,
        *,
        case_id: str,
        confidence: float | None = None,
    ) -> TaskPrediction:
        reference_chars = [char for char in reference if not char.isspace()]
        hypothesis_chars = [char for char in hypothesis if not char.isspace()]
        if not reference_chars:
            raise ValueError("ASR reference cannot be empty")
        cer = _edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)
        score = 1.0 - min(1.0, cer)
        return TaskPrediction(
            case_id=case_id,
            score=score,
            confidence=score if confidence is None else confidence,
            failed=not bool(hypothesis_chars),
        )


class SpeakerCloningAdapter:
    def prediction(
        self, similarity: float, *, case_id: str, confidence: float | None = None
    ) -> TaskPrediction:
        score = min(1.0, max(0.0, similarity))
        return TaskPrediction(
            case_id=case_id,
            score=score,
            confidence=score if confidence is None else confidence,
            failed=False,
        )


class SoundEventAdapter:
    def prediction(
        self,
        *,
        expected: set[str],
        predicted: set[str],
        case_id: str,
        confidence: float | None = None,
    ) -> TaskPrediction:
        if not expected:
            raise ValueError("expected sound-event labels cannot be empty")
        true_positive = len(expected & predicted)
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(expected)
        f1 = 2 * precision * recall / (precision + recall) if true_positive else 0.0
        return TaskPrediction(
            case_id=case_id,
            score=f1,
            confidence=f1 if confidence is None else confidence,
            failed=not bool(predicted),
        )
