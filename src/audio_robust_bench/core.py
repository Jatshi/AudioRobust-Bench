"""Deterministic manifests and task-agnostic reliability metrics."""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CorruptionAxis:
    name: str
    levels: list[float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("corruption axis name cannot be empty")
        if not self.levels:
            raise ValueError("corruption axis requires at least one level")
        if any(not math.isfinite(level) for level in self.levels):
            raise ValueError("corruption levels must be finite")


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    source_id: str
    corruptions: dict[str, float]
    seed: int


def build_corruption_manifest(
    source_ids: list[str], axes: list[CorruptionAxis], *, seed: int
) -> list[BenchmarkCase]:
    if not source_ids:
        raise ValueError("source_ids cannot be empty")
    if len({axis.name for axis in axes}) != len(axes):
        raise ValueError("corruption axis names must be unique")
    combinations = itertools.product(*(axis.levels for axis in axes))
    grid = [dict(zip((axis.name for axis in axes), values, strict=True)) for values in combinations]
    cases: list[BenchmarkCase] = []
    for source_id in source_ids:
        if not source_id:
            raise ValueError("source_id cannot be empty")
        for corruption in grid:
            canonical = ";".join(f"{key}={corruption[key]:.8g}" for key in sorted(corruption))
            digest = hashlib.sha256(f"{seed}|{source_id}|{canonical}".encode()).hexdigest()
            case_seed = random.Random(int(digest[:16], 16)).randrange(0, 2**31)
            cases.append(
                BenchmarkCase(
                    case_id=f"arb-{digest[:16]}",
                    source_id=source_id,
                    corruptions=corruption,
                    seed=case_seed,
                )
            )
    return cases


@dataclass(frozen=True, slots=True)
class TaskPrediction:
    case_id: str
    score: float
    confidence: float
    failed: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SliceReport:
    examples: int
    mean_score: float
    failure_rate: float


@dataclass(frozen=True, slots=True)
class ReliabilityReport:
    examples: int
    mean_score: float
    failure_rate: float
    ece: float
    by_corruption: dict[str, dict[str, SliceReport]] = field(default_factory=dict)


class ReliabilityEvaluator:
    def __init__(self, *, ece_bins: int = 10) -> None:
        if ece_bins <= 0:
            raise ValueError("ece_bins must be positive")
        self.ece_bins = ece_bins

    def evaluate(
        self, cases: list[BenchmarkCase], predictions: list[TaskPrediction]
    ) -> ReliabilityReport:
        if not cases:
            raise ValueError("cases cannot be empty")
        indexed = {prediction.case_id: prediction for prediction in predictions}
        if len(indexed) != len(predictions):
            raise ValueError("prediction case_ids must be unique")
        expected = {case.case_id for case in cases}
        if set(indexed) != expected:
            missing = sorted(expected - set(indexed))
            extra = sorted(set(indexed) - expected)
            raise ValueError(f"prediction ids mismatch; missing={missing}, extra={extra}")
        ordered = [indexed[case.case_id] for case in cases]
        slices: dict[str, dict[str, SliceReport]] = {}
        for axis in sorted({key for case in cases for key in case.corruptions}):
            by_level: dict[str, list[TaskPrediction]] = {}
            for case, prediction in zip(cases, ordered, strict=True):
                level = str(case.corruptions[axis])
                by_level.setdefault(level, []).append(prediction)
            slices[axis] = {
                level: self._slice(rows) for level, rows in sorted(by_level.items())
            }
        return ReliabilityReport(
            examples=len(cases),
            mean_score=sum(item.score for item in ordered) / len(ordered),
            failure_rate=sum(item.failed for item in ordered) / len(ordered),
            ece=self._ece(ordered),
            by_corruption=slices,
        )

    @staticmethod
    def _slice(rows: list[TaskPrediction]) -> SliceReport:
        return SliceReport(
            examples=len(rows),
            mean_score=sum(item.score for item in rows) / len(rows),
            failure_rate=sum(item.failed for item in rows) / len(rows),
        )

    def _ece(self, rows: list[TaskPrediction]) -> float:
        total = len(rows)
        ece = 0.0
        for bin_index in range(self.ece_bins):
            lower = bin_index / self.ece_bins
            upper = (bin_index + 1) / self.ece_bins
            bucket = [
                item
                for item in rows
                if lower <= item.confidence <= upper
                and (bin_index == self.ece_bins - 1 or item.confidence < upper)
            ]
            if not bucket:
                continue
            accuracy = sum(item.score for item in bucket) / len(bucket)
            confidence = sum(item.confidence for item in bucket) / len(bucket)
            ece += len(bucket) / total * abs(accuracy - confidence)
        return ece
