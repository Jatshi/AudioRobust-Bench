from __future__ import annotations

from audio_robust_bench import (
    BenchmarkCase,
    CorruptionAxis,
    ReliabilityEvaluator,
    TaskPrediction,
    build_corruption_manifest,
)
from audio_robust_bench.cli import main


def test_corruption_manifest_is_deterministic_and_has_stable_ids() -> None:
    axes = [
        CorruptionAxis(name="snr_db", levels=[20.0, 10.0, 0.0]),
        CorruptionAxis(name="rt60_s", levels=[0.0, 0.4]),
    ]
    first = build_corruption_manifest(["utt-1", "utt-2"], axes, seed=17)
    second = build_corruption_manifest(["utt-1", "utt-2"], axes, seed=17)

    assert first == second
    assert len(first) == 12
    assert len({case.case_id for case in first}) == 12


def test_reliability_evaluator_reports_accuracy_ece_and_failure_rate() -> None:
    cases = [
        BenchmarkCase(case_id="a", source_id="u1", corruptions={"snr_db": 20.0}, seed=1),
        BenchmarkCase(case_id="b", source_id="u2", corruptions={"snr_db": 0.0}, seed=2),
    ]
    predictions = [
        TaskPrediction(case_id="a", score=1.0, confidence=0.9, failed=False),
        TaskPrediction(case_id="b", score=0.0, confidence=0.8, failed=True),
    ]

    report = ReliabilityEvaluator(ece_bins=5).evaluate(cases, predictions)

    assert report.mean_score == 0.5
    assert report.failure_rate == 0.5
    assert 0.0 <= report.ece <= 1.0
    assert report.by_corruption["snr_db"]["0.0"].failure_rate == 1.0


def test_cli_builds_jsonl_manifest(tmp_path) -> None:
    output = tmp_path / "manifest.jsonl"
    exit_code = main(
        [
            "build-manifest",
            "--source-id",
            "utt-1",
            "--axis",
            "snr_db=20,0",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
