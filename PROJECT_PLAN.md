# AudioRobust-Bench 1.0 implementation plan

## Goal and acceptance

The final release must run the same versioned corruption manifest through ASR,
speaker-cloning, and sound-event adapters, then emit machine-readable reliability
curves with provenance. Acceptance requires full tests/Ruff/build, three real-model
smoke runs, at least three fixed seeds, raw manifests, and no unsupported claims.

## Completed CPU slice

- [x] Deterministic Cartesian corruption manifest with stable IDs and seeds.
- [x] Noise, reverb, bandwidth, packet-loss, and clipping waveform transforms.
- [x] ASR character accuracy, speaker similarity, and event F1 adapters.
- [x] Mean score, failure rate, ECE, and per-strength slice reports.
- [x] Manifest CLI and unit tests.

## Remaining verified slices

- [ ] Add JSONL prediction ingestion and report export with schema version and Git SHA.
- [ ] Add adapters that invoke the three source projects without importing private weights.
- [ ] Add real-model smoke fixtures licensed for redistribution.
- [ ] Run the `5 seeds x corruption grid x 3 tasks` AutoDL matrix.
- [ ] Bootstrap confidence intervals and plot degradation/calibration curves.
- [ ] Publish dataset card, model-source cards, SHA-256 manifest, README GIF, and learning guide.

## Non-goals

- No new speech recognizer, cloning model, or event detector is trained in this repository.
- No third-party fork is represented as original work.
- No clean-average-only leaderboard; reliability under degradation is the primary object.

## Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Data identity | source ID + corruption values + seed hash | Stable, auditable, path-independent |
| Corruption order | noise -> reverb -> bandwidth -> packet loss -> clipping | Explicit and deterministic |
| Common score | task adapters normalize to `[0,1]` | Allows one report without pretending task metrics are identical |
| Large artifacts | Hugging Face/DVC + hashes | Git remains source-first |

## Known failure history

The first test run failed at import because the new modules did not yet exist. This
was intentional TDD evidence. The scaffold initially lacked NumPy despite waveform
code requiring it; the dependency is now explicit in `pyproject.toml`.
