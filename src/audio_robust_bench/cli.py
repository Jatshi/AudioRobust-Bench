"""Command-line entry points for reproducible manifest construction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from audio_robust_bench.core import CorruptionAxis, build_corruption_manifest


def _axis(value: str) -> CorruptionAxis:
    try:
        name, raw_levels = value.split("=", maxsplit=1)
        levels = [float(item) for item in raw_levels.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("axis must look like name=level1,level2") from error
    try:
        return CorruptionAxis(name=name, levels=levels)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audio-robust-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--source-id", action="append", required=True)
    manifest.add_argument("--axis", action="append", type=_axis, required=True)
    manifest.add_argument("--seed", type=int, default=17)
    manifest.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-manifest":
        cases = build_corruption_manifest(args.source_id, args.axis, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                handle.write(json.dumps(asdict(case), ensure_ascii=False, sort_keys=True) + "\n")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


def entrypoint() -> None:
    raise SystemExit(main())
