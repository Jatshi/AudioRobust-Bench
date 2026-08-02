"""Render the README GIF from the measured smoke manifest."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "smoke" / "run_manifest.json"
OUTPUT = ROOT / "assets" / "audio_robust_v2_demo.gif"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = [
        ("ASR accuracy", "asr", "#5EEAD4"),
        ("Speaker consistency", "speaker_consistency", "#A78BFA"),
        ("Event consistency", "event_consistency", "#FBBF24"),
    ]
    strengths = [0.0, 10.0, 100.0]
    labels = ["0 dB", "10 dB", "clean"]
    frames: list[Image.Image] = []
    for frame_index in range(28):
        progress = min(1.0, frame_index / 20)
        image = Image.new("RGB", (1000, 560), "#07111F")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((35, 28, 965, 530), 28, fill="#0D1B2A", outline="#1F3B55", width=2)
        draw.text((70, 58), "AudioRobust-Bench 2.0", fill="#F8FAFC", font=font(38, True))
        draw.text((70, 108), "One manifest · three real models · auditable claim boundaries", fill="#94A3B8", font=font(21))
        left, top, right, bottom = 95, 180, 650, 450
        draw.line((left, bottom, right, bottom), fill="#35516B", width=2)
        draw.line((left, top, left, bottom), fill="#35516B", width=2)
        for tick in range(6):
            y = bottom - tick * (bottom - top) / 5
            draw.line((left, y, right, y), fill="#173047", width=1)
            draw.text((52, y - 10), f"{tick/5:.1f}", fill="#64748B", font=font(15))
        xs = [left + i * (right - left) / 2 for i in range(3)]
        for x, label in zip(xs, labels, strict=True):
            draw.text((x - 22, bottom + 12), label, fill="#94A3B8", font=font(16))
        for title, key, color in tasks:
            values = [
                payload["reports"][key]["by_corruption"]["snr_db"][str(value)]["mean_score"]
                for value in strengths
            ]
            points = [(xs[i], bottom - values[i] * (bottom - top)) for i in range(3)]
            visible = 1 + int(progress * (len(points) - 1))
            partial = points[:visible]
            if visible < len(points):
                x0, y0 = points[visible - 1]
                x1, y1 = points[visible]
                local = progress * (len(points) - 1) - (visible - 1)
                partial.append((x0 + (x1 - x0) * local, y0 + (y1 - y0) * local))
            if len(partial) > 1:
                draw.line(partial, fill=color, width=5, joint="curve")
            for x, y in partial[:-1] if visible < len(points) else partial:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        draw.text((700, 185), "Measured on RTX 4090", fill="#E2E8F0", font=font(22, True))
        for index, (title, key, color) in enumerate(tasks):
            mean = payload["reports"][key]["mean_score"]
            y = 235 + index * 65
            draw.rounded_rectangle((700, y, 920, y + 46), 12, fill="#12283B")
            draw.ellipse((716, y + 15, 730, y + 29), fill=color)
            draw.text((742, y + 9), f"{title}: {mean:.3f}", fill="#E2E8F0", font=font(17))
        draw.text((700, 442), "Smoke scope · not corpus accuracy", fill="#FB7185", font=font(17, True))
        draw.text((70, 495), "Source hashes, model IDs, environment and per-SNR scores are preserved in run_manifest.json", fill="#64748B", font=font(15))
        frames.append(image)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
