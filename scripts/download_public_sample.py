from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav"
MIRROR_URL = f"https://ghproxy.net/{URL}"
SHA256 = "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the pinned public JFK speech sample")
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/public/jfk.wav"))
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        payload = output.read_bytes()
        if hashlib.sha256(payload).hexdigest() == SHA256:
            print(output.resolve())
            return 0
    errors: list[str] = []
    for url in (URL, MIRROR_URL):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "AudioRobust-Bench/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            break
        except OSError as exc:
            errors.append(f"{url}: {exc}")
    else:
        raise RuntimeError("sample download failed: " + "; ".join(errors))
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SHA256:
        raise RuntimeError(f"sample hash mismatch: expected {SHA256}, got {actual}")
    output.write_bytes(payload)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
