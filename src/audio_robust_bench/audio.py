"""Deterministic waveform corruptions shared across all benchmark tasks."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from audio_robust_bench.core import BenchmarkCase

FloatAudio: TypeAlias = NDArray[np.float32]


def _validate_audio(audio: FloatAudio, sample_rate: int) -> FloatAudio:
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("audio must be a non-empty mono waveform")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.all(np.isfinite(signal)):
        raise ValueError("audio contains non-finite samples")
    return signal.copy()


def _add_noise(signal: FloatAudio, snr_db: float, rng: np.random.Generator) -> FloatAudio:
    signal_power = float(np.mean(np.square(signal)))
    if signal_power <= 1e-12:
        return signal
    noise = rng.standard_normal(signal.shape).astype(np.float32)
    noise_power = float(np.mean(np.square(noise)))
    target_noise_power = signal_power / (10 ** (snr_db / 10.0))
    return signal + noise * np.sqrt(target_noise_power / max(noise_power, 1e-12))


def _add_reverb(signal: FloatAudio, sample_rate: int, rt60_s: float) -> FloatAudio:
    if rt60_s <= 0:
        return signal
    length = max(2, min(signal.size, int(sample_rate * min(rt60_s, 2.0))))
    times = np.arange(length, dtype=np.float32) / sample_rate
    impulse = np.exp(-6.9078 * times / rt60_s).astype(np.float32)
    impulse[0] = 1.0
    impulse /= np.sqrt(np.sum(np.square(impulse)))
    return np.convolve(signal, impulse, mode="full")[: signal.size].astype(np.float32)


def _bandlimit(signal: FloatAudio, sample_rate: int, bandwidth_hz: float) -> FloatAudio:
    if bandwidth_hz <= 0 or bandwidth_hz >= sample_rate / 2:
        return signal
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    spectrum[frequencies > bandwidth_hz] = 0
    return np.fft.irfft(spectrum, n=signal.size).astype(np.float32)


def _drop_packets(
    signal: FloatAudio, sample_rate: int, ratio: float, rng: np.random.Generator
) -> FloatAudio:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("packet_loss must be in [0, 1]")
    if ratio == 0:
        return signal
    packet = max(1, sample_rate // 100)
    result = signal.copy()
    count = int(np.ceil(signal.size / packet))
    drop = rng.random(count) < ratio
    for index, should_drop in enumerate(drop):
        if should_drop:
            result[index * packet : min((index + 1) * packet, signal.size)] = 0
    return result


def apply_corruptions(audio: FloatAudio, sample_rate: int, case: BenchmarkCase) -> FloatAudio:
    """Apply corruptions in a fixed, documented order using the case seed."""

    signal = _validate_audio(audio, sample_rate)
    rng = np.random.default_rng(case.seed)
    values = case.corruptions
    if "snr_db" in values:
        signal = _add_noise(signal, values["snr_db"], rng)
    if "rt60_s" in values:
        signal = _add_reverb(signal, sample_rate, values["rt60_s"])
    if "bandwidth_hz" in values:
        signal = _bandlimit(signal, sample_rate, values["bandwidth_hz"])
    if "packet_loss" in values:
        signal = _drop_packets(signal, sample_rate, values["packet_loss"], rng)
    if "clip_threshold" in values:
        threshold = values["clip_threshold"]
        if not 0 < threshold <= 1:
            raise ValueError("clip_threshold must be in (0, 1]")
        signal = np.clip(signal, -threshold, threshold)
    return np.asarray(signal, dtype=np.float32)
