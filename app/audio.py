"""Audio decoding and Silero VAD preparation shared by the API and CLI."""

from __future__ import annotations

import math
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioAssessment:
    samples: np.ndarray
    sample_rate: int
    speech_duration: float
    speech_coverage: float
    speech_dbfs: float
    noise_dbfs: float | None
    estimated_snr_db: float | None
    clipped_fraction: float
    speech_segments: tuple[tuple[int, int], ...]

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def decode_audio(
    source: Path,
    target: Path,
    *,
    max_duration: float = 31.0,
) -> np.ndarray:
    """Decode common audio/video containers to mono, 16 kHz, signed 16-bit WAV."""
    if source.suffix.lower() == ".wav":
        try:
            with wave.open(str(source), "rb") as wav:
                if (
                    wav.getnchannels() == 1
                    and wav.getsampwidth() == 2
                    and wav.getframerate() == SAMPLE_RATE
                ):
                    samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
                    if samples.size == 0:
                        raise ValueError("Audio file contains no decodable samples.")
                    if samples.size / SAMPLE_RATE >= max_duration - 0.05:
                        raise ValueError(f"Audio duration exceeds the {max_duration - 1:.0f} second limit.")
                    return samples
        except (wave.Error, EOFError):
            pass

    command = [
        os.getenv("FFMPEG_BINARY", "ffmpeg"),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-t",
        str(max_duration),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=90)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to decode uploaded audio.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"Audio decoding failed: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio decoding timed out.") from exc

    try:
        with wave.open(str(target), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != SAMPLE_RATE:
                raise ValueError("ffmpeg did not produce the expected PCM WAV format.")
            pcm = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("Decoded audio is not a valid WAV file.") from exc

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError("Audio file contains no decodable samples.")
    if samples.size / SAMPLE_RATE >= max_duration - 0.05:
        raise ValueError(f"Audio duration exceeds the {max_duration - 1:.0f} second limit.")
    return samples


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


def assess_speech(samples: np.ndarray, vad_model: torch.nn.Module) -> AudioAssessment:
    """Run Silero VAD, retain the full speech span, and report signal diagnostics."""
    tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
    try:
        tensor = tensor.to(next(vad_model.parameters()).device)
    except StopIteration:
        pass
    timestamps = get_speech_timestamps(
        tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=150,
        speech_pad_ms=100,
        return_seconds=False,
    )
    segments = tuple(
        (max(0, int(item["start"])), min(samples.size, int(item["end"])))
        for item in timestamps
        if int(item["end"]) > int(item["start"])
    )
    if not segments:
        raise ValueError("No speech detected. Record a clear spoken phrase and try again.")

    speech_samples = np.concatenate([samples[start:end] for start, end in segments])
    first, last = segments[0][0], segments[-1][1]
    trimmed = np.ascontiguousarray(samples[first:last], dtype=np.float32)

    non_speech_mask = np.ones(samples.size, dtype=bool)
    for start, end in segments:
        non_speech_mask[start:end] = False
    noise = samples[non_speech_mask]
    noise_dbfs = _dbfs(noise) if noise.size >= SAMPLE_RATE // 4 else None
    speech_dbfs = _dbfs(speech_samples)
    estimated_snr = speech_dbfs - noise_dbfs if noise_dbfs is not None else None

    return AudioAssessment(
        samples=trimmed,
        sample_rate=SAMPLE_RATE,
        speech_duration=len(speech_samples) / SAMPLE_RATE,
        speech_coverage=len(speech_samples) / max(samples.size, 1),
        speech_dbfs=speech_dbfs,
        noise_dbfs=noise_dbfs,
        estimated_snr_db=estimated_snr,
        clipped_fraction=float(np.mean(np.abs(speech_samples) >= 0.999)),
        speech_segments=segments,
    )


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono float samples as a standard PCM16 WAV file."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
