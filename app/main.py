"""
Aurix ML service (reference implementation).

Pipeline:
    upload -> ffmpeg (16 kHz mono) -> silence trim -> Whisper -> phrase match

Speaker verification and PAD are placeholders here. Replace them with the
real models (ECAPA-TDNN and AASIST) when they are ready.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_CACHE = os.getenv("WHISPER_CACHE", "/models")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pl")
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

DIGIT_WORDS = {
    "0": "zero",
    "1": "jeden",
    "2": "dwa",
    "3": "trzy",
    "4": "cztery",
    "5": "pięć",
    "6": "sześć",
    "7": "siedem",
    "8": "osiem",
    "9": "dziewięć",
}

app = FastAPI(title="Aurix ML", version="0.1.0")

_model: WhisperModel | None = None


def get_model() -> WhisperModel:
    """Load Whisper lazily so /health works without downloading the model."""
    global _model

    if _model is None:
        _model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            download_root=WHISPER_CACHE,
        )

    return _model


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [DIGIT_WORDS.get(token, token) for token in text.split()]

    return " ".join(tokens)


def phrase_match(expected: str, transcript: str) -> float:
    """Fraction of expected words found in the transcript (fuzzy per word)."""
    expected_words = normalize(expected).split()
    heard_words = normalize(transcript).split()

    if not expected_words:
        return 0.0

    matched = sum(
        1
        for word in expected_words
        if any(fuzz.ratio(word, heard) >= 80 for heard in heard_words)
    )

    return matched / len(expected_words)


def to_wav(source: Path, target: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-ar", "16000", "-ac", "1", str(target),
        ],
        check=True,
        capture_output=True,
    )


def trim_silence(source: Path, target: Path) -> None:
    """Trim leading and trailing silence, keeping the speech segment."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-af",
            "silenceremove="
            "start_periods=1:start_duration=0.1:start_threshold=-45dB:"
            "stop_periods=-1:stop_duration=0.1:stop_threshold=-45dB",
            "-ar", "16000", "-ac", "1", str(target),
        ],
        check=True,
        capture_output=True,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": WHISPER_MODEL,
        "loaded": _model is not None,
    }


@app.post("/v1/verify")
async def verify(
    audio: UploadFile = File(...),
    expected_phrase: str = Form(...),
    x_internal_token: str | None = Header(default=None),
) -> dict:
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "input"
        wav = Path(tmp) / "input.wav"
        trimmed = Path(tmp) / "trimmed.wav"

        source.write_bytes(await audio.read())

        try:
            to_wav(source, wav)
            trim_silence(wav, trimmed)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(status_code=400, detail="audio decoding failed") from exc

        segments, info = get_model().transcribe(str(trimmed), language=WHISPER_LANGUAGE or None, vad_filter=True)
        transcript = " ".join(segment.text for segment in segments)

    phrase_match_score = phrase_match(expected_phrase, transcript)

    # Placeholders until ECAPA-TDNN and AASIST are wired in.
    speaker_score = 0.92
    pad_score = 0.99
    accepted = speaker_score >= 0.80 and pad_score >= 0.90 and phrase_match_score >= 0.80

    return {
        "accepted": accepted,
        "speaker_score": speaker_score,
        "pad_score": pad_score,
        "phrase_match": phrase_match_score,
        "transcript": transcript.strip(),
        "duration": round(info.duration, 2),
    }
