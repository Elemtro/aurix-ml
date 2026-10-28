"""FastAPI service for phrase checking, speaker verification and PAD."""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.audio import AudioAssessment, assess_speech, decode_audio, write_wav
from app.models import get_voice_models

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_CACHE = os.getenv("WHISPER_CACHE", str(Path(os.getenv("MODEL_CACHE", "/models")) / "whisper"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "pl")
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(32 * 1024 * 1024)))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "30"))
MIN_SPEECH_SECONDS = float(os.getenv("MIN_SPEECH_SECONDS", "0.75"))
SPEAKER_THRESHOLD = float(os.getenv("SPEAKER_THRESHOLD", "0.80"))
PAD_THRESHOLD = float(os.getenv("PAD_THRESHOLD", "0.90"))
PHRASE_THRESHOLD = float(os.getenv("PHRASE_THRESHOLD", "0.80"))
PAD_ENFORCED = os.getenv("PAD_ENFORCED", "false").strip().lower() in {"1", "true", "yes", "on"}

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

app = FastAPI(title="Aurix ML", version="0.2.0")
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    """Load Whisper only when the first verification is requested."""
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=os.getenv("WHISPER_DEVICE", "cpu"),
                compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                download_root=WHISPER_CACHE,
            )
        return _whisper_model


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(DIGIT_WORDS.get(token, token) for token in text.split())


def phrase_match(expected: str, transcript: str) -> float:
    """Fraction of challenge words found in the transcript using fuzzy matching."""
    from rapidfuzz import fuzz

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


def _save_upload(data: bytes, filename: str, directory: Path, stem: str) -> Path:
    suffix = Path(filename or "").suffix[:12]
    target = directory / f"{stem}{suffix}"
    target.write_bytes(data)
    return target


def _quality_payload(assessment: AudioAssessment) -> dict:
    return {
        "speech_duration": round(assessment.speech_duration, 3),
        "speech_coverage": round(assessment.speech_coverage, 3),
        "speech_dbfs": round(assessment.speech_dbfs, 2),
        "noise_dbfs": None if assessment.noise_dbfs is None else round(assessment.noise_dbfs, 2),
        "estimated_snr_db": None
        if assessment.estimated_snr_db is None
        else round(assessment.estimated_snr_db, 2),
        "clipped_fraction": round(assessment.clipped_fraction, 5),
    }


def _run_verification(
    audio_bytes: bytes,
    audio_filename: str,
    reference_bytes: bytes | None,
    reference_filename: str | None,
    expected_phrase: str,
) -> dict:
    from app.audio import SAMPLE_RATE

    with tempfile.TemporaryDirectory(prefix="aurix-ml-") as temporary_directory:
        directory = Path(temporary_directory)
        source = _save_upload(audio_bytes, audio_filename, directory, "probe")
        reference_source = (
            _save_upload(reference_bytes, reference_filename or "reference.audio", directory, "reference")
            if reference_bytes is not None
            else None
        )
        # Keep decoded PCM paths separate from uploaded paths. A WAV upload is
        # already named ``probe.wav``/``reference.wav`` and ffmpeg refuses to
        # overwrite its own input when source and destination are identical.
        probe_wav = directory / "probe-normalized.wav"
        reference_wav = directory / "reference-normalized.wav"
        trimmed_wav = directory / "probe-speech.wav"

        try:
            probe = decode_audio(source, probe_wav, max_duration=MAX_AUDIO_SECONDS + 1)
            reference = (
                decode_audio(reference_source, reference_wav, max_duration=MAX_AUDIO_SECONDS + 1)
                if reference_source is not None
                else None
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            models = get_voice_models()
            probe_assessment = assess_speech(probe, models.get_vad())
            reference_assessment = (
                assess_speech(reference, models.get_vad()) if reference is not None else None
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"VAD model failed to load or run: {exc}") from exc

        if probe_assessment.speech_duration < MIN_SPEECH_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"At least {MIN_SPEECH_SECONDS:.2f} seconds of speech are required.",
            )
        if reference_assessment is not None and reference_assessment.speech_duration < MIN_SPEECH_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"Reference audio needs at least {MIN_SPEECH_SECONDS:.2f} seconds of speech.",
            )

        write_wav(trimmed_wav, probe_assessment.samples, SAMPLE_RATE)
        try:
            raw_cosine = (
                max(
                    -1.0,
                    min(1.0, models.cosine_similarity(reference_assessment.samples, probe_assessment.samples)),
                )
                if reference_assessment is not None
                else None
            )
            pad_score = models.pad_score(probe_assessment.samples)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Voice model inference failed: {exc}") from exc

        # The existing Symfony client expects a 0..1 similarity score. Keep raw cosine
        # available too; thresholds must be calibrated on genuine/impostor recordings.
        speaker_score = (
            min(1.0, max(0.0, (raw_cosine + 1.0) / 2.0))
            if raw_cosine is not None
            else 0.0
        )
        try:
            segments, info = get_whisper_model().transcribe(
                str(trimmed_wav),
                language=WHISPER_LANGUAGE or None,
                vad_filter=False,
            )
            transcript = " ".join(segment.text for segment in segments).strip()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Whisper transcription failed: {exc}") from exc

        match_score = phrase_match(expected_phrase, transcript)
        speaker_ok = speaker_score >= SPEAKER_THRESHOLD
        phrase_ok = match_score >= PHRASE_THRESHOLD
        pad_ok = pad_score >= PAD_THRESHOLD
        accepted = speaker_ok and phrase_ok and (pad_ok or not PAD_ENFORCED)
        reasons = []
        if not speaker_ok:
            reasons.append(
                "speaker reference required" if reference_assessment is None else "similarity below threshold"
            )
        if PAD_ENFORCED and not pad_ok:
            reasons.append("spoofing suspected")
        if not phrase_ok:
            reasons.append("phrase mismatch")
        warnings = []
        if not PAD_ENFORCED:
            warnings.append("PAD is diagnostic only; calibrate it before enforcing spoof decisions.")
        pad_decision = "report_only"
        if PAD_ENFORCED:
            pad_decision = "bona_fide" if pad_ok else "spoof_suspected"

        return {
            "accepted": accepted,
            "reasons": reasons,
            "warnings": warnings,
            "speaker_score": round(speaker_score, 5),
            "speaker_cosine": None if raw_cosine is None else round(raw_cosine, 5),
            "speaker_threshold": SPEAKER_THRESHOLD,
            "pad_score": round(pad_score, 5),
            "pad_threshold": PAD_THRESHOLD,
            "pad_enforced": PAD_ENFORCED,
            "pad_decision": pad_decision,
            "phrase_match": round(match_score, 5),
            "phrase_threshold": PHRASE_THRESHOLD,
            "transcript": transcript,
            "duration": round(info.duration, 2),
            "quality": {
                "probe": _quality_payload(probe_assessment),
                "reference": None
                if reference_assessment is None
                else _quality_payload(reference_assessment),
            },
            "models": models.status(),
        }


@app.get("/health")
def health() -> dict:
    model_status = get_voice_models().status()
    return {
        "status": "ok",
        # Keep the scaffold's health response fields for existing clients.
        "model": WHISPER_MODEL,
        "loaded": _whisper_model is not None,
        "whisper_model": WHISPER_MODEL,
        "whisper_loaded": _whisper_model is not None,
        "pad_enforced": PAD_ENFORCED,
        "models": model_status,
    }


@app.post("/v1/verify")
async def verify(
    audio: UploadFile = File(...),
    expected_phrase: str = Form(...),
    reference_audio: UploadFile | None = File(default=None),
    x_internal_token: str | None = Header(default=None),
) -> dict:
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="invalid internal token")

    audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
    reference_bytes = await reference_audio.read(MAX_AUDIO_BYTES + 1) if reference_audio is not None else None
    if not audio_bytes or (reference_audio is not None and not reference_bytes):
        raise HTTPException(status_code=400, detail="audio must not be empty; reference_audio cannot be empty")
    if len(audio_bytes) > MAX_AUDIO_BYTES or (
        reference_bytes is not None and len(reference_bytes) > MAX_AUDIO_BYTES
    ):
        limit_mb = max(1, MAX_AUDIO_BYTES // (1024 * 1024))
        raise HTTPException(status_code=413, detail=f"audio files must not exceed {limit_mb} MB")

    return await run_in_threadpool(
        _run_verification,
        audio_bytes,
        audio.filename or "probe.audio",
        reference_bytes,
        reference_audio.filename if reference_audio is not None else None,
        expected_phrase,
    )
