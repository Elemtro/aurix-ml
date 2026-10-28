"""Lazy model adapters for Silero VAD, ECAPA-TDNN and AASIST PAD."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AASIST_REPO = "SpeechAntiSpoofingBenchmarks/AASIST"
AASIST_REVISION = "16774d458d86d2a021ae31646c1bf66a5331b53e"
AASIST_WINDOW_SAMPLES = 64_600


def model_cache() -> Path:
    path = Path(os.getenv("MODEL_CACHE", "/models"))
    path.mkdir(parents=True, exist_ok=True)
    # Keep all model artifacts and Hugging Face's Xet logs inside the mounted cache.
    os.environ.setdefault("HF_HOME", str(path / "huggingface_home"))
    os.environ.setdefault("HF_XET_CACHE", str(path / "huggingface_home" / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return path


class VoiceModels:
    """Keeps one copy of each model resident for the lifetime of the process."""

    def __init__(self) -> None:
        requested = os.getenv("MODEL_DEVICE", "cpu").lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE requests CUDA, but CUDA is unavailable.")
        self.device = requested
        self._vad: torch.nn.Module | None = None
        self._speaker: object | None = None
        self._pad_session: object | None = None
        self._lock = threading.RLock()

    def get_vad(self) -> torch.nn.Module:
        with self._lock:
            if self._vad is None:
                from silero_vad import load_silero_vad

                self._vad = load_silero_vad(onnx=False).to(self.device).eval()
            return self._vad

    def _get_speaker_model(self):
        with self._lock:
            if self._speaker is None:
                cache = model_cache()
                from speechbrain.inference.speaker import SpeakerRecognition
                from speechbrain.utils.fetching import LocalStrategy

                self._speaker = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=str(cache / "speechbrain_ecapa_voxceleb"),
                    run_opts={"device": self.device},
                    local_strategy=LocalStrategy.COPY,
                )
            return self._speaker

    def speaker_embedding(self, samples: np.ndarray) -> torch.Tensor:
        model = self._get_speaker_model()
        waveform = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            embedding = model.encode_batch(waveform)
            embedding = F.normalize(embedding.reshape(1, -1), p=2, dim=1)
        return embedding.detach().cpu()

    def cosine_similarity(self, reference: np.ndarray, probe: np.ndarray) -> float:
        reference_embedding = self.speaker_embedding(reference)
        probe_embedding = self.speaker_embedding(probe)
        return float(F.cosine_similarity(reference_embedding, probe_embedding).item())

    def _get_pad_session(self):
        with self._lock:
            if self._pad_session is None:
                cache = model_cache()
                try:
                    import onnxruntime as ort
                    from huggingface_hub import hf_hub_download
                except ImportError as exc:
                    raise RuntimeError(
                        "Install onnxruntime and huggingface_hub to run AASIST PAD."
                    ) from exc

                model_path = hf_hub_download(
                    repo_id=AASIST_REPO,
                    filename="aasist.onnx",
                    revision=os.getenv("AASIST_REVISION", AASIST_REVISION),
                    cache_dir=str(cache / "huggingface"),
                )
                self._pad_session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"],
                )
            return self._pad_session

    @staticmethod
    def _fixed_aasist_window(samples: np.ndarray) -> np.ndarray:
        """Match the published evaluation wrapper's 64,600-sample window."""
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise ValueError("Cannot score an empty waveform with AASIST.")
        if waveform.size < AASIST_WINDOW_SAMPLES:
            repeats = int(np.ceil(AASIST_WINDOW_SAMPLES / waveform.size))
            waveform = np.tile(waveform, repeats)
        return np.ascontiguousarray(waveform[:AASIST_WINDOW_SAMPLES], dtype=np.float32)

    def pad_score(self, samples: np.ndarray) -> float:
        """Return an AASIST bona-fide score in [0, 1] (higher means more live)."""
        session = self._get_pad_session()
        model_input = session.get_inputs()[0]
        waveform = self._fixed_aasist_window(samples)
        rank = len(model_input.shape)
        if rank == 3:
            waveform = waveform[None, None, :]
        elif rank == 2:
            waveform = waveform[None, :]
        elif rank == 1:
            pass
        else:
            raise RuntimeError(f"Unexpected AASIST ONNX input shape: {model_input.shape}")

        outputs = session.run(None, {model_input.name: waveform})
        result = next((np.asarray(output) for output in outputs if np.asarray(output).size in (1, 2)), None)
        if result is None:
            raise RuntimeError("AASIST returned an unsupported output shape.")
        logits = result.astype(np.float64).reshape(-1)
        if logits.size == 2:
            logits -= logits.max()
            probabilities = np.exp(logits) / np.exp(logits).sum()
            # Official clovaai labels: class 1 is bona fide, class 0 is spoof.
            score = float(probabilities[1])
        else:
            value = float(logits[0])
            score = value if 0.0 <= value <= 1.0 else 1.0 / (1.0 + np.exp(-value))
        return float(np.clip(score, 0.0, 1.0))

    def status(self) -> dict[str, bool | str]:
        return {
            "device": self.device,
            "vad_loaded": self._vad is not None,
            "ecapa_loaded": self._speaker is not None,
            "aasist_loaded": self._pad_session is not None,
        }


_models: VoiceModels | None = None
_models_lock = threading.Lock()


def get_voice_models() -> VoiceModels:
    global _models
    with _models_lock:
        if _models is None:
            _models = VoiceModels()
        return _models
