# aurix-ml

Reference implementation of the Aurix voice verification backend.

> **Status:** early scaffold. Whisper (ASR) is real; speaker verification
> and PAD are placeholders to be replaced with ECAPA-TDNN and AASIST.

## What it does

1. Receives an audio sample and the expected phrase.
2. Converts it to 16 kHz mono WAV with `ffmpeg`.
3. Trims leading/trailing silence.
4. Transcribes it with `faster-whisper` (Polish).
5. Compares the transcript with the expected phrase (`rapidfuzz`).

## Run standalone

```bash
docker compose up -d --build
curl http://localhost:8010/health
```

The Whisper model is downloaded on the first `/v1/verify` request and cached
in the `whisper_models` volume.

## Example

```bash
curl -X POST http://localhost:8010/v1/verify \
  -H "X-Internal-Token: dev-token" \
  -F "audio=@sample.webm" \
  -F "expected_phrase=kawa most zima trzy siedem"
```

```json
{
  "accepted": true,
  "speaker_score": 0.92,
  "pad_score": 0.99,
  "phrase_match": 0.95,
  "transcript": "kawa most zima trzy siedem",
  "duration": 2.8
}
```

## Integration with aurix

The web application calls this service over the internal Docker network:

```
http://ml:8000/v1/verify
```

It is not exposed publicly in the main compose stack.
