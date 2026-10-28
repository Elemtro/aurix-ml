# aurix-ml

Reference implementation of the Aurix voice verification backend.

> **Status:** the service now runs Whisper (ASR), Silero VAD, SpeechBrain ECAPA-TDNN speaker embeddings, and AASIST PAD. A speaker reference is required for a real speaker comparison. PAD is diagnostic by default until calibrated on representative genuine and spoof recordings.

## What it does

1. Receives an audio sample, the expected phrase, and an optional speaker reference.
2. Decodes common audio/video formats with ffmpeg to mono, 16 kHz PCM.
3. Detects speech and measures basic signal quality with Silero VAD.
4. Transcribes speech in Polish with faster-whisper and compares words with rapidfuzz.
5. When reference_audio is supplied, compares SpeechBrain ECAPA-TDNN speaker embeddings with cosine similarity.
6. Reports an AASIST bona-fide score. The default PAD_ENFORCED=false keeps this score out of the accept/reject decision because the current recordings show false rejects.

Without reference_audio, the API remains compatible with the current aurix-main request: it returns HTTP 200 and the documented score fields, sets speaker_score to 0.0, and rejects the attempt. The service does not invent an enrolled speaker score.

## Run standalone

Use Docker Desktop with Docker Compose:

~~~bash
docker compose up -d --build
curl http://localhost:8010/health
~~~

The image includes ffmpeg and Python dependencies. Model weights are downloaded lazily on first use and stored in the named ml_models volume. docker compose logs -f ml shows service logs.

## Verify an audio file

The original request shape used by aurix-main remains supported:

~~~bash
curl -X POST http://localhost:8010/v1/verify \
  -H "X-Internal-Token: dev-token" \
  -F "audio=@sample.webm" \
  -F "expected_phrase=kawa most zima trzy siedem"
~~~

To compare a speaker as well, include a clean reference recording of that speaker:

~~~bash
curl -X POST http://localhost:8010/v1/verify \
  -H "X-Internal-Token: dev-token" \
  -F "audio=@sample.webm" \
  -F "reference_audio=@reference.wav" \
  -F "expected_phrase=kawa most zima trzy siedem"
~~~

The response preserves the original fields:

~~~json
{
  "accepted": true,
  "speaker_score": 0.92,
  "pad_score": 0.42,
  "phrase_match": 1.0,
  "transcript": "kawa most zima trzy siedem",
  "duration": 2.8
}
~~~

Additional diagnostics include raw speaker cosine, thresholds, quality metrics, PAD enforcement status, reasons, warnings, and model load status. speaker_score is the cosine similarity mapped to 0–1. If there is no reference, it is 0.0. AASIST output is model confidence, not a calibrated probability.

## Models and calibration

- **Silero VAD** detects speech and supplies the segment used for scoring; it does not denoise.
- **SpeechBrain ECAPA-TDNN** creates speaker embeddings. The current API compares a new recording with reference_audio; persistent speaker enrollment is not implemented.
- **Cosine similarity** compares the two normalized embeddings. PLDA is not enabled because it needs a multi-speaker calibration set.
- **AASIST ONNX** provides the PAD score. Set PAD_ENFORCED=true to apply PAD_THRESHOLD as a hard gate after calibration. The current user recordings are genuine-only examples and cannot validate spoof detection.
- **faster-whisper** transcribes the expected challenge phrase in Polish.

## Integration with aurix

The web application calls the service on its internal Docker network at:

~~~text
http://ml:8000/v1/verify
~~~

The endpoint accepts the fields currently sent by aurix-main (audio, expected_phrase) and returns accepted, speaker_score, pad_score, phrase_match, transcript, and duration. The web caller currently sends no speaker reference, so the API returns speaker_score: 0.0 and fails closed.

### Changes needed in aurix-main for reference audio

1. Add a server-side speaker enrollment/profile store. Save a reference recording or, preferably, an ECAPA embedding for the signed-in user. Do not let the browser choose another user's reference.
2. Pass the resolved reference path through VerifyController.php and VoiceEngineInterface.php to HttpVoiceEngine.php.
3. In HttpVoiceEngine.php, add a reference path to verify() and send it as multipart field reference_audio:

~~~php
'body' => [
    'audio' => fopen($audioPath, 'r'),
    'reference_audio' => fopen($referenceAudioPath, 'r'),
    'expected_phrase' => $expectedPhrase,
],
~~~

Keep the field name reference_audio exactly; the Python endpoint accepts WAV and common compressed audio formats and compares the enrolled voice with the new recording. If enrollment is not available for an account, reject the attempt instead of sending an arbitrary reference.

aurix-main currently applies its own fixed PAD threshold of 0.90. Since PAD is diagnostic by default and the current genuine recordings score below that threshold, the current Symfony client will reject those attempts until its decision policy is aligned with PAD calibration. The Python endpoint remains HTTP/schema compatible; successful end-to-end speaker verification requires the enrollment and threshold work above.
