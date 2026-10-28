"""Check expected phrase and speaker decisions on the four local recording pairs."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT / "testdata" / "real_audio_tests"
CASES = {
    "p1": {
        "phrase": "chmura rower książka wiatr",
        "speaker_match": True,
        "phrase_match": True,
        "accepted": True,
    },
    "p2": {
        "phrase": "fortuna często rozumowi przeczy",
        "speaker_match": True,
        "phrase_match": False,
        "accepted": False,
    },
    "p3": {
        "phrase": "gdzie drwa rąbią, tam wióry lecą",
        "speaker_match": True,
        "phrase_match": True,
        "accepted": True,
    },
    "p4": {
        "phrase": "w nocy wszystkie koty są szare",
        "speaker_match": False,
        "phrase_match": True,
        "accepted": False,
    },
}


def verify(base_url: str, token: str, probe: Path, reference: Path | None, phrase: str):
    boundary = f"----Aurix{uuid.uuid4().hex}"
    chunks = []
    uploads = [("audio", probe)]
    if reference is not None:
        uploads.append(("reference_audio", reference))
    for name, path in uploads:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode(),
                b"Content-Type: application/octet-stream\r\n\r\n",
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"expected_phrase\"\r\n\r\n".encode(),
            phrase.encode("utf-8"),
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        f"{base_url}/v1/verify",
        data=b"".join(chunks),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Internal-Token": token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except (ValueError, json.JSONDecodeError):
            return error.code, {"detail": error.read().decode("utf-8", errors="replace")}
    except (TimeoutError, urllib.error.URLError) as error:
        return 0, {"detail": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("AURIX_ML_URL", "http://localhost:8010"))
    parser.add_argument("--token", default=os.getenv("INTERNAL_TOKEN", "dev-token"))
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    missing = [
        str(AUDIO_DIR / f"{name}{suffix}.aac")
        for name in CASES
        for suffix in ("", "_r")
        if not (AUDIO_DIR / f"{name}{suffix}.aac").is_file()
    ]
    if missing:
        parser.error("Missing audio files: " + ", ".join(missing))

    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
            health = json.load(response)
    except Exception as error:
        parser.error(f"API unavailable at {base_url}: {error}")
    if health.get("status") != "ok":
        parser.error(f"API is not ready: {health}")
    if health.get("pad_enforced", False):
        parser.error(
            "PAD_ENFORCED is enabled. These four recordings can check phrase/speaker "
            "behavior, but cannot calibrate spoof detection. Set PAD_ENFORCED=false for this test."
        )

    # aurix-main currently sends only audio and expected_phrase. Keep that API
    # contract usable, while failing closed until speaker enrollment exists.
    status, body = verify(
        base_url,
        args.token,
        AUDIO_DIR / "p1.aac",
        None,
        CASES["p1"]["phrase"],
    )
    contract_keys = {"accepted", "speaker_score", "pad_score", "phrase_match", "transcript", "duration"}
    contract_passed = (
        status == 200
        and contract_keys.issubset(body)
        and body.get("accepted") is False
        and body.get("speaker_score") == 0.0
    )
    print(
        f"{'PASS' if contract_passed else 'FAIL'} API compatibility: HTTP {status}; "
        f"legacy_fields={contract_keys.issubset(body)}; "
        f"missing_reference_rejected={body.get('accepted') is False}"
    )
    contract_status = status
    contract_body = body

    results = []
    for index, (name, case) in enumerate(CASES.items(), start=1):
        started = time.monotonic()
        status, body = verify(
            base_url,
            args.token,
            AUDIO_DIR / f"{name}.aac",
            AUDIO_DIR / f"{name}_r.aac",
            case["phrase"],
        )
        actual = {
            "speaker_match": body.get("speaker_score", 0) >= body.get("speaker_threshold", 1),
            "phrase_match": body.get("phrase_match", 0) >= body.get("phrase_threshold", 1),
            "accepted": body.get("accepted") is True,
        }
        expected = {key: value for key, value in case.items() if key != "phrase"}
        passed = status == 200 and actual == expected
        detail = body.get("transcript", body.get("detail", ""))
        print(
            f"{'PASS' if passed else 'FAIL'} {index}/4 {name}: HTTP {status}; "
            f"accepted={actual['accepted']} (expected={case['accepted']}); "
            f"speaker_match={actual['speaker_match']} (expected={case['speaker_match']}); "
            f"phrase_match={actual['phrase_match']} (expected={case['phrase_match']}); "
            f"speaker={body.get('speaker_score')} "
            f"phrase={body.get('phrase_match')} pad={body.get('pad_score')} "
            f"pad_decision={body.get('pad_decision')} transcript={detail}; "
            f"{time.monotonic() - started:.1f}s"
        )
        results.append(
            {
                "id": name,
                "expected_phrase": case["phrase"],
                "expected": {key: value for key, value in case.items() if key != "phrase"},
                "actual": actual,
                "http_status": status,
                "passed": passed,
                "response": body,
            }
        )

    report = {
        "health_before": health,
        "aurix_main_contract": {
            "passed": contract_passed,
            "http_status": contract_status,
            "response": contract_body,
        },
        "results": results,
    }
    report_path = AUDIO_DIR / "results.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    passed_count = sum(result["passed"] for result in results)
    total_passed = passed_count + int(contract_passed)
    print(f"Wynik: {total_passed}/5 sprawdzeń; raport: {report_path}")
    return 0 if total_passed == len(CASES) + 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
