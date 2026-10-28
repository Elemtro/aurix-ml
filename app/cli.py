"""Run a local verification pass without starting FastAPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.main import _run_verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a probe recording against a speaker reference.")
    parser.add_argument("--audio", required=True, type=Path, help="New probe recording (WAV, WebM, MP4, etc.)")
    parser.add_argument("--reference", required=True, type=Path, help="Previously recorded reference of the same speaker")
    parser.add_argument("--phrase", required=True, help="Expected phrase spoken in --audio")
    args = parser.parse_args()

    result = _run_verification(
        args.audio.read_bytes(),
        args.audio.name,
        args.reference.read_bytes(),
        args.reference.name,
        args.phrase,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
