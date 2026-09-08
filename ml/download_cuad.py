"""Explicit, checksum-aware CUAD downloader."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str = "") -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "government-contract-review-demo/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual = sha256(partial)
        if expected_sha256 and actual.casefold() != expected_sha256.casefold():
            raise ValueError(f"checksum mismatch: expected {expected_sha256}, got {actual}")
        partial.replace(destination)
        return actual
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CUAD only when explicitly invoked.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("ml/data/cuad.zip"))
    parser.add_argument("--sha256", default="", help="Expected checksum; strongly recommended")
    parser.add_argument("--accept-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_license:
        parser.error("pass --accept-license after reviewing ml/CUAD_ATTRIBUTION.md")
    print(f"downloaded {args.output} (sha256={download(args.url, args.output, args.sha256)})")


if __name__ == "__main__":
    main()
