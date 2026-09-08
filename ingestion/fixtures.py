"""Deterministic public/synthetic enterprise corpus generation."""

from __future__ import annotations

from pathlib import Path

FIXTURE_DOCUMENT_COUNT = 120


def generate_fixture_corpus(
    destination: str | Path,
    *,
    count: int = FIXTURE_DOCUMENT_COUNT,
) -> tuple[Path, ...]:
    """Generate stable, non-sensitive documents and return their paths."""
    if count < FIXTURE_DOCUMENT_COUNT:
        raise ValueError(f"fixture corpus must contain at least {FIXTURE_DOCUMENT_COUNT} documents")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for number in range(1, count + 1):
        path = root / f"enterprise-brief-{number:03d}.txt"
        text = (
            f"PUBLIC SYNTHETIC ENTERPRISE BRIEF {number:03d}\n\n"
            f"Document identifier: SYN-{number:03d}\n"
            f"Portfolio: J2 intelligence readiness cohort {(number - 1) % 12 + 1:02d}\n"
            f"Region: synthetic-sector-{(number - 1) % 8 + 1:02d}\n\n"
            "This generated record contains fictional, unclassified data for deterministic "
            "ingestion and access-control testing. It is not operational intelligence.\n"
        )
        path.write_text(text, encoding="utf-8", newline="\n")
        paths.append(path)
    return tuple(paths)
