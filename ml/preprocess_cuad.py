"""Convert CUAD's SQuAD-style annotations to deterministic JSONL classification data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from ml.windowing import (
    DEFAULT_WINDOW_CHARS,
    DEFAULT_WINDOW_STRIDE,
    answer_centered_bounds,
    sliding_bounds,
    validate_window_parameters,
)

DOMAIN_MAPPING = Path(__file__).with_name("cuad_category_domain_mapping.json")


def _normalized_category(value: str) -> str:
    value = re.sub(r"[_-]\d+$", "", value.strip())
    return "_".join(value.casefold().replace("/", " ").replace("-", " ").split())


def cuad_category(question: str, qa_id: str = "") -> str:
    """Extract the CUAD category, preferring its canonical QA ID suffix."""

    if "__" in qa_id:
        category = qa_id.rsplit("__", 1)[1]
    else:
        quoted = question.split('"')
        category = quoted[1] if len(quoted) >= 3 else question.split(":", 1)[0]
    return _normalized_category(category)


def load_cuad(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = sorted(name for name in archive.namelist() if name.lower().endswith(".json"))
            preferred = [name for name in names if "cuad" in name.casefold()]
            if not (preferred or names):
                raise ValueError("archive contains no JSON file")
            with archive.open((preferred or names)[0]) as stream:
                return json.load(stream)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _answer_spans(paragraph: dict[str, Any]) -> list[dict[str, Any]]:
    context = str(paragraph.get("context", ""))
    spans = []
    for qa in paragraph.get("qas", []):
        label = cuad_category(qa.get("question", ""), qa.get("id", ""))
        for answer in qa.get("answers", []):
            answer_text = str(answer.get("text", ""))
            answer_start = answer.get("answer_start")
            if not label or not answer_text:
                continue
            if not isinstance(answer_start, int):
                raise ValueError(f"answer_start must be an integer for {qa.get('id', '')}")
            answer_end = answer_start + len(answer_text)
            if not (
                0 <= answer_start < answer_end <= len(context)
                and context[answer_start:answer_end] == answer_text
            ):
                raise ValueError(f"answer span does not match context for {qa.get('id', '')}")
            spans.append(
                {
                    "label": label,
                    "start": answer_start,
                    "end": answer_end,
                    "text": answer_text,
                }
            )
    return sorted(spans, key=lambda span: (span["start"], span["end"], span["label"]))


def _paragraph_windows(
    document_id: str,
    paragraph_number: int,
    paragraph: dict[str, Any],
    window_chars: int,
    stride_chars: int,
    negative_windows_per_positive: int,
) -> list[dict[str, Any]]:
    context = str(paragraph.get("context", ""))
    spans = _answer_spans(paragraph)
    positive_by_bounds: dict[tuple[int, int], dict[str, Any]] = {}
    for span in spans:
        bounds = answer_centered_bounds(
            len(context),
            span["start"],
            span["end"],
            window_chars,
        )
        labels = sorted(
            {
                candidate["label"]
                for candidate in spans
                if bounds[0] <= candidate["start"] and candidate["end"] <= bounds[1]
            }
        )
        positive_by_bounds[bounds] = {
            "start_char": bounds[0],
            "end_char": bounds[1],
            "text": context[bounds[0] : bounds[1]],
            "labels": labels,
            "is_negative": False,
        }

    negative_candidates = [
        (start, end)
        for start, end in sliding_bounds(len(context), window_chars, stride_chars)
        if not any(start < span["end"] and span["start"] < end for span in spans)
    ]
    negative_limit = (
        max(1, len(positive_by_bounds) * negative_windows_per_positive)
        if negative_windows_per_positive
        else 0
    )
    negatives = [
        {
            "start_char": start,
            "end_char": end,
            "text": context[start:end],
            "labels": [],
            "is_negative": True,
        }
        for start, end in negative_candidates[:negative_limit]
    ]
    windows = [*positive_by_bounds.values(), *negatives]
    windows.sort(key=lambda item: (item["start_char"], item["end_char"], item["is_negative"]))
    for window_number, window in enumerate(windows):
        window["id"] = f"{document_id}:{paragraph_number}:w{window_number:04d}"
        window["document_id"] = document_id
        window["paragraph_number"] = paragraph_number
    return windows


def examples(
    dataset: dict[str, Any],
    window_chars: int = DEFAULT_WINDOW_CHARS,
    stride_chars: int = DEFAULT_WINDOW_STRIDE,
    negative_windows_per_positive: int = 1,
) -> list[dict[str, Any]]:
    validate_window_parameters(window_chars, stride_chars)
    if negative_windows_per_positive < 0:
        raise ValueError("negative_windows_per_positive cannot be negative")
    output = []
    for document in dataset.get("data", []):
        document_id = document.get("title", "")
        for paragraph_number, paragraph in enumerate(document.get("paragraphs", [])):
            output.extend(
                _paragraph_windows(
                    document_id,
                    paragraph_number,
                    paragraph,
                    window_chars,
                    stride_chars,
                    negative_windows_per_positive,
                )
            )
    return output


def split(
    items: list[dict[str, Any]], validation_percent: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, validation = [], []
    for item in items:
        bucket = int(hashlib.sha256(item["document_id"].encode("utf-8")).hexdigest()[:8], 16) % 100
        (validation if bucket < validation_percent else train).append(item)
    return train, validation


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="CUAD JSON or downloaded ZIP")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data/processed"))
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--window-chars", type=int, default=DEFAULT_WINDOW_CHARS)
    parser.add_argument("--stride-chars", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--negative-windows-per-positive", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.validation_percent <= 50:
        parser.error("--validation-percent must be between 1 and 50")
    try:
        items = examples(
            load_cuad(args.source),
            args.window_chars,
            args.stride_chars,
            args.negative_windows_per_positive,
        )
    except ValueError as error:
        parser.error(str(error))
    train, validation = split(items, args.validation_percent)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    labels = sorted({label for item in train + validation for label in item["labels"]})
    (args.output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )
    window_config = {
        "schema_version": "1.0",
        "strategy": "answer_centered_positive_sliding_negative",
        "window_chars": args.window_chars,
        "stride_chars": args.stride_chars,
        "negative_windows_per_positive": args.negative_windows_per_positive,
    }
    (args.output_dir / "window_config.json").write_text(
        json.dumps(window_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(DOMAIN_MAPPING, args.output_dir / DOMAIN_MAPPING.name)
    print(f"wrote {len(train)} train, {len(validation)} validation, {len(labels)} labels")


if __name__ == "__main__":
    main()
