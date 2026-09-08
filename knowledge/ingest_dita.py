"""Ingest FAR/DFARS DITA topics into canonical source records."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree

from knowledge.schema import SourceRecord, write_jsonl


def _tag(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower()


def _text(element: ElementTree.Element | None) -> str:
    return " ".join(" ".join(element.itertext()).split()) if element is not None else ""


def _first(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next((child for child in element.iter() if _tag(child) == name), None)


def parse_topic(
    path: Path,
    *,
    authority: str,
    document_id: str,
    document_title: str,
    base_url: str = "",
) -> list[SourceRecord]:
    """Parse one DITA/XML file, emitting one record per section or topic body."""

    root = ElementTree.parse(path).getroot()
    topic = (
        root
        if _tag(root) in {"topic", "concept", "reference", "task"}
        else next(
            (
                element
                for element in root.iter()
                if _tag(element) in {"topic", "concept", "reference", "task"}
            ),
            None,
        )
    )
    if topic is None:
        return []
    topic_id = topic.get("id") or path.stem
    topic_title = _text(_first(topic, "title")) or topic_id
    sections = [element for element in topic.iter() if _tag(element) in {"section", "refbody"}]
    # FAR conbody often mixes the main clause's p/ol children with Alternate
    # sections. Selecting only sections silently discarded the main clause.
    if sections:
        body = next(
            (
                element
                for element in topic.iter()
                if _tag(element) in {"body", "conbody", "taskbody"}
            ),
            None,
        )
        if body is not None:
            remainder = deepcopy(body)

            def remove_sections(element):
                for child in list(element):
                    if _tag(child) in {"section", "refbody"}:
                        element.remove(child)
                    else:
                        remove_sections(child)

            remove_sections(remainder)
            if _text(remainder):
                remainder.set("id", f"{topic_id}-main")
                sections.insert(0, remainder)
    if not sections:
        body = next(
            (
                element
                for element in topic.iter()
                if _tag(element) in {"body", "conbody", "taskbody"}
            ),
            topic,
        )
        sections = [body]

    records: list[SourceRecord] = []
    for position, section in enumerate(sections, start=1):
        section_id = section.get("id") or (
            topic_id if len(sections) == 1 else f"{topic_id}-s{position}"
        )
        heading = _text(_first(section, "title")) or topic_title
        text = _text(section)
        if text and text != heading:
            records.append(
                SourceRecord(
                    source_type="official_dita",
                    authority=authority,
                    document_title=document_title,
                    document_id=document_id,
                    locator=section_id,
                    heading=heading,
                    text=text,
                    url=f"{base_url.rstrip('/')}/{path.name}" if base_url else "",
                    metadata={"source_file": path.name, "topic_id": topic_id},
                ).normalized()
            )
    return records


def ingest_path(
    source: Path,
    *,
    authority: str,
    document_id: str,
    document_title: str,
    base_url: str = "",
) -> list[SourceRecord]:
    """Ingest a DITA/XML file, directory, or ZIP archive."""

    if zipfile.is_zipfile(source):
        with tempfile.TemporaryDirectory() as directory:
            with zipfile.ZipFile(source) as archive:
                for item in archive.infolist():
                    target = Path(directory, item.filename).resolve()
                    if not target.is_relative_to(Path(directory).resolve()):
                        raise ValueError(f"unsafe archive member: {item.filename}")
                archive.extractall(directory)
            return ingest_path(
                Path(directory),
                authority=authority,
                document_id=document_id,
                document_title=document_title,
                base_url=base_url,
            )
    paths = (
        [source]
        if source.is_file()
        else sorted(path for path in source.rglob("*") if path.suffix.lower() in {".dita", ".xml"})
    )
    records: list[SourceRecord] = []
    for path in paths:
        records.extend(
            parse_topic(
                path,
                authority=authority,
                document_id=document_id,
                document_title=document_title,
                base_url=base_url,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output")
    parser.add_argument("--authority", required=True, choices=("FAR", "DFARS"))
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--document-title", required=True)
    parser.add_argument("--base-url", default="")
    args = parser.parse_args()
    records = ingest_path(
        args.source,
        authority=args.authority,
        document_id=args.document_id,
        document_title=args.document_title,
        base_url=args.base_url,
    )
    print(f"wrote {write_jsonl(records, args.output)} records to {args.output}")


if __name__ == "__main__":
    main()
