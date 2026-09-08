"""Deterministic J2 ingestion, grounding, ACL, and entity benchmark."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from app.config import Settings
from app.grounding import authorized_evidence
from app.intelligence import CitedGenerationService, IntelligenceRepository
from app.models import (
    EntityCandidate,
    Evidence,
    GroundingStatus,
    IntelligenceQuery,
    PrincipalContext,
    ProvenanceLink,
)
from ingestion.connectors import FilesystemConnector
from ingestion.fixtures import FIXTURE_DOCUMENT_COUNT, generate_fixture_corpus
from ingestion.pipeline import ingest
from ingestion.store import InMemoryIngestionStore


def run() -> dict[str, object]:
    principal = PrincipalContext(
        subject="benchmark-analyst",
        groups=["mission-analysts"],
        security_domain="demo",
        scopes=["rag:query", "entities:read", "entities:review"],
    )
    with tempfile.TemporaryDirectory() as directory:
        generate_fixture_corpus(directory)
        connector = FilesystemConnector(
            directory,
            corpus="j2-public-synthetic",
            security_label="public",
            acl_principals=("public",),
        )
        store = InMemoryIngestionStore()
        first = ingest(connector, store)
        second = ingest(connector, store, cursor=first.cursor)
        fixture_records = list(store.records.values())

    query_tokens = {"syn", "042"}
    ranked = sorted(
        fixture_records,
        key=lambda record: (
            -len(query_tokens & set(re.findall(r"[a-z0-9]+", record.text.casefold()))),
            record.record_id,
        ),
    )
    recall_at_5 = float(
        any(record.document_id == "enterprise-brief-042.txt" for record in ranked[:5])
    )
    public_evidence = [
        Evidence(
            evidence_id=record.record_id,
            source="synthetic fixture",
            title=record.document_id,
            excerpt=record.text,
            document_id=record.document_id,
            security_label=record.security_label,
            acl_principals=list(record.acl_principals),
        )
        for record in fixture_records[:5]
    ]
    restricted = Evidence(
        evidence_id="restricted",
        source="synthetic fixture",
        title="Restricted",
        excerpt="Synthetic restricted content.",
        security_label="demo",
        acl_principals=["group:restricted-cell"],
    )
    authorized = authorized_evidence([*public_evidence, restricted], principal)
    acl_leakage_count = sum(item.evidence_id == "restricted" for item in authorized)

    benchmark_settings = Settings(
        bedrock_enabled=False,
        workspace_password="benchmark-only-password",
        demo_jwt_secret="benchmark-only-signing-secret-32-characters",
    )
    rag = CitedGenerationService(benchmark_settings).generate(
        IntelligenceQuery(query="foundational data provenance analyst review"),
        principal,
    )
    citation_correctness = (
        sum(
            statement.grounding_status == GroundingStatus.VERIFIED and bool(statement.citation_ids)
            for statement in rag.statements
        )
        / len(rag.statements)
        if rag.statements
        else 0.0
    )

    repository = IntelligenceRepository()
    provenance = [
        ProvenanceLink(
            citation_id="demo:foundational-data:1",
            document_id="foundational-data-guide",
            version="1",
        )
    ]
    repository.resolve(
        [
            EntityCandidate(
                name="Benchmark Organization",
                entity_type="organization",
                attributes={"status": "active"},
                provenance=provenance,
            )
        ]
    )[0]
    repository.resolve(
        [
            EntityCandidate(
                name="benchmark-organization",
                entity_type="organization",
                attributes={"status": "inactive"},
                provenance=provenance,
            )
        ]
    )
    metrics = {
        "fixture_documents": first.applied.inserted,
        "fixture_records": len(fixture_records),
        "idempotent_second_run_changes": second.changes,
        "retrieval_recall_at_5": recall_at_5,
        "citation_correctness": citation_correctness,
        "acl_leakage_count": acl_leakage_count,
        "change_events": len(repository.list_changes()),
    }
    hard_failures = (
        metrics["fixture_documents"] < FIXTURE_DOCUMENT_COUNT
        or metrics["idempotent_second_run_changes"] != 0
        or metrics["retrieval_recall_at_5"] < 1.0
        or metrics["citation_correctness"] < 1.0
        or metrics["acl_leakage_count"] != 0
        or metrics["change_events"] < 1
    )
    return {"passed": not hard_failures, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
