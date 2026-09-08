"""Cited RAG and foundational intelligence workflows."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from app.adapters import (
    FallbackRetrievalAdapter,
    LocalRetrievalAdapter,
    OpenSearchRetrievalAdapter,
    RetrievalAdapter,
)
from app.config import Settings
from app.grounding import authorized_evidence, enforce_statement_grounding
from app.models import (
    AnalystDecisionRequest,
    ChangeEvent,
    CitedStatement,
    EntityCandidate,
    Evidence,
    GenerationMode,
    GroundingStatus,
    IntelligenceEntity,
    IntelligenceQuery,
    IntelligenceResponse,
    PrincipalContext,
    Relationship,
    ReviewDecision,
)
from ingestion.fixtures import FIXTURE_DOCUMENT_COUNT

MISSION_EVIDENCE = (
    Evidence(
        evidence_id="demo:mission-planning:1",
        source="Synthetic J2 fixture",
        title="Mission planning reference",
        excerpt=(
            "Approved mission planning products preserve source provenance, identify "
            "information gaps, and require analyst review before publication."
        ),
        document_id="mission-planning-reference",
        version="1",
        acl_principals=["public"],
    ),
    Evidence(
        evidence_id="demo:foundational-data:1",
        source="Synthetic J2 fixture",
        title="Foundational data quality guide",
        excerpt=(
            "Entity records should retain aliases, relationships, effective dates, "
            "source citations, and a visible history of analyst-approved changes."
        ),
        document_id="foundational-data-guide",
        version="1",
        acl_principals=["group:mission-analysts"],
    ),
    Evidence(
        evidence_id="demo:restricted-cell:1",
        source="Synthetic restricted fixture",
        title="Restricted cell exercise note",
        excerpt="This synthetic note is visible only to the restricted-cell group.",
        document_id="restricted-cell-note",
        version="1",
        security_label="demo",
        acl_principals=["group:restricted-cell"],
    ),
)


def ingestion_operational_status(settings: Settings) -> dict[str, object]:
    durable_documents: int | None = None
    if settings.document_registry_table:
        try:
            import boto3

            table = boto3.resource(
                "dynamodb",
                region_name=settings.aws_region,
            ).Table(settings.document_registry_table)
            durable_documents = int(
                table.scan(
                    Select="COUNT",
                    FilterExpression=("attribute_not_exists(deleted) OR deleted = :false"),
                    ExpressionAttributeValues={":false": False},
                ).get("Count", 0)
            )
        except Exception:
            durable_documents = None
    return {
        "security_domain": settings.security_domain,
        "fixture_documents_expected": FIXTURE_DOCUMENT_COUNT,
        "durable_documents": durable_documents,
        "durable_store_configured": bool(settings.document_registry_table),
        "graph_connector_configured": bool(settings.graph_connector_secret_arn),
    }


class LocalMissionRetrievalAdapter(LocalRetrievalAdapter):
    """Small local corpus used when the deployed enterprise index is unavailable."""

    def retrieve(
        self,
        queries: Sequence[str],
        principal: PrincipalContext | None = None,
        filters: Any | None = None,
    ) -> list[Evidence]:
        tokens = set(re.findall(r"[a-z0-9]+", " ".join(queries).casefold()))
        ranked: list[tuple[int, Evidence]] = []
        for item in MISSION_EVIDENCE:
            if principal and not (set(item.acl_principals) & principal.acl_principals):
                continue
            if principal and item.security_label not in {
                "public",
                principal.security_domain,
            }:
                continue
            if filters and filters.document_ids and item.document_id not in filters.document_ids:
                continue
            item_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    f"{item.title} {item.excerpt}".casefold(),
                )
            )
            score = len(tokens & item_tokens)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].evidence_id))
        return [item.model_copy(deep=True) for _, item in ranked[:8]]


class CitedGenerationService:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval or self._build_retrieval()

    def _build_retrieval(self) -> RetrievalAdapter:
        fallback = LocalMissionRetrievalAdapter()
        if not self.settings.opensearch_endpoint:
            return fallback
        enterprise_settings = self.settings.model_copy(
            update={"opensearch_index": self.settings.enterprise_index}
        )
        return FallbackRetrievalAdapter(
            OpenSearchRetrievalAdapter(enterprise_settings),
            fallback=fallback,
        )

    def generate(
        self,
        request: IntelligenceQuery,
        principal: PrincipalContext,
    ) -> IntelligenceResponse:
        retrieved = self.retrieval.retrieve(
            [request.query],
            principal=principal,
            filters=request.filters,
        )
        evidence = authorized_evidence(retrieved, principal)[: self.settings.max_rag_evidence]
        if not evidence:
            return IntelligenceResponse(
                query=request.query,
                mode=request.mode,
                workflow=request.workflow,
                answer="No authorized source supports a response to this request.",
                statements=[],
                evidence=[],
                synthesis_mode="no-authorized-evidence",
            )
        statements, mode = self._synthesize(request, evidence)
        grounded = enforce_statement_grounding(statements, evidence, principal)
        verified = [
            statement.text
            for statement in grounded
            if statement.grounding_status == GroundingStatus.VERIFIED
        ]
        answer = "\n\n".join(verified)
        if not answer:
            answer = "Retrieved sources did not support a grounded response."
        return IntelligenceResponse(
            query=request.query,
            mode=request.mode,
            workflow=request.workflow,
            answer=answer,
            statements=grounded,
            evidence=evidence,
            synthesis_mode=mode,
        )

    def _synthesize(
        self,
        request: IntelligenceQuery,
        evidence: list[Evidence],
    ) -> tuple[list[CitedStatement], str]:
        if self.settings.bedrock_enabled:
            try:
                return self._remote_synthesis(request, evidence), (
                    f"bedrock-mantle:{self.settings.bedrock_model_id}"
                )
            except Exception:
                pass
        prefix = {
            GenerationMode.ANSWER: "Source-grounded answer",
            GenerationMode.SUMMARY: "Source summary",
            GenerationMode.DRAFT: "Draft for analyst review",
        }[request.mode]
        statements = [
            CitedStatement(
                text=f"{prefix}: {item.excerpt}",
                citation_ids=[item.evidence_id],
                grounding_status=GroundingStatus.VERIFIED,
            )
            for item in evidence[:4]
        ]
        return statements, "offline-extractive"

    def _remote_synthesis(
        self,
        request: IntelligenceQuery,
        evidence: list[Evidence],
    ) -> list[CitedStatement]:
        import boto3
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        prompt = {
            "instruction": (
                "Return JSON containing statements, each with text and citation_ids. "
                "Use only supplied evidence IDs. Do not add unsupported assertions. "
                "Draft content must be explicitly suitable for analyst review."
            ),
            "request": request.model_dump(mode="json"),
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        url = f"https://bedrock-mantle.{self.settings.aws_region}.api.aws/openai/v1/responses"
        body = json.dumps(
            {
                "model": self.settings.bedrock_model_id,
                "input": json.dumps(prompt, separators=(",", ":")),
                "max_output_tokens": 2_000,
                "store": False,
            },
            separators=(",", ":"),
        )
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials are unavailable")
        credentials = credentials.get_frozen_credentials()
        signed = AWSRequest(
            method="POST",
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(credentials, "bedrock-mantle", self.settings.aws_region).add_auth(signed)
        response = requests.post(
            url,
            data=body,
            headers=dict(signed.headers),
            timeout=self.settings.bedrock_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        output_text = _extract_response_text(payload)
        generated = json.loads(output_text)
        statements = generated.get("statements", [])
        if not isinstance(statements, list):
            raise ValueError("Model response did not contain statements")
        return [CitedStatement.model_validate(item) for item in statements]


class IntelligenceRepository:
    """Thread-safe local workflow repository; AWS deployments can replace this adapter."""

    def __init__(
        self,
        known_citation_ids: set[str] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._known_citation_ids = known_citation_ids or {
            item.evidence_id for item in MISSION_EVIDENCE
        }
        self._entities: dict[str, IntelligenceEntity] = {}
        self._entity_keys: dict[tuple[str, str], str] = {}
        self._changes: list[ChangeEvent] = []
        self._relationships: dict[str, Relationship] = {}

    def resolve(self, candidates: Sequence[EntityCandidate]) -> list[IntelligenceEntity]:
        resolved: list[IntelligenceEntity] = []
        with self._lock:
            for candidate in candidates:
                normalized = _normalize_name(candidate.name)
                key = (candidate.entity_type.casefold(), normalized)
                entity_id = self._entity_keys.get(key)
                if entity_id is None:
                    entity = IntelligenceEntity(
                        canonical_name=candidate.name.strip(),
                        normalized_name=normalized,
                        entity_type=candidate.entity_type.strip(),
                        attributes=dict(candidate.attributes),
                        aliases=list(dict.fromkeys(candidate.aliases)),
                        provenance=list(candidate.provenance),
                    )
                    self._entities[entity.entity_id] = entity
                    self._entity_keys[key] = entity.entity_id
                else:
                    previous = self._entities[entity_id]
                    merged_attributes = {**previous.attributes, **candidate.attributes}
                    changed_fields = sorted(
                        field
                        for field, value in merged_attributes.items()
                        if previous.attributes.get(field) != value
                    )
                    entity = previous.model_copy(
                        update={
                            "attributes": merged_attributes,
                            "aliases": list(
                                dict.fromkeys(
                                    [
                                        *previous.aliases,
                                        candidate.name,
                                        *candidate.aliases,
                                    ]
                                )
                            ),
                            "provenance": _merge_provenance(
                                previous.provenance,
                                candidate.provenance,
                            ),
                            "updated_at": datetime.now(UTC),
                            "review_status": ReviewDecision.DRAFT,
                        }
                    )
                    self._entities[entity_id] = entity
                    if changed_fields:
                        self._changes.append(
                            ChangeEvent(
                                entity_id=entity_id,
                                changed_fields=changed_fields,
                                before={
                                    field: previous.attributes.get(field)
                                    for field in changed_fields
                                },
                                after={
                                    field: merged_attributes.get(field) for field in changed_fields
                                },
                                provenance=list(candidate.provenance),
                            )
                        )
                resolved.append(entity.model_copy(deep=True))
            self._resolve_relationships(candidates)
            resolved = [self._entities[item.entity_id].model_copy(deep=True) for item in resolved]
        return resolved

    def _resolve_relationships(
        self,
        candidates: Sequence[EntityCandidate],
    ) -> None:
        for candidate in candidates:
            source_id = self._entity_keys.get(
                (candidate.entity_type.casefold(), _normalize_name(candidate.name))
            )
            if source_id is None:
                continue
            for relationship_spec in candidate.relationships:
                relation, separator, target_name = relationship_spec.partition(":")
                if not separator or not relation.strip() or not target_name.strip():
                    continue
                target_id = next(
                    (
                        entity_id
                        for (_entity_type, normalized), entity_id in self._entity_keys.items()
                        if normalized == _normalize_name(target_name)
                    ),
                    None,
                )
                if target_id is None:
                    continue
                citation_ids = [
                    item.citation_id
                    for item in candidate.provenance
                    if item.citation_id in self._known_citation_ids
                ]
                existing = next(
                    (
                        item
                        for item in self._relationships.values()
                        if item.source_entity_id == source_id
                        and item.target_entity_id == target_id
                        and item.relationship_type == relation.strip()
                    ),
                    None,
                )
                relationship = existing or Relationship(
                    source_entity_id=source_id,
                    relationship_type=relation.strip(),
                    target_entity_id=target_id,
                    citation_ids=list(dict.fromkeys(citation_ids)),
                    grounding_status=(
                        GroundingStatus.VERIFIED if citation_ids else GroundingStatus.UNVERIFIED
                    ),
                )
                self._relationships[relationship.relationship_id] = relationship
                source = self._entities[source_id]
                self._entities[source_id] = source.model_copy(
                    update={
                        "relationship_ids": list(
                            dict.fromkeys(
                                [
                                    *source.relationship_ids,
                                    relationship.relationship_id,
                                ]
                            )
                        )
                    }
                )

    def list_entities(self) -> list[IntelligenceEntity]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._entities.values()]

    def list_changes(self) -> list[ChangeEvent]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._changes]

    def list_relationships(self) -> list[Relationship]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._relationships.values()]

    def decide_entity(
        self,
        entity_id: str,
        decision: AnalystDecisionRequest,
        principal: PrincipalContext,
    ) -> IntelligenceEntity:
        if "entities:review" not in principal.scopes:
            raise PermissionError("entities:review scope is required")
        if decision.decision == ReviewDecision.DRAFT:
            raise ValueError("Analyst decision must approve or reject the entity")
        with self._lock:
            entity = self._entities.get(entity_id)
            if entity is None:
                raise KeyError("Entity not found")
            updated = entity.model_copy(
                update={
                    "review_status": decision.decision,
                    "reviewed_by": principal.subject,
                    "review_note": decision.note,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._entities[entity_id] = updated
            return updated.model_copy(deep=True)

    def register_evidence(self, evidence: Sequence[Evidence]) -> None:
        with self._lock:
            self._known_citation_ids.update(
                item.evidence_id for item in evidence if item.evidence_id
            )


class DynamoDBIntelligenceRepository(IntelligenceRepository):
    """Durable AWS adapter using opaque JSON payloads and operational IDs only."""

    def __init__(
        self,
        entity_table: str,
        change_table: str,
        region: str,
        resource: Any | None = None,
    ) -> None:
        super().__init__()
        if resource is None:
            import boto3

            resource = boto3.resource("dynamodb", region_name=region)
        self._entity_table = resource.Table(entity_table)
        self._change_table = resource.Table(change_table)
        self._hydrate()

    def _hydrate(self) -> None:
        for item in self._scan(self._entity_table):
            payload = item.get("payload")
            if not isinstance(payload, str):
                continue
            entity = IntelligenceEntity.model_validate_json(payload)
            self._entities[entity.entity_id] = entity
            self._entity_keys[(entity.entity_type.casefold(), entity.normalized_name)] = (
                entity.entity_id
            )
        for item in self._scan(self._change_table):
            payload = item.get("payload")
            if isinstance(payload, str):
                self._changes.append(ChangeEvent.model_validate_json(payload))

    @staticmethod
    def _scan(table: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        response = table.scan(ProjectionExpression="payload")
        items.extend(response.get("Items", []))
        while response.get("LastEvaluatedKey"):
            response = table.scan(
                ProjectionExpression="payload",
                ExclusiveStartKey=response["LastEvaluatedKey"],
            )
            items.extend(response.get("Items", []))
        return items

    def resolve(
        self,
        candidates: Sequence[EntityCandidate],
    ) -> list[IntelligenceEntity]:
        previous_changes = len(self._changes)
        entities = super().resolve(candidates)
        for entity in entities:
            self._entity_table.put_item(
                Item={
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "normalized_name": entity.normalized_name,
                    "payload": entity.model_dump_json(),
                }
            )
        for change in self._changes[previous_changes:]:
            self._change_table.put_item(
                Item={
                    "change_id": change.change_id,
                    "entity_id": change.entity_id,
                    "detected_at": change.detected_at.isoformat(),
                    "payload": change.model_dump_json(),
                }
            )
        return entities

    def decide_entity(
        self,
        entity_id: str,
        decision: AnalystDecisionRequest,
        principal: PrincipalContext,
    ) -> IntelligenceEntity:
        entity = super().decide_entity(entity_id, decision, principal)
        self._entity_table.put_item(
            Item={
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "normalized_name": entity.normalized_name,
                "payload": entity.model_dump_json(),
            }
        )
        return entity


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _merge_provenance(
    existing: Sequence[Any],
    incoming: Sequence[Any],
) -> list[Any]:
    merged = {
        (item.citation_id, item.document_id, item.version): item for item in [*existing, *incoming]
    }
    return list(merged.values())


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for output in payload.get("output", []):
        for content in output.get("content", []):
            if isinstance(content.get("text"), str):
                return content["text"]
    raise ValueError("Model response did not contain output text")
