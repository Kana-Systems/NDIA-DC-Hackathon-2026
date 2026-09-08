from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.adapters import OpenSearchRetrievalAdapter
from app.config import Settings
from app.grounding import authorized_evidence, enforce_statement_grounding
from app.intelligence import (
    CitedGenerationService,
    DynamoDBIntelligenceRepository,
    IntelligenceRepository,
)
from app.main import create_app
from app.models import (
    CitedStatement,
    EntityCandidate,
    Evidence,
    GenerationMode,
    GroundingStatus,
    IntelligenceQuery,
    PrincipalContext,
    ProvenanceLink,
)


def _principal(*groups: str) -> PrincipalContext:
    return PrincipalContext(
        subject="analyst",
        groups=list(groups),
        security_domain="demo",
        scopes=[
            "rag:query",
            "entities:read",
            "entities:review",
        ],
    )


def test_authorization_and_grounding_deny_unknown_citations() -> None:
    evidence = [
        Evidence(
            evidence_id="allowed",
            source="fixture",
            title="Allowed",
            excerpt="Allowed evidence.",
            security_label="demo",
            acl_principals=["group:alpha"],
        ),
        Evidence(
            evidence_id="restricted",
            source="fixture",
            title="Restricted",
            excerpt="Restricted evidence.",
            security_label="demo",
            acl_principals=["group:restricted"],
        ),
    ]
    principal = _principal("alpha")

    assert [item.evidence_id for item in authorized_evidence(evidence, principal)] == ["allowed"]
    grounded = enforce_statement_grounding(
        [
            CitedStatement(
                text="Mixed statement",
                citation_ids=["allowed", "restricted", "invented"],
            ),
            CitedStatement(text="Unsupported", citation_ids=["invented"]),
        ],
        evidence,
        principal,
    )
    assert grounded[0].citation_ids == ["allowed"]
    assert grounded[0].grounding_status == GroundingStatus.VERIFIED
    assert grounded[1].citation_ids == []
    assert grounded[1].grounding_status == GroundingStatus.UNVERIFIED


def test_opensearch_applies_principal_acl_and_domain_filters() -> None:
    class SearchClient:
        body: dict = {}

        def search(self, *, index: str, body: dict) -> dict:
            self.body = body
            return {"hits": {"hits": []}}

    client = SearchClient()
    OpenSearchRetrievalAdapter(
        Settings(opensearch_endpoint="https://search.example.test"),
        client=client,
    ).retrieve(["foundational data"], principal=_principal("alpha"))

    filters = client.body["query"]["bool"]["filter"]
    assert {"terms": {"acl_principals": ["group:alpha", "public", "user:analyst"]}} in filters
    assert {"terms": {"security_label": ["public", "demo"]}} in filters
    assert {"term": {"deleted": False}} in filters


def test_cited_generation_returns_only_authorized_sources() -> None:
    response = CitedGenerationService(Settings(bedrock_enabled=False)).generate(
        IntelligenceQuery(
            query="foundational data aliases relationships provenance",
            mode=GenerationMode.SUMMARY,
        ),
        _principal("mission-analysts"),
    )

    assert response.evidence
    assert all(
        set(item.acl_principals) & _principal("mission-analysts").acl_principals
        for item in response.evidence
    )
    assert all(item.grounding_status == GroundingStatus.VERIFIED for item in response.statements)


def test_entity_resolution_detects_changes_and_grounded_relationships() -> None:
    repository = IntelligenceRepository()
    provenance = [
        ProvenanceLink(
            citation_id="demo:foundational-data:1",
            document_id="source-1",
            version="1",
        )
    ]
    first = repository.resolve(
        [
            EntityCandidate(
                name="Example Organization",
                entity_type="organization",
                attributes={"status": "active", "country": "US"},
                provenance=provenance,
            )
        ]
    )[0]
    related = repository.resolve(
        [
            EntityCandidate(
                name="Related Facility",
                entity_type="facility",
                attributes={"status": "active"},
                provenance=provenance,
            ),
            EntityCandidate(
                name="Example Organization",
                entity_type="organization",
                relationships=["operates:Related Facility"],
                provenance=provenance,
            ),
        ]
    )
    assert related[1].relationship_ids
    assert repository.list_relationships()[0].grounding_status == GroundingStatus.VERIFIED

    second = repository.resolve(
        [
            EntityCandidate(
                name="example-organization",
                entity_type="organization",
                attributes={"status": "inactive"},
                provenance=provenance,
            )
        ]
    )[0]

    assert second.entity_id == first.entity_id
    assert repository.list_changes()[0].changed_fields == ["status"]


def test_dynamodb_hydration_ignores_ingestion_changes_without_workflow_payload() -> None:
    class Table:
        def __init__(self, items):
            self.items = items

        def scan(self, **_kwargs):
            return {"Items": self.items}

    class Resource:
        def __init__(self):
            self.tables = {
                "entities": Table([]),
                "changes": Table([{"change_id": "ingestion-change"}]),
            }

        def Table(self, name):
            return self.tables[name]

    repository = DynamoDBIntelligenceRepository(
        entity_table="entities",
        change_table="changes",
        region="us-gov-west-1",
        resource=Resource(),
    )

    assert repository.list_entities() == []
    assert repository.list_changes() == []


def test_demo_token_and_intelligence_api() -> None:
    settings = Settings(
        workspace_username="reviewer",
        workspace_password=SecretStr("test-password"),
        demo_jwt_secret=SecretStr("unit-test-signing-secret-32-characters"),
        bedrock_enabled=False,
    )
    client = TestClient(create_app(settings))
    token_response = client.post(
        "/api/v1/auth/demo-token",
        data={"username": "reviewer", "password": "test-password"},
    )
    assert token_response.status_code == 200

    response = client.post(
        "/api/v1/intelligence/query",
        headers={"Authorization": f"Bearer {token_response.json()['access_token']}"},
        json={
            "query": "mission planning provenance analyst review",
            "mode": "answer",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["evidence"]
    assert payload["statements"][0]["grounding_status"] == "verified"

    status_response = client.get(
        "/api/v1/intelligence/ingestion/status",
        headers={"Authorization": f"Bearer {token_response.json()['access_token']}"},
    )
    assert status_response.status_code == 200
    assert status_response.json()["fixture_documents_expected"] == 120
