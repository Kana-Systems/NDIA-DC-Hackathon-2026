import io
import json

import pytest
from botocore.credentials import Credentials

from app.adapters import (
    BedrockGPT56TerraSynthesisAdapter,
    FallbackRetrievalAdapter,
    LocalRetrievalAdapter,
    OpenSearchRetrievalAdapter,
    PackagedMLClassificationAdapter,
    YamlRuleEngineAdapter,
    dfars_7012_applicable,
)
from app.config import Settings
from app.models import Evidence, RuleFinding, Severity
from app.parsers import DocumentParser
from app.sample import sample_contract_bytes, sample_metadata


def _sample_document():
    settings = Settings()
    return DocumentParser(settings).parse("sample.docx", sample_contract_bytes())


def test_packaged_classifier_maps_labels_to_source_locations() -> None:
    def predictor(texts: list[str], threshold: float):
        assert len(texts) == 6
        assert threshold == 0.6
        return [
            *[{"labels": []} for _ in range(5)],
            {
                "model_id": "test-classifier",
                "labels": [{"label": "termination", "score": 0.91}],
            },
        ]

    settings = Settings(classifier_enabled=True, classifier_threshold=0.6)
    clauses = PackagedMLClassificationAdapter(settings, predictor=predictor).classify(
        _sample_document()
    )
    modeled = next(item for item in clauses if item.clause_id == "FAR 52.249")

    assert modeled.confidence == 0.91
    assert modeled.location == _sample_document().segments[5].location
    assert modeled.classifier_model_id == "test-classifier"


def test_packaged_classifier_falls_back_when_inference_fails() -> None:
    def predictor(_texts: list[str], _threshold: float):
        raise RuntimeError("offline")

    clauses = PackagedMLClassificationAdapter(
        Settings(classifier_enabled=True),
        predictor=predictor,
    ).classify(_sample_document())

    assert any(item.clause_id == "FAR 52.249" for item in clauses)


@pytest.mark.parametrize(
    ("label", "text", "expected_id"),
    [
        (
            "termination_for_convenience",
            "Either party may terminate the agreement for convenience.",
            "TRIAGE:termination",
        ),
        (
            "ip_ownership_assignment",
            "Supplier assigns all intellectual property to Customer.",
            "TRIAGE:intellectual-property",
        ),
        (
            "joint_ip_ownership",
            "The parties jointly own all IP developed under the agreement.",
            "TRIAGE:intellectual-property",
        ),
        (
            "license_grant",
            "Licensor grants Customer a worldwide license to use the software.",
            "TRIAGE:intellectual-property",
        ),
    ],
)
def test_realistic_cuad_labels_map_to_structural_triage_candidates(
    label: str,
    text: str,
    expected_id: str,
) -> None:
    document = _sample_document().model_copy(
        update={"segments": [_sample_document().segments[0].model_copy(update={"text": text})]}
    )

    def predictor(_texts: list[str], _threshold: float):
        return [
            {
                "model_id": "cuad-transformer",
                "labels": [{"label": label, "score": 0.93}],
            }
        ]

    clauses = PackagedMLClassificationAdapter(
        Settings(classifier_domain_mapping_path="missing-mapping.json"),
        predictor=predictor,
    ).classify(document)

    mapped = next(item for item in clauses if item.clause_id == expected_id)
    assert mapped.classifier_model_id == "cuad-transformer"
    assert mapped.confidence == 0.93


def test_classifier_consumes_configured_domain_mapping(tmp_path) -> None:
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {"mappings": {"custom_license_category": {"domain_labels": ["intellectual_property"]}}}
        ),
        encoding="utf-8",
    )
    document = _sample_document().model_copy(
        update={
            "segments": [
                _sample_document()
                .segments[0]
                .model_copy(
                    update={"text": "Licensor grants Customer a perpetual software license."}
                )
            ]
        }
    )

    def predictor(_texts: list[str], _threshold: float):
        return [
            {
                "model_id": "custom-model",
                "labels": [{"label": "custom_license_category", "score": 0.88}],
            }
        ]

    clauses = PackagedMLClassificationAdapter(
        Settings(classifier_domain_mapping_path=str(mapping)),
        predictor=predictor,
    ).classify(document)

    assert any(item.clause_id == "TRIAGE:intellectual-property" for item in clauses)


def test_sam_and_dfars_data_rights_structural_patterns_are_supported() -> None:
    document = _sample_document().model_copy(
        update={
            "segments": [
                _sample_document()
                .segments[0]
                .model_copy(
                    update={
                        "text": (
                            "The offeror shall be registered in the System for Award "
                            "Management. The Government receives government purpose "
                            "rights in technical data."
                        )
                    }
                )
            ]
        }
    )

    clauses = PackagedMLClassificationAdapter(Settings(classifier_enabled=False)).classify(document)

    assert {item.clause_id for item in clauses} >= {
        "FAR 52.204-7",
        "DFARS 252.227",
    }


def test_negated_bare_clause_words_are_not_detected() -> None:
    document = _sample_document().model_copy(
        update={
            "segments": [
                _sample_document()
                .segments[0]
                .model_copy(
                    update={
                        "text": (
                            "There is no termination clause and no disputes clause; "
                            "this does not include FAR 52.249-2."
                        )
                    }
                )
            ]
        }
    )

    clauses = PackagedMLClassificationAdapter(Settings()).classify(document)

    assert not clauses


@pytest.mark.parametrize(
    "text",
    [
        "FAR 52.249-2 does not apply.",
        "DFARS 252.204-7012 is not applicable.",
        "FAR 52.204-7 is omitted.",
    ],
)
def test_postfix_non_applicability_is_not_clause_presence(text: str) -> None:
    document = _sample_document().model_copy(
        update={"segments": [_sample_document().segments[0].model_copy(update={"text": text})]}
    )

    clauses = PackagedMLClassificationAdapter(Settings()).classify(document)

    assert not clauses


def test_bare_covered_defense_information_is_not_clause_presence() -> None:
    document = _sample_document().model_copy(
        update={
            "segments": [
                _sample_document()
                .segments[0]
                .model_copy(
                    update={"text": "The system may contain covered defense information and CUI."}
                )
            ]
        }
    )

    clauses = PackagedMLClassificationAdapter(Settings()).classify(document)

    assert not any(item.clause_id == "DFARS 252.204-7012" for item in clauses)


def test_military_department_is_dod_without_substring_guessing() -> None:
    military = sample_metadata().model_copy(update={"agency": "Department of the Air Force"})
    evaluation = military.model_copy(update={"acquisition_stage": "evaluation"})
    civilian = sample_metadata().model_copy(update={"agency": "Civilian Defense Research Office"})
    cots = military.model_copy(update={"commercial_product": True, "cots_only": True})

    assert dfars_7012_applicable(military) is True
    assert dfars_7012_applicable(evaluation) is True
    assert dfars_7012_applicable(civilian) is False
    assert dfars_7012_applicable(cots) is False


def test_cots_only_requires_commercial_product() -> None:
    payload = sample_metadata().model_dump()
    payload.update({"commercial_product": False, "cots_only": True})

    with pytest.raises(ValueError, match="cots_only requires commercial_product"):
        type(sample_metadata()).model_validate(payload)


class FakeSearch:
    def __init__(self) -> None:
        self.bodies = []

    def search(self, *, index: str, body: dict):
        assert index == "knowledge"
        self.bodies.append(body)
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "record-1",
                        "_source": {
                            "citation_id": "far:52-204-7:1",
                            "authority": "FAR",
                            "document_title": "FAR 52.204-7",
                            "heading": "System for Award Management",
                            "text": "Offerors must be registered in SAM.",
                            "url": "https://www.acquisition.gov/far/52.204-7",
                        },
                    }
                ]
            }
        }


class FakeBedrock:
    def invoke_model(self, **kwargs):
        request = json.loads(kwargs["body"])
        assert request["dimensions"] == 1024
        assert request["normalize"] is True
        return {"body": io.BytesIO(json.dumps({"embedding": [0.0] * 1024}).encode())}


def test_opensearch_uses_bm25_and_optional_titan_vector() -> None:
    search = FakeSearch()
    settings = Settings(
        opensearch_endpoint="https://search.example.gov",
        opensearch_index="knowledge",
        opensearch_vector_enabled=True,
    )

    evidence = OpenSearchRetrievalAdapter(
        settings,
        client=search,
        bedrock_client=FakeBedrock(),
    ).retrieve(["SAM registration"])

    assert "multi_match" in search.bodies[0]["query"]
    assert len(search.bodies[1]["query"]["knn"]["embedding"]["vector"]) == 1024
    assert evidence[0].evidence_id == "far:52-204-7:1"
    assert evidence[0].source == "FAR"


def test_opensearch_rrf_retains_disjoint_vector_results() -> None:
    class DisjointSearch:
        def __init__(self):
            self.calls = 0

        def search(self, **_kwargs):
            self.calls += 1
            prefix = "lexical" if self.calls == 1 else "vector"
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": f"{prefix}-{index}",
                            "_source": {
                                "citation_id": f"{prefix}-{index}",
                                "authority": "FAR",
                                "heading": f"{prefix} result",
                                "text": f"{prefix} evidence {index}",
                            },
                        }
                        for index in range(1, 3)
                    ]
                }
            }

    settings = Settings(
        opensearch_endpoint="https://search.example.gov",
        opensearch_index="knowledge",
        opensearch_vector_enabled=True,
        opensearch_top_k=2,
    )
    evidence = OpenSearchRetrievalAdapter(
        settings,
        client=DisjointSearch(),
        bedrock_client=FakeBedrock(),
    ).retrieve(["test"])

    assert {item.evidence_id for item in evidence} == {"lexical-1", "vector-1"}


def test_remote_retrieval_failure_uses_local_evidence() -> None:
    class BrokenRetrieval:
        def retrieve(self, _queries):
            raise RuntimeError("unavailable")

    evidence = FallbackRetrievalAdapter(
        BrokenRetrieval(),
        LocalRetrievalAdapter(),
    ).retrieve(["FAR payment"])

    assert evidence


def test_packaged_yaml_rules_are_loaded() -> None:
    settings = Settings(rules_path="knowledge/rules/review_rules.yaml")
    findings = YamlRuleEngineAdapter(settings).evaluate(
        _sample_document(),
        sample_metadata(),
        clauses=[],
    )

    assert any(item.rule_id == "sam-registration-far-52-204-7" for item in findings)


def _rule_and_evidence():
    rule = RuleFinding(
        rule_id="R-1",
        title="Payment timing",
        severity=Severity.HIGH,
        description="Payment timing needs review.",
        recommendation="Review payment terms.",
        evidence_queries=["FAR payment"],
    )
    evidence = Evidence(
        evidence_id="EV-FAR-32",
        source="FAR",
        title="FAR Part 32",
        excerpt="Contract payment policy.",
    )
    return rule, evidence


def test_remote_synthesis_tolerates_code_fences_and_missing_optional_fields() -> None:
    rule, evidence = _rule_and_evidence()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": (
                    '```json\n{"findings":[{"rule_id":"R-1","citation_ids":["EV-FAR-32"]}]}\n```'
                )
            }

    class Transport:
        def post(self, url, *, data, headers, timeout):
            assert url == ("https://bedrock-mantle.us-gov-west-1.api.aws/openai/v1/responses")
            payload = json.loads(data)
            assert payload == {
                "model": "openai.gpt-5.6-terra",
                "input": payload["input"],
                "max_output_tokens": 3000,
                "store": False,
            }
            assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
            assert "Bearer" not in headers["Authorization"]
            assert timeout == 25
            return Response()

    class RefreshableCredentials:
        def __init__(self) -> None:
            self.frozen = False

        def get_frozen_credentials(self):
            self.frozen = True
            return Credentials("access", "secret", "token")

    credentials = RefreshableCredentials()
    summary, findings, mode = BedrockGPT56TerraSynthesisAdapter(
        Settings(bedrock_enabled=True),
        transport=Transport(),
        credentials=credentials,
    ).synthesize([rule], [evidence])

    assert mode == "bedrock-gpt-5.6-terra"
    assert summary
    assert findings[0].title == rule.title
    assert findings[0].severity == rule.severity
    assert credentials.frozen is True


def test_malformed_remote_synthesis_uses_offline_fallback() -> None:
    rule, evidence = _rule_and_evidence()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"output": [{"content": [{"type": "output_text", "text": "not json"}]}]}

    class Transport:
        def post(self, *_args, **_kwargs):
            return Response()

    _, findings, mode = BedrockGPT56TerraSynthesisAdapter(
        Settings(bedrock_enabled=True),
        transport=Transport(),
        credentials=Credentials("access", "secret"),
    ).synthesize([rule], [evidence])

    assert mode == "offline-deterministic"
    assert findings[0].citation_ids == ["EV-FAR-32"]


def test_nested_responses_output_content_is_parsed() -> None:
    rule, evidence = _rule_and_evidence()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    '{"executive_summary":"Nested response.",'
                                    '"findings":[{"rule_id":"R-1",'
                                    '"citation_ids":["EV-FAR-32"]}]}'
                                ),
                            }
                        ],
                    }
                ]
            }

    class Transport:
        def post(self, *_args, **_kwargs):
            return Response()

    summary, findings, mode = BedrockGPT56TerraSynthesisAdapter(
        Settings(bedrock_enabled=True),
        transport=Transport(),
        credentials=Credentials("access", "secret"),
    ).synthesize([rule], [evidence])

    assert summary == "Nested response."
    assert findings[0].citation_ids == ["EV-FAR-32"]
    assert mode == "bedrock-gpt-5.6-terra"
