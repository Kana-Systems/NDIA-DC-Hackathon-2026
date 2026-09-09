"""Replaceable classification, rule, retrieval, and synthesis adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import yaml

from app.config import Settings
from app.models import (
    AcquisitionMetadata,
    AcquisitionStage,
    ClassifiedClause,
    CorpusFilters,
    Evidence,
    Finding,
    ParsedDocument,
    PrincipalContext,
    RuleFinding,
    Severity,
)

CLAUSE_PATTERNS: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    (
        "FAR 52.212-4",
        "Contract Terms and Conditions - Commercial Products and Services",
        "commercial",
        re.compile(r"\b(?:FAR\s*)?52\.212-4\b", re.I),
    ),
    (
        "FAR 52.233-1",
        "Disputes",
        "disputes",
        re.compile(
            r"\b(?:FAR\s*)?52\.233-1\b|"
            r"\bcontract disputes act\b|"
            r"\bcontracting officer(?:'s)? final decision\b",
            re.I,
        ),
    ),
    (
        "FAR 52.249",
        "Termination",
        "termination",
        re.compile(
            r"\b(?:FAR\s*)?52\.249(?:-\d+)?\b|"
            r"\bgovernment may terminate\b|"
            r"\bterminate (?:this contract|for convenience|for default)\b",
            re.I,
        ),
    ),
    (
        "FAR 52.232",
        "Payments",
        "payment",
        re.compile(
            r"\b(?:FAR\s*)?52\.232(?:-\d+)?\b|"
            r"\bpayment (?:shall|will|is due)\b",
            re.I,
        ),
    ),
    (
        "DFARS 252.204-7012",
        "Safeguarding Covered Defense Information",
        "cybersecurity",
        re.compile(
            r"\b(?:DFARS\s*)?252\.204-7012\b|"
            r"\b(?:covered defense information|CUI) "
            r"(?:shall|must|will) be (?:safeguarded|protected)\b|"
            r"\bcyber incidents? (?:shall|must|will) be reported\b",
            re.I,
        ),
    ),
    (
        "FAR 52.204-7",
        "System for Award Management",
        "registration",
        re.compile(
            r"\b(?:FAR\s*)?52\.204-7\b|"
            r"\b(?:offeror|contractor) (?:shall|must) be registered in "
            r"(?:the )?System for Award Management\b",
            re.I,
        ),
    ),
    (
        "DFARS 252.227",
        "Rights in Technical Data and Computer Software",
        "data_rights",
        re.compile(
            r"\b(?:DFARS\s*)?252\.227-\d{4}\b|"
            r"\b(?:unlimited|government purpose|restricted|limited) rights? "
            r"in (?:technical data|computer software)\b",
            re.I,
        ),
    ),
)

PATTERN_BY_CLAUSE_ID = {
    clause_id: (category, pattern) for clause_id, _title, category, pattern in CLAUSE_PATTERNS
}


def _has_positive_clause_evidence(
    text: str,
    category: str,
    pattern: re.Pattern[str],
) -> bool:
    match = pattern.search(text)
    if not match:
        return False
    negated = {
        "termination": r"\b(?:no|without)\s+(?:\w+\s+){0,2}termination\b",
        "disputes": r"\b(?:no|without)\s+(?:\w+\s+){0,2}disputes?\b",
        "payment": r"\bpayment\s+(?:shall|will|is)\s+not\b",
        "cybersecurity": r"\b(?:no|without)\s+(?:\w+\s+){0,3}(?:CUI|cybersecurity)\b",
        "commercial": r"\b(?:does not include|excludes?)\s+(?:FAR\s*)?52\.212-4\b",
    }
    if re.search(negated.get(category, r"(?!x)x"), text, re.I):
        return False
    prefix_exclusion = re.search(
        rf"\b(?:does not include|excludes?|without)\b.{{0,40}}"
        rf"{re.escape(match.group(0))}",
        text,
        re.I,
    )
    postfix_exclusion = re.search(
        rf"{re.escape(match.group(0))}.{{0,40}}\b"
        rf"(?:does not apply|is not applicable|is (?:hereby )?omitted|shall not apply)\b",
        text,
        re.I,
    )
    return not (prefix_exclusion or postfix_exclusion)


def dfars_7012_applicable(metadata: AcquisitionMetadata) -> bool:
    agency = re.sub(r"\s+", " ", metadata.agency.casefold().strip())
    dod_agencies = {
        "dod",
        "department of defense",
        "u.s. department of defense",
        "department of the army",
        "department of the navy",
        "department of the air force",
        "united states army",
        "united states navy",
        "united states air force",
        "united states marine corps",
        "united states space force",
        "defense logistics agency",
        "defense information systems agency",
        "missile defense agency",
    }
    is_dod = agency in dod_agencies
    contract_stage = metadata.acquisition_stage in {
        AcquisitionStage.SOLICITATION,
        AcquisitionStage.EVALUATION,
        AcquisitionStage.AWARD,
        AcquisitionStage.POST_AWARD,
    }
    return is_dod and contract_stage and not metadata.cots_only


MODEL_LABELS: dict[str, tuple[str, str, str]] = {
    "cybersecurity": (
        "DFARS 252.204-7012",
        "Safeguarding Covered Defense Information",
        "cybersecurity",
    ),
    "sam_registration": (
        "FAR 52.204-7",
        "System for Award Management",
        "registration",
    ),
    "termination": ("FAR 52.249", "Termination", "termination"),
    "data_rights": (
        "DFARS 252.227",
        "Rights in Technical Data",
        "data_rights",
    ),
}

SAFE_CUAD_DOMAIN_MAPPING: dict[str, tuple[str, ...]] = {
    "termination_for_convenience": ("termination",),
    "ip_ownership_assignment": ("intellectual_property", "data_rights"),
    "joint_ip_ownership": ("intellectual_property", "data_rights"),
    "license_grant": ("intellectual_property", "data_rights"),
}

TRIAGE_DOMAIN_CANDIDATES: dict[
    str,
    tuple[str, str, str, re.Pattern[str]],
] = {
    "termination": (
        "TRIAGE:termination",
        "Termination language triage candidate",
        "termination",
        re.compile(
            r"\btermination for convenience\b|"
            r"\b(?:either party|contractor|customer|supplier) "
            r"(?:may|shall have the right to) terminate\b",
            re.I,
        ),
    ),
    "intellectual_property": (
        "TRIAGE:intellectual-property",
        "Intellectual property triage candidate",
        "intellectual_property",
        re.compile(
            r"\b(?:assigns?|shall own|jointly own)\b.{0,80}"
            r"\b(?:intellectual property|IP)\b|"
            r"\b(?:licensor|owner) grants?\b.{0,80}\blicen[cs]e\b",
            re.I,
        ),
    ),
    "data_rights": (
        "TRIAGE:data-rights",
        "Data rights triage candidate",
        "data_rights",
        re.compile(
            r"\b(?:technical data|computer software) rights?\b|"
            r"\b(?:unlimited|government purpose|restricted|limited) rights? "
            r"in (?:technical data|computer software)\b",
            re.I,
        ),
    ),
}


class ClauseClassificationAdapter(Protocol):
    def classify(self, document: ParsedDocument) -> list[ClassifiedClause]: ...


class RuleEngineAdapter(Protocol):
    def evaluate(
        self,
        document: ParsedDocument,
        metadata: AcquisitionMetadata,
        clauses: Sequence[ClassifiedClause],
    ) -> list[RuleFinding]: ...


class RetrievalAdapter(Protocol):
    def retrieve(
        self,
        queries: Sequence[str],
        principal: PrincipalContext | None = None,
        filters: CorpusFilters | None = None,
    ) -> list[Evidence]: ...


class SynthesisAdapter(Protocol):
    def synthesize(
        self,
        rule_findings: Sequence[RuleFinding],
        evidence: Sequence[Evidence],
    ) -> tuple[str, list[Finding], str]: ...


class KeywordClauseClassificationAdapter:
    """Transparent baseline classifier suitable for a deterministic demo."""

    model_ids = ["deterministic-keyword-v1"]

    def classify(self, document: ParsedDocument) -> list[ClassifiedClause]:
        clauses: list[ClassifiedClause] = []
        seen: set[tuple[str, str]] = set()
        for segment in document.segments:
            for clause_id, title, category, pattern in CLAUSE_PATTERNS:
                if (
                    _has_positive_clause_evidence(segment.text, category, pattern)
                    and (clause_id, segment.segment_id) not in seen
                ):
                    seen.add((clause_id, segment.segment_id))
                    clauses.append(
                        ClassifiedClause(
                            clause_id=clause_id,
                            title=title,
                            category=category,
                            confidence=0.98 if clause_id in segment.text.upper() else 0.82,
                            segment_id=segment.segment_id,
                            location=segment.location,
                            excerpt=segment.text[:300],
                            classifier_model_id=self.model_ids[0],
                        )
                    )
        return clauses


class PackagedMLClassificationAdapter:
    """Use the packaged ``ml.inference`` contract, retaining keyword coverage."""

    def __init__(
        self,
        settings: Settings,
        predictor: Callable[[list[str], float], list[dict[str, Any]]] | None = None,
        fallback: ClauseClassificationAdapter | None = None,
    ) -> None:
        self.settings = settings
        self._predictor = predictor
        self.fallback = fallback or KeywordClauseClassificationAdapter()
        self._model: Any | None = None
        self._cuad_mapping = self._load_cuad_mapping()
        self.model_ids = list(getattr(self.fallback, "model_ids", ["deterministic-keyword-v1"]))

    def classify(self, document: ParsedDocument) -> list[ClassifiedClause]:
        baseline = self.fallback.classify(document)
        if not self.settings.classifier_enabled:
            return baseline
        try:
            predictions = self._predict([segment.text for segment in document.segments])
            modeled = self._map_predictions(document, predictions)
        except Exception:
            # Source text and inference failures are intentionally not logged.
            self.model_ids = list(getattr(self.fallback, "model_ids", ["deterministic-keyword-v1"]))
            return baseline
        self.model_ids = list(
            dict.fromkeys(
                [
                    *getattr(
                        self.fallback,
                        "model_ids",
                        ["deterministic-keyword-v1"],
                    )
                ]
                + [
                    str(item.get("model_id") or "packaged-classifier-unknown")
                    for item in predictions
                ]
            )
        )
        combined = {(clause.clause_id, clause.segment_id): clause for clause in baseline}
        combined.update({(clause.clause_id, clause.segment_id): clause for clause in modeled})
        return list(combined.values())

    def _predict(self, texts: list[str]) -> list[dict[str, Any]]:
        if self._predictor:
            return self._predictor(texts, self.settings.classifier_threshold)
        inference = import_module("ml.inference")
        if self._model is None:
            self._model = inference.model_fn(self.settings.classifier_model_dir)
        payload = {"texts": texts, "threshold": self.settings.classifier_threshold}
        result = inference.predict_fn(payload, self._model)
        predictions = result.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(texts):
            raise ValueError("Classifier returned an invalid prediction count")
        return predictions

    def _map_predictions(
        self,
        document: ParsedDocument,
        predictions: Sequence[dict[str, Any]],
    ) -> list[ClassifiedClause]:
        clauses: list[ClassifiedClause] = []
        for segment, prediction in zip(document.segments, predictions, strict=True):
            for label_result in prediction.get("labels", []):
                label = str(label_result.get("label", "")).strip().lower()
                model_id = str(prediction.get("model_id") or "packaged-classifier-unknown")
                candidates: list[tuple[str, str, str, re.Pattern[str]]] = []
                if label in MODEL_LABELS:
                    clause_id, title, category = MODEL_LABELS[label]
                    pattern_entry = PATTERN_BY_CLAUSE_ID.get(clause_id)
                    if pattern_entry:
                        candidates.append((clause_id, title, category, pattern_entry[1]))
                else:
                    candidates.extend(
                        TRIAGE_DOMAIN_CANDIDATES[domain]
                        for domain in self._cuad_mapping.get(label, ())
                        if domain in TRIAGE_DOMAIN_CANDIDATES
                    )
                for clause_id, title, category, pattern in candidates:
                    if not _has_positive_clause_evidence(
                        segment.text,
                        category,
                        pattern,
                    ):
                        continue
                    clauses.append(
                        ClassifiedClause(
                            clause_id=clause_id,
                            title=title,
                            category=category,
                            confidence=float(label_result.get("score", 0)),
                            segment_id=segment.segment_id,
                            location=segment.location,
                            excerpt=segment.text[:300],
                            classifier_model_id=model_id,
                        )
                    )
        return clauses

    def _load_cuad_mapping(self) -> dict[str, tuple[str, ...]]:
        fallback = dict(SAFE_CUAD_DOMAIN_MAPPING)
        path = Path(self.settings.classifier_domain_mapping_path)
        try:
            if not path.is_file() or path.stat().st_size > 1_000_000:
                return fallback
            payload = json.loads(path.read_text(encoding="utf-8"))
            mappings = payload.get("mappings")
            if not isinstance(mappings, dict):
                return fallback
            for label, details in mappings.items():
                if not isinstance(label, str) or not isinstance(details, dict):
                    continue
                domains = details.get("domain_labels")
                if isinstance(domains, list) and all(isinstance(domain, str) for domain in domains):
                    fallback[label.casefold()] = tuple(domain.casefold() for domain in domains)
        except (OSError, ValueError, TypeError):
            return fallback
        return fallback


class DeterministicRuleEngineAdapter:
    """Auditable rules; no model output is trusted to determine severity."""

    def evaluate(
        self,
        document: ParsedDocument,
        metadata: AcquisitionMetadata,
        clauses: Sequence[ClassifiedClause],
    ) -> list[RuleFinding]:
        categories = {clause.category for clause in clauses}
        findings: list[RuleFinding] = []

        if dfars_7012_applicable(metadata) and "cybersecurity" not in categories:
            findings.append(
                RuleFinding(
                    rule_id="GC-004",
                    title="DFARS 252.204-7012 not identified",
                    severity=Severity.HIGH,
                    description=(
                        "The metadata identifies a DoD solicitation/contract that is not "
                        "solely COTS, and DFARS 252.204-7012 was not identified."
                    ),
                    recommendation=(
                        "Confirm covered defense information applicability and clause "
                        "requirements with the contracting officer and counsel."
                    ),
                    evidence_queries=[
                        "DFARS 204.7304(c) prescription 252.204-7012 safeguarding CUI"
                    ],
                )
            )

        risky_patterns = (
            (
                "GC-005",
                re.compile(r"\bunlimited\s+liability\b", re.I),
                "Playbook observation (non-legal): Unlimited liability language",
                Severity.CRITICAL,
                "Negotiate a defined liability cap and explicit carve-outs.",
                "FAR risk allocation liability",
            ),
            (
                "GC-006",
                re.compile(r"\bpayment\b.{0,80}\b(?:60|90|120)\s+days\b", re.I | re.S),
                "Playbook observation (non-legal): Extended payment period",
                Severity.MEDIUM,
                "Reconcile the payment period with Prompt Payment requirements.",
                "FAR 32.9 prompt payment",
            ),
        )
        for rule_id, pattern, title, severity, recommendation, query in risky_patterns:
            for segment in document.segments:
                if pattern.search(segment.text):
                    findings.append(
                        RuleFinding(
                            rule_id=rule_id,
                            title=title,
                            severity=severity,
                            description=(
                                "Non-legal drafting observation based on configured "
                                f"playbook language: “{segment.text[:220]}”"
                            ),
                            recommendation=recommendation,
                            segment_ids=[segment.segment_id],
                            evidence_queries=[query],
                        )
                    )
                    break
        return findings


class YamlRuleEngineAdapter:
    """Apply packaged YAML rules in addition to the conservative built-ins."""

    def __init__(
        self,
        settings: Settings,
        fallback: RuleEngineAdapter | None = None,
    ) -> None:
        self.settings = settings
        self.fallback = fallback or DeterministicRuleEngineAdapter()

    def evaluate(
        self,
        document: ParsedDocument,
        metadata: AcquisitionMetadata,
        clauses: Sequence[ClassifiedClause],
    ) -> list[RuleFinding]:
        findings = self.fallback.evaluate(document, metadata, clauses)
        try:
            configured_rules = self._load_rules()
            findings.extend(self._evaluate_rules(configured_rules, document, metadata))
        except Exception:
            # A missing or malformed optional rules file must not disable safe rules.
            pass
        return findings

    def _load_rules(self) -> list[dict[str, Any]]:
        path = Path(self.settings.rules_path)
        if not path.is_file():
            return []
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("rules"), list):
            raise ValueError("Invalid rule configuration")
        return [item for item in payload["rules"] if isinstance(item, dict)]

    @staticmethod
    def _evaluate_rules(
        rules: Sequence[dict[str, Any]],
        document: ParsedDocument,
        metadata: AcquisitionMetadata,
    ) -> list[RuleFinding]:
        findings: list[RuleFinding] = []
        complete_text = document.text.casefold()
        for rule in rules:
            rule_id = str(rule.get("id", "")).strip()
            title = str(rule.get("title", "")).strip()
            if not rule_id or not title:
                continue
            if rule_id == "cybersecurity-dfars-252-204-7012":
                # The application applies the stricter metadata gate in the built-in engine.
                continue
            severity = Severity(str(rule.get("severity", Severity.MEDIUM.value)))
            applies = rule.get("applies_when", {})
            terms = applies.get("contract_contains_any", []) if isinstance(applies, dict) else []
            matching_segments = [
                segment.segment_id
                for segment in document.segments
                if any(str(term).casefold() in segment.text.casefold() for term in terms)
            ]
            if terms and not matching_segments:
                continue

            required = [str(value) for value in rule.get("required_clauses", [])]
            missing = [value for value in required if value.casefold() not in complete_text]
            detected_segments: list[str] = []
            detect_regex = rule.get("detect_regex")
            expected_regex = rule.get("expected_regex")
            if isinstance(detect_regex, str):
                detector = re.compile(detect_regex)
                expected = (
                    re.compile(str(expected_regex)) if isinstance(expected_regex, str) else None
                )
                detected_segments = [
                    segment.segment_id
                    for segment in document.segments
                    if detector.search(segment.text)
                    and (expected is None or not expected.search(segment.text))
                ]

            if not missing and not detected_segments:
                continue
            description = (
                f"Required clause(s) not identified: {', '.join(missing)}."
                if missing
                else "A clause reference does not match the expected FAR/DFARS format."
            )
            citations = [str(value) for value in rule.get("citations", [])]
            findings.append(
                RuleFinding(
                    rule_id=rule_id,
                    title=(
                        title
                        if rule_id == "malformed-clause-reference"
                        else f"Playbook observation (non-legal): {title}"
                    ),
                    severity=(
                        severity if rule_id == "malformed-clause-reference" else Severity.INFO
                    ),
                    description=description,
                    recommendation="Validate applicability and correct the contract language.",
                    segment_ids=list(dict.fromkeys(matching_segments + detected_segments)),
                    evidence_queries=[title, *citations],
                )
            )
        return findings


EVIDENCE_CATALOG = (
    Evidence(
        evidence_id="EV-FAR-49",
        source="Federal Acquisition Regulation",
        title="FAR Part 49 - Termination of Contracts",
        excerpt=(
            "Part 49 establishes policies and procedures relating to complete "
            "or partial termination."
        ),
        url="https://www.acquisition.gov/far/part-49",
    ),
    Evidence(
        evidence_id="EV-FAR-33",
        source="Federal Acquisition Regulation",
        title="FAR Subpart 33.2 - Disputes and Appeals",
        excerpt="Subpart 33.2 prescribes policies and procedures for disputes and appeals.",
        url="https://www.acquisition.gov/far/subpart-33.2",
    ),
    Evidence(
        evidence_id="EV-FAR-32",
        source="Federal Acquisition Regulation",
        title="FAR Part 32 - Contract Financing",
        excerpt="Part 32 prescribes policies for contract financing and payment.",
        url="https://www.acquisition.gov/far/part-32",
    ),
    Evidence(
        evidence_id="EV-DFARS-7012",
        source="Defense Federal Acquisition Regulation Supplement",
        title="DFARS 252.204-7012",
        excerpt="Safeguarding Covered Defense Information and Cyber Incident Reporting.",
        url=(
            "https://www.acquisition.gov/dfars/"
            "252.204-7012-safeguarding-covered-defense-information-and-"
            "cyber-incident-reporting."
        ),
    ),
    Evidence(
        evidence_id="EV-DFARS-204.7304-C",
        source="Defense Federal Acquisition Regulation Supplement",
        title="DFARS 204.7304(c) - Solicitation provision and contract clauses",
        excerpt=(
            "Use 252.204-7012 in all solicitations and contracts except those "
            "solely for the acquisition of COTS items."
        ),
        url=(
            "https://www.acquisition.gov/dfars/"
            "204.7304-solicitation-provision-and-contract-clauses."
        ),
    ),
    Evidence(
        evidence_id="EV-FAR-RISK",
        source="Federal Acquisition Regulation",
        title="FAR 16.103 - Negotiating Contract Type",
        excerpt="Contract type and risk should provide reasonable contractor risk and incentive.",
        url="https://www.acquisition.gov/far/16.103",
    ),
)


class LocalRetrievalAdapter:
    """Small embedded corpus for offline, traceable retrieval."""

    def retrieve(
        self,
        queries: Sequence[str],
        principal: PrincipalContext | None = None,
        filters: CorpusFilters | None = None,
    ) -> list[Evidence]:
        query_tokens = set(re.findall(r"[a-z0-9.]+", " ".join(queries).lower()))
        ranked: list[tuple[int, Evidence]] = []
        for item in EVIDENCE_CATALOG:
            if principal is not None and not (set(item.acl_principals) & principal.acl_principals):
                continue
            if filters and filters.document_ids and item.document_id not in filters.document_ids:
                continue
            text_tokens = set(
                re.findall(
                    r"[a-z0-9.]+",
                    f"{item.title} {item.excerpt} {item.source}".lower(),
                )
            )
            score = len(query_tokens & text_tokens)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].evidence_id))
        return [item.model_copy(deep=True) for _, item in ranked[:8]]


class OpenSearchRetrievalAdapter:
    """AWS SigV4 OpenSearch retrieval using BM25 and optional Titan vectors."""

    EMBEDDING_DIMENSIONS = 1024

    def __init__(
        self,
        settings: Settings,
        client: Any | None = None,
        bedrock_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client
        self._bedrock_client = bedrock_client

    def retrieve(
        self,
        queries: Sequence[str],
        principal: PrincipalContext | None = None,
        filters: CorpusFilters | None = None,
    ) -> list[Evidence]:
        query = " ".join(value.strip() for value in queries if value.strip())
        if not query:
            return []
        client = self._client or self._create_client()
        access_filter = self._access_filter(principal, filters)
        lexical_query: dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": ["text^3", "heading^2", "document_title"],
                "type": "best_fields",
            }
        }
        if access_filter:
            lexical_query = {
                "bool": {
                    "must": [lexical_query],
                    "filter": access_filter,
                }
            }
        hit_sets = [
            client.search(
                index=self.settings.opensearch_index,
                body={
                    "size": self.settings.opensearch_top_k,
                    "query": lexical_query,
                },
            )
        ]
        if self.settings.opensearch_vector_enabled:
            embedding = self._embed(query)
            hit_sets.append(
                client.search(
                    index=self.settings.opensearch_index,
                    body={
                        "size": self.settings.opensearch_top_k,
                        "query": {
                            "knn": {
                                "embedding": {
                                    "vector": embedding,
                                    "k": self.settings.opensearch_top_k,
                                    **(
                                        {"filter": {"bool": {"filter": access_filter}}}
                                        if access_filter
                                        else {}
                                    ),
                                }
                            }
                        },
                    },
                )
            )
        rrf_scores: dict[str, float] = {}
        hits_by_id: dict[str, dict[str, Any]] = {}
        for response in hit_sets:
            hits = response.get("hits", {}).get("hits", [])
            for rank, hit in enumerate(hits, start=1):
                source = hit.get("_source", {})
                hit_id = str(source.get("citation_id") or hit.get("_id") or "").strip()
                if not hit_id:
                    continue
                hits_by_id.setdefault(hit_id, hit)
                rrf_scores[hit_id] = rrf_scores.get(hit_id, 0.0) + 1 / (60 + rank)
        ranked_ids = sorted(
            hits_by_id,
            key=lambda hit_id: -rrf_scores[hit_id],
        )
        return [
            self._map_hit(hits_by_id[hit_id])
            for hit_id in ranked_ids[: self.settings.opensearch_top_k]
        ]

    @staticmethod
    def _access_filter(
        principal: PrincipalContext | None,
        filters: CorpusFilters | None,
    ) -> list[dict[str, Any]]:
        clauses: list[dict[str, Any]] = []
        if principal is not None:
            clauses.extend(
                [
                    {"terms": {"acl_principals": sorted(principal.acl_principals)}},
                    {
                        "terms": {
                            "security_label": [
                                "public",
                                principal.security_domain,
                            ]
                        }
                    },
                    {"term": {"deleted": False}},
                ]
            )
        if filters:
            if filters.source_types:
                clauses.append({"terms": {"source_type": filters.source_types}})
            if filters.document_ids:
                clauses.append({"terms": {"document_id": filters.document_ids}})
            if filters.entity_ids:
                clauses.append({"terms": {"entity_ids": filters.entity_ids}})
            if filters.effective_after:
                clauses.append({"range": {"effective_date": {"gte": filters.effective_after}}})
        return clauses

    def _create_client(self) -> Any:
        import boto3
        from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

        endpoint = urlparse(self.settings.opensearch_endpoint or "")
        if not endpoint.hostname:
            raise ValueError("A valid OpenSearch endpoint is required")
        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials are unavailable")
        auth = AWSV4SignerAuth(credentials, self.settings.aws_region, "es")
        self._client = OpenSearch(
            hosts=[
                {
                    "host": endpoint.hostname,
                    "port": endpoint.port or (443 if endpoint.scheme == "https" else 80),
                }
            ],
            http_auth=auth,
            use_ssl=endpoint.scheme == "https",
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=self.settings.bedrock_timeout_seconds,
            max_retries=1,
            retry_on_timeout=False,
        )
        return self._client

    def _embed(self, text: str) -> list[float]:
        if self._bedrock_client is None:
            import boto3

            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=self.settings.aws_region,
            )
        response = self._bedrock_client.invoke_model(
            modelId=self.settings.titan_embedding_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": self.EMBEDDING_DIMENSIONS,
                    "normalize": True,
                }
            ),
        )
        payload = json.loads(response["body"].read())
        embedding = payload.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != self.EMBEDDING_DIMENSIONS:
            raise ValueError("Titan returned an invalid embedding")
        return [float(value) for value in embedding]

    @staticmethod
    def _map_hit(hit: dict[str, Any]) -> Evidence:
        source = hit.get("_source", {})
        evidence_id = str(source.get("citation_id") or hit.get("_id") or "").strip()
        text = str(source.get("text") or "").strip()
        if not evidence_id or not text:
            raise ValueError("OpenSearch hit is not a normalized knowledge record")
        title = str(
            source.get("heading")
            or source.get("document_title")
            or source.get("document_id")
            or "Acquisition guidance"
        )
        return Evidence(
            evidence_id=evidence_id,
            source=str(source.get("authority") or source.get("source_type") or "Official source"),
            title=title,
            excerpt=text[:1000],
            url=str(source.get("url") or "") or None,
            document_id=str(source.get("document_id") or ""),
            version=str(source.get("version") or ""),
            security_label=str(source.get("security_label") or "public"),
            acl_principals=[str(value) for value in source.get("acl_principals", ["public"])],
            entity_ids=[str(value) for value in source.get("entity_ids", [])],
        )


class FallbackRetrievalAdapter:
    """Use a remote adapter when available, otherwise return local evidence."""

    def __init__(
        self,
        primary: RetrievalAdapter,
        fallback: RetrievalAdapter | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback or LocalRetrievalAdapter()

    def retrieve(
        self,
        queries: Sequence[str],
        principal: PrincipalContext | None = None,
        filters: CorpusFilters | None = None,
    ) -> list[Evidence]:
        try:
            if principal is None and filters is None:
                evidence = self.primary.retrieve(queries)
                return evidence or self.fallback.retrieve(queries)
            evidence = self.primary.retrieve(
                queries,
                principal=principal,
                filters=filters,
            )
            return evidence or self.fallback.retrieve(
                queries,
                principal=principal,
                filters=filters,
            )
        except Exception:
            # Never log queries: they can be derived from sensitive contract content.
            if principal is None and filters is None:
                return self.fallback.retrieve(queries)
            return self.fallback.retrieve(
                queries,
                principal=principal,
                filters=filters,
            )


class BedrockGPT56TerraSynthesisAdapter:
    """SigV4 Bedrock Mantle Responses API with deterministic fallback."""

    def __init__(
        self,
        settings: Settings,
        transport: Any | None = None,
        credentials: Any | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self._credentials = credentials

    def synthesize(
        self,
        rule_findings: Sequence[RuleFinding],
        evidence: Sequence[Evidence],
    ) -> tuple[str, list[Finding], str]:
        if self.settings.bedrock_enabled:
            try:
                return self._remote_synthesis(rule_findings, evidence)
            except Exception:
                # Deliberately avoid logging prompts, source text, or model output.
                pass
        summary, findings = self._offline_synthesis(rule_findings, evidence)
        return summary, findings, "offline-deterministic"

    def _remote_synthesis(
        self,
        rule_findings: Sequence[RuleFinding],
        evidence: Sequence[Evidence],
    ) -> tuple[str, list[Finding], str]:
        prompt = {
            "instruction": (
                "Return JSON with executive_summary and findings. Preserve rule_id, "
                "title, severity, description, recommendation, segment_ids. citation_ids "
                "may contain only supplied evidence_id values."
            ),
            "findings": [finding.model_dump(mode="json") for finding in rule_findings],
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        payload = self.request_json(prompt)
        source_map = {finding.rule_id: finding for finding in rule_findings}
        findings: list[Finding] = []
        raw_findings = payload.get("findings")
        if not isinstance(raw_findings, list):
            raise ValueError("Synthesis response is missing findings")
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            source = source_map.get(str(item.get("rule_id", "")))
            if source is None:
                continue
            raw_citations = item.get("citation_ids", [])
            citations = (
                [str(value) for value in raw_citations if isinstance(value, str)]
                if isinstance(raw_citations, list)
                else []
            )
            findings.append(
                Finding(
                    finding_id=f"F-{len(findings) + 1:03d}",
                    rule_id=source.rule_id,
                    title=str(item.get("title") or source.title),
                    # Severity remains owned by the deterministic rule engine.
                    severity=source.severity,
                    description=str(item.get("description") or source.description),
                    recommendation=str(item.get("recommendation") or source.recommendation),
                    citation_ids=citations,
                )
            )
        if rule_findings and not findings:
            raise ValueError("Synthesis response contains no recognized findings")
        summary = payload.get("executive_summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = self._summary(findings)
        return summary.strip(), findings, "bedrock-gpt-5.6-terra"

    def request_json(self, prompt: dict, max_output_tokens: int = 3000) -> dict:
        """Run Terra with an explicit structured prompt; errors propagate to caller."""
        import boto3
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        url = f"https://bedrock-mantle.{self.settings.aws_region}.api.aws/openai/v1/responses"
        body = json.dumps(
            {
                "model": self.settings.bedrock_model_id,
                "input": json.dumps(prompt, separators=(",", ":")),
                "max_output_tokens": max_output_tokens,
                "store": False,
            },
            separators=(",", ":"),
        )
        credentials = self._credentials or boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("AWS credentials are unavailable")
        request = AWSRequest(
            method="POST",
            url=url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        get_frozen_credentials = getattr(credentials, "get_frozen_credentials", None)
        if callable(get_frozen_credentials):
            credentials = get_frozen_credentials()
        SigV4Auth(
            credentials,
            "bedrock-mantle",
            self.settings.aws_region,
        ).add_auth(request)
        transport = self._transport or requests.Session()
        response = transport.post(
            url,
            data=body,
            headers=dict(request.headers),
            timeout=self.settings.bedrock_timeout_seconds,
        )
        response.raise_for_status()
        payload_response = response.json()
        self.last_usage = payload_response.get("usage", {})
        text = self._extract_response_text(payload_response)
        return self._parse_response(text)

    @staticmethod
    def _extract_response_text(payload: Any) -> str:
        if not isinstance(payload, dict):
            raise ValueError("Responses API result must be an object")
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text
        texts: list[str] = []
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text)
        if not texts:
            raise ValueError("Responses API result has no output text")
        return "\n".join(texts)

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
            candidate = re.sub(r"\s*```$", "", candidate)
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Synthesis response is not JSON")
        payload = json.loads(candidate[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Synthesis response must be an object")
        return payload

    @staticmethod
    def _offline_synthesis(
        rule_findings: Sequence[RuleFinding],
        evidence: Sequence[Evidence],
    ) -> tuple[str, list[Finding]]:
        findings = [
            Finding(
                finding_id=f"F-{index:03d}",
                rule_id=item.rule_id,
                title=item.title,
                severity=item.severity,
                description=item.description,
                recommendation=item.recommendation,
                citation_ids=BedrockGPT56TerraSynthesisAdapter._relevant_citations(
                    item,
                    evidence,
                ),
            )
            for index, item in enumerate(rule_findings, start=1)
        ]
        return BedrockGPT56TerraSynthesisAdapter._summary(findings), findings

    @staticmethod
    def _relevant_citations(
        finding: RuleFinding,
        evidence: Sequence[Evidence],
    ) -> list[str]:
        stopwords = {
            "acquisition",
            "clause",
            "contract",
            "federal",
            "regulation",
            "required",
            "the",
            "and",
            "for",
        }

        def tokens(value: str) -> set[str]:
            return {
                token
                for token in re.findall(r"[a-z0-9.:-]+", value.casefold())
                if len(token) >= 3 and token not in stopwords
            }

        query_tokens = tokens(
            " ".join([finding.title, finding.description, *finding.evidence_queries])
        )
        ranked: list[tuple[int, str]] = []
        for item in evidence:
            if item.evidence_id in finding.evidence_queries:
                score = 100
            else:
                score = len(query_tokens & tokens(f"{item.source} {item.title} {item.excerpt}"))
            if score > 0:
                ranked.append((score, item.evidence_id))
        ranked.sort(key=lambda value: (-value[0], value[1]))
        return [evidence_id for _, evidence_id in ranked[:2]]

    @staticmethod
    def _summary(findings: Sequence[Finding]) -> str:
        high_count = sum(
            finding.severity in {Severity.CRITICAL, Severity.HIGH} for finding in findings
        )
        summary = (
            f"Review identified {len(findings)} potential issue(s), including "
            f"{high_count} high or critical item(s). Human acquisition and legal "
            "review is required before action."
        )
        return summary
