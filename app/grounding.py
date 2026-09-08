"""Authorization and citation checks shared by every intelligence workflow."""

from __future__ import annotations

from collections.abc import Iterable

from app.models import CitedStatement, Evidence, GroundingStatus, PrincipalContext


def evidence_is_authorized(
    evidence: Evidence,
    principal: PrincipalContext,
) -> bool:
    """Deny access unless both the domain and an ACL principal are authorized."""

    if evidence.security_label not in {"public", principal.security_domain}:
        return False
    return bool(set(evidence.acl_principals) & principal.acl_principals)


def authorized_evidence(
    evidence: Iterable[Evidence],
    principal: PrincipalContext,
) -> list[Evidence]:
    return [
        item.model_copy(deep=True) for item in evidence if evidence_is_authorized(item, principal)
    ]


def authorized_citation_ids(
    evidence: Iterable[Evidence],
    principal: PrincipalContext,
) -> set[str]:
    return {item.evidence_id for item in evidence if evidence_is_authorized(item, principal)}


def enforce_statement_grounding(
    statements: Iterable[CitedStatement],
    evidence: Iterable[Evidence],
    principal: PrincipalContext,
) -> list[CitedStatement]:
    """Strip unknown or unauthorized citations and mark unsupported assertions."""

    allowed = authorized_citation_ids(evidence, principal)
    grounded: list[CitedStatement] = []
    for statement in statements:
        citation_ids = list(
            dict.fromkeys(
                citation_id for citation_id in statement.citation_ids if citation_id in allowed
            )
        )
        grounded.append(
            statement.model_copy(
                update={
                    "citation_ids": citation_ids,
                    "grounding_status": (
                        GroundingStatus.VERIFIED if citation_ids else GroundingStatus.UNVERIFIED
                    ),
                }
            )
        )
    return grounded
