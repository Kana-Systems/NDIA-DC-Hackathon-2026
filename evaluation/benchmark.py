"""Run the deterministic, network-free contract review benchmark."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ml.heuristic import HeuristicClassifier
from ml.metrics import multilabel_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = Path(__file__).with_name("expected_outputs.json")
VALID_CLAUSE = re.compile(r"\b(?:52\.\d{3}-\d+|252\.\d{3}-\d{4})\b", re.I)
CLAUSE_LIKE = re.compile(r"\b(?:FAR|DFARS)\s+(\d{2,3}[.-]\d{3}[.-]\d{1,4})\b", re.I)


def _dfars_204_7304_c_applies(metadata: dict[str, Any]) -> bool:
    department = str(metadata.get("agency_department", "")).strip().casefold()
    instrument = str(metadata.get("instrument_type", "")).strip().casefold()
    is_dod = department in {"dod", "department of defense", "u.s. department of defense"}
    return (
        is_dod
        and instrument in {"solicitation", "contract"}
        and metadata.get("acquisition_solely_cots") is not True
    )


def analyze(text: str, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    valid = {match.group(0).casefold() for match in VALID_CLAUSE.finditer(text)}
    findings: list[dict[str, Any]] = []
    if _dfars_204_7304_c_applies(metadata) and "252.204-7012" not in valid:
        findings.append(
            {
                "id": "missing-required-clause:252.204-7012",
                "type": "missing_required_clause",
                "rule_id": "cybersecurity-dfars-252-204-7012",
                "severity": "high",
                "required_clause": "252.204-7012",
                "citations": [
                    "dfars:dfars:204-7304-c:5343579076a3",
                    "dfars:dfars:252-204-7012:67cc8cf25a2b",
                ],
            }
        )
    sam_trigger = re.compile(r"\b(?:solicitation|offeror)\b", re.I)
    if sam_trigger.search(text) and "52.204-7" not in valid:
        findings.append(
            {
                "id": "missing-required-clause:52.204-7",
                "type": "missing_required_clause",
                "rule_id": "sam-registration-far-52-204-7",
                "severity": "medium",
                "required_clause": "52.204-7",
                "citations": ["far:far:52-204-7:8e198e7e9db4"],
            }
        )
    malformed = sorted(
        {
            match.group(0)
            for match in CLAUSE_LIKE.finditer(text)
            if not VALID_CLAUSE.search(match.group(1))
        }
    )
    for evidence in malformed:
        findings.append(
            {
                "id": f"malformed-clause-reference:{evidence}",
                "type": "malformed_clause_reference",
                "rule_id": "malformed-clause-reference",
                "severity": "medium",
                "evidence": evidence,
                "citations": [],
            }
        )
    return sorted(findings, key=lambda item: item["id"])


def run(expected_path: Path = DEFAULT_EXPECTED) -> dict[str, Any]:
    specification = json.loads(expected_path.read_text(encoding="utf-8"))
    case_results = []
    predicted_ids, expected_ids = [], []
    classifier = HeuristicClassifier()
    for case in specification["cases"]:
        contract_path = ROOT / case["fixture"]
        text = contract_path.read_text(encoding="utf-8")
        findings = analyze(text, case.get("metadata", {}))
        predicted = [item["id"] for item in findings]
        expected = case["expected_finding_ids"]
        predicted_ids.append(predicted)
        expected_ids.append(expected)
        case_results.append(
            {
                "id": case["id"],
                "passed": predicted == expected,
                "findings": findings,
                "classifier": classifier.predict(text),
            }
        )
    return {
        "schema_version": "1.0",
        "metrics": multilabel_metrics(predicted_ids, expected_ids),
        "passed": all(item["passed"] for item in case_results),
        "cases": case_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.expected)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
