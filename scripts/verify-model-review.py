"""Exercise actual trained weights, ingested federal text, and live Terra together."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.model_review import ModelReviewService  # noqa: E402
from app.parsers import DocumentParser  # noqa: E402
from app.sample import sample_contract_bytes, sample_metadata  # noqa: E402


def main():
    settings = Settings(
        workspace_password="local-smoke-password",
        demo_jwt_secret="local-smoke-signing-secret-32-characters",
        bedrock_enabled=True,
        model_review_enabled=True,
        bedrock_timeout_seconds=120,
        classifier_model_dir=os.getenv("CLASSIFIER_MODEL_DIR", "artifacts/models/legal-bert-cuad"),
        local_corpus_path=os.getenv("LOCAL_CORPUS_PATH", "artifacts/knowledge/federal-v2.sqlite"),
    )
    document = DocumentParser(settings).parse("sample.docx", sample_contract_bytes())
    report = ModelReviewService(settings).review(document, sample_metadata())
    output = Path("artifacts/model-review-smoke.json")
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mode": report.synthesis_mode,
                "classifiers": report.classifier_model_ids,
                "findings": len(report.findings),
                "evidence": len(report.evidence),
                "corpus": report.corpus_manifest,
                "report": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
