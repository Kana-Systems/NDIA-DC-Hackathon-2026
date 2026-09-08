from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_image_packages_verified_model_and_corpus() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert "artifacts/models/legal-bert-cuad" in dockerfile
    assert "artifacts/knowledge/federal-v2.sqlite" in dockerfile
    assert "python scripts/verify-shared-artifacts.py" in dockerfile


def test_deployment_fetches_lfs_and_enables_model_review() -> None:
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    terraform = (ROOT / "infra/terraform/main.tf").read_text(encoding="utf-8")

    assert "lfs: true" in workflow
    assert "python scripts/verify-shared-artifacts.py" in workflow
    assert '{ name = "MODEL_REVIEW_ENABLED", value = "true" }' in terraform
    assert '{ name = "LOCAL_CORPUS_PATH"' in terraform


def test_target_objects_and_retired_ui_are_not_deployed() -> None:
    terraform = (ROOT / "infra/terraform/main.tf").read_text(encoding="utf-8")
    api = (ROOT / "app/api.py").read_text(encoding="utf-8")

    assert '"/ui"' not in terraform
    assert "target-objects" not in api
