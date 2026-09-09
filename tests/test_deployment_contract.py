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


def test_workspace_aws_storage_is_durable_and_seed_steps_disabled() -> None:
    table = (ROOT / "infra/terraform/workspace.tf").read_text()
    task = (ROOT / "infra/terraform/main.tf").read_text()
    role = (ROOT / "infra/terraform/j2.tf").read_text()
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()
    assert 'hash_key     = "scope"' in table
    assert 'range_key    = "id"' in table
    assert "point_in_time_recovery" in table
    assert "deletion_protection_enabled = true" in table
    assert "aws_kms_key.j2.arn" in table
    assert '{ name = "WORKSPACE_TABLE", value = aws_dynamodb_table.lens_workspace.name }' in task
    assert "aws_dynamodb_table.lens_workspace.arn" in role
    assert "if: ${{ false }} # Synthetic fixtures" in workflow
