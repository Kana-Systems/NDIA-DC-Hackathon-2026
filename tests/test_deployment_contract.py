import json
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


def test_llama_classifier_canary_keeps_packaged_rollback() -> None:
    terraform = (ROOT / "infra/terraform/main.tf").read_text(encoding="utf-8")
    endpoint = (ROOT / "infra/terraform/sagemaker-inference.tf").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    image = (ROOT / "Dockerfile.sagemaker").read_text(encoding="utf-8")
    thresholds = json.loads(
        (ROOT / "ml/deployment/llama_r128_ensemble_thresholds.json").read_text(encoding="utf-8")
    )

    assert '"sagemaker:InvokeEndpoint"' in terraform
    assert '{ name = "CLASSIFIER_ENDPOINT_NAME"' in terraform
    assert "/srv/app/artifacts/models/legal-bert-cuad" in terraform
    assert "/srv/app/ml/deployment/llama_r128_ensemble_thresholds.json" in terraform
    assert "TF_VAR_classifier_endpoint_name" in workflow
    assert "TF_VAR_classifier_model_data_url" in workflow
    assert "Dockerfile.sagemaker" in workflow
    assert 'enable_network_isolation = true' in endpoint
    assert "instance_type" in endpoint
    assert "var.classifier_endpoint_instance_type" in endpoint
    assert "transformers==4.57.6" in image
    assert "@sha256:" in image
    assert thresholds["model_id"] == "Llama-3.1-CUAD-r128-ensemble-3seed"
    assert thresholds["promotion_authorized"] is False
    assert len(thresholds["member_weights_sha256"]) == 3


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
