import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_classifier_pull_policy_matches_pinned_upstream_repository() -> None:
    policy_path = "scripts/iam/classifier-base-image-pull.json"
    policy = json.loads((ROOT / policy_path).read_text())
    image = (ROOT / "Dockerfile.sagemaker").read_text().splitlines()[0]
    match = re.fullmatch(
        r"FROM (\d{12})\.dkr\.ecr\.(us-gov-[\w-]+)\.amazonaws\.com/([^@]+)@sha256:[0-9a-f]{64}",
        image,
    )
    assert match is not None
    account, region, repository = match.groups()
    assert policy["Statement"] == [
        {
            "Sid": "ClassifierBaseImagePull",
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:BatchGetImage",
                "ecr:GetDownloadUrlForLayer",
            ],
            "Resource": f"arn:aws-us-gov:ecr:{region}:{account}:repository/{repository}",
        }
    ]
    bootstrap = (ROOT / "scripts/bootstrap-govcloud.ps1").read_text()
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()
    assert "iam/classifier-base-image-pull.json" in bootstrap
    assert f"file://{policy_path}" in workflow
    assert workflow.index("Ensure classifier base image pull access") < workflow.index(
        "Build private classifier inference image"
    )


def test_production_image_packages_verified_model_and_corpus() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert "artifacts/models/legal-bert-cuad" in dockerfile
    assert "artifacts/knowledge/federal-v2.sqlite" in dockerfile
    assert "python scripts/verify-shared-artifacts.py" in dockerfile


def test_classifier_reuse_tracks_all_build_inputs_and_passrole_is_scoped() -> None:
    image = (ROOT / "Dockerfile.sagemaker").read_text()
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()
    terraform = (ROOT / "infra/terraform/sagemaker-inference.tf").read_text()
    # A COPY/ADD/ARG or another stage requires expanding the build-cache key.
    assert set(re.findall(r"^([A-Z]+) ", image, re.MULTILINE)) <= {"FROM", "RUN"}
    assert image.count("FROM ") == 1
    assert "sha256sum Dockerfile.sagemaker" in workflow
    assert "ImageNotFoundException" in workflow
    assert "-target=aws_iam_role_policy.classifier_deploy" in workflow
    grant = terraform.split('resource "aws_iam_role_policy" "classifier_deploy"')[1].split(
        'resource "aws_sagemaker_model"'
    )[0]
    assert 'Action   = "iam:PassRole"' in grant
    assert "Resource = aws_iam_role.sagemaker_training.arn" in grant
    assert '"iam:PassedToService" = "sagemaker.amazonaws.com"' in grant
    for action in (
        "CreateModel",
        "DescribeModel",
        "DeleteModel",
        "CreateEndpointConfig",
        "DescribeEndpointConfig",
        "DeleteEndpointConfig",
        "CreateEndpoint",
        "DescribeEndpoint",
        "UpdateEndpoint",
        "DeleteEndpoint",
        "AddTags",
        "ListTags",
        "DeleteTags",
        "InvokeEndpoint",
    ):
        assert f'"sagemaker:{action}"' in grant
    assert '"sagemaker:*"' not in grant
    for resource in ("model", "endpoint-config", "endpoint"):
        assert f":{resource}/${{local.managed_classifier_endpoint_name}}" in grant


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
    artifact = json.loads(
        (ROOT / "ml/deployment/llama_r128_ensemble_artifact.json").read_text(encoding="utf-8")
    )

    assert '"sagemaker:InvokeEndpoint"' in terraform
    assert '{ name = "CLASSIFIER_ENDPOINT_NAME"' in terraform
    assert "/srv/app/artifacts/models/legal-bert-cuad" in terraform
    assert "/srv/app/ml/deployment/llama_r128_ensemble_thresholds.json" in terraform
    assert "TF_VAR_classifier_endpoint_name" in workflow
    assert "TF_VAR_classifier_model_data_url" in workflow
    assert "Verify pinned classifier artifact" in workflow
    assert "aws s3api head-object" in workflow
    assert "Reclaim disk for classifier image" in workflow
    assert "available_kib >= 28 * 1024 * 1024" in workflow
    assert "Dockerfile.sagemaker" in workflow
    assert "sagemaker-runtime invoke-endpoint" in workflow
    assert "enable_network_isolation = true" in endpoint
    assert "instance_type" in endpoint
    assert "var.classifier_endpoint_instance_type" in endpoint
    assert "volume_size_in_gb" not in endpoint
    assert "kms_key_arn" not in endpoint
    assert "NVMe device is encrypted in hardware" in endpoint
    assert 'MMS_DEFAULT_WORKERS_PER_MODEL  = "1"' in endpoint
    assert 'SAGEMAKER_MODEL_SERVER_WORKERS = "1"' in endpoint
    assert "transformers==4.57.6" in image
    assert "@sha256:" in image
    assert thresholds["model_id"] == "Llama-3.1-CUAD-r128-ensemble-3seed"
    assert thresholds["promotion_authorized"] is False
    assert len(thresholds["member_weights_sha256"]) == 3
    assert artifact["model_id"] == thresholds["model_id"]
    assert artifact["archive_bytes"] > 10_000_000_000
    assert artifact["archive_sha256"] in artifact["s3_uri"]
    assert artifact["s3_version_id"]
    assert artifact["server_side_encryption"] == "aws:kms"


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
