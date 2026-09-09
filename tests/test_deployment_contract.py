import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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


def test_govcloud_alb_has_allowlist_waf_and_rate_limit_controls() -> None:
    policy_path = "scripts/iam/security-perimeter-deploy.json"
    policy = json.loads((ROOT / policy_path).read_text())
    statements = {statement["Sid"]: statement for statement in policy["Statement"]}
    firewall = statements["ManageProjectWebFirewall"]
    parameter = statements["ReadTrustedIngressParameter"]
    waf = (ROOT / "infra/terraform/waf.tf").read_text()
    main = (ROOT / "infra/terraform/main.tf").read_text()
    bootstrap = (ROOT / "scripts/bootstrap-govcloud.ps1").read_text()
    workflow = (ROOT / ".github/workflows/deploy.yml").read_text()

    assert parameter["Action"] == "ssm:GetParameter"
    assert parameter["Resource"].endswith(":parameter/contract-review/*")
    assert {
        "wafv2:CreateWebACL",
        "wafv2:AssociateWebACL",
        "wafv2:PutLoggingConfiguration",
        "wafv2:CreateIPSet",
    }.issubset(firewall["Action"])
    assert all(
        resource == "*" or "${aws:PrincipalAccount}" in resource
        for statement in policy["Statement"]
        for resource in (
            statement["Resource"]
            if isinstance(statement["Resource"], list)
            else [statement["Resource"]]
        )
    )
    assert '"AWSManagedRulesCommonRuleSet"' in waf
    assert '"AWSManagedRulesKnownBadInputsRuleSet"' in waf
    assert '"AWSManagedRulesAmazonIpReputationList"' in waf
    assert 'name     = "TrustedIngressOnly"' in waf
    assert 'name     = "TrustedSourceRateLimit"' in waf
    assert 'name = "authorization"' in waf
    assert 'name = "cookie"' in waf
    assert "aws_wafv2_web_acl_association" in waf
    assert "local.effective_allowed_ingress_cidrs" in main
    assert "iam/security-perimeter-deploy.json" in bootstrap
    assert policy_path in workflow
    assert 'split("${aws:PrincipalAccount}") | join($account)' in workflow
    assert '"file://${rendered_policy}"' in workflow
    assert ".Replace(" in bootstrap
    assert "'${aws:PrincipalAccount}'" in bootstrap
    assert workflow.index("Ensure security perimeter deployment access") < workflow.index(
        "Terraform plan"
    )


@pytest.mark.parametrize("account", ["123456789012", "210987654321", "invalid-account"])
def test_security_perimeter_step_uploads_rendered_account_policy(
    tmp_path: Path, account: str
) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/deploy.yml").read_text())
    step = next(
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if step.get("name") == "Ensure security perimeter deployment access"
    )
    runner_temp = tmp_path / "runner temp"
    runner_temp.mkdir()
    uploaded = tmp_path / "uploaded.json"
    # Exercise the real workflow shell and jq transformation without AWS access.
    aws = tmp_path / "aws"
    aws.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args == ['sts', 'get-caller-identity', '--query', 'Account', '--output', 'text']:\n"
        "    print(os.environ['TEST_ACCOUNT'])\n"
        "else:\n"
        "    assert args[:2] == ['iam', 'put-role-policy'], args\n"
        "    assert args[2:6] == ['--role-name', 'contract-review-github',\n"
        "                          '--policy-name', 'security-perimeter-deploy'], args\n"
        "    assert args[6] == '--policy-document' and args[7].startswith('file://'), args\n"
        "    source = pathlib.Path(args[7][7:])\n"
        "    assert source.parent == pathlib.Path(os.environ['RUNNER_TEMP']), source\n"
        "    pathlib.Path(os.environ['TEST_UPLOADED']).write_bytes(source.read_bytes())\n"
    )
    aws.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_TEMP": str(runner_temp),
            "AWS_GOV_ROLE_ARN": "arn:aws-us-gov:iam::123456789012:role/contract-review-github",
            "TEST_ACCOUNT": account,
            "TEST_UPLOADED": str(uploaded),
        },
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if account == "invalid-account":
        assert result.returncode != 0
        assert not uploaded.exists()
    else:
        assert result.returncode == 0, result.stderr
        template = (ROOT / "scripts/iam/security-perimeter-deploy.json").read_text()
        expected = json.loads(template.replace("${aws:PrincipalAccount}", account))
        assert json.loads(uploaded.read_text()) == expected
        assert "${aws:PrincipalAccount}" not in uploaded.read_text()


def test_production_image_packages_verified_model_and_corpus() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "--index-url https://download.pytorch.org/whl/cpu" in dockerfile
    assert "artifacts/models/legal-bert-cuad" in dockerfile
    assert "artifacts/knowledge/federal-v2.sqlite" in dockerfile
    assert "python scripts/verify-shared-artifacts.py" in dockerfile


def test_classifier_reuse_tracks_build_inputs_and_deploy_policy_is_scoped() -> None:
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
    variables = (ROOT / "infra/terraform/variables.tf").read_text(encoding="utf-8")
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
    assert 'default     = "ml.g6.2xlarge"' in variables
    assert "ml.g6e." not in variables
    assert 'var.classifier_endpoint_instance_type == "ml.g6.2xlarge"' in variables
    assert 'MODEL_INFERENCE_BATCH_SIZE     = "1"' in endpoint
    # Replacing a runtime must not collide with its still-existing AWS name.
    assert "artifact    = var.classifier_model_data_url" in endpoint
    assert "image       = var.classifier_inference_image_uri" in endpoint
    assert "environment = local.classifier_environment" in endpoint
    assert "model_revision = local.classifier_model_revision" in endpoint
    assert "instance_type  = var.classifier_endpoint_instance_type" in endpoint
    config = endpoint.split('resource "aws_sagemaker_endpoint_configuration"')[1].split(
        'resource "aws_sagemaker_endpoint"'
    )[0]
    assert "${local.classifier_config_revision}" in config
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
