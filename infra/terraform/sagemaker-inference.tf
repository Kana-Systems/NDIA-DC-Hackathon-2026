locals {
  # The pre-G6 name belongs to an interrupted CreateEndpoint operation that AWS
  # still holds in Creating. Keep a stable, distinct name for the G6 deployment.
  managed_classifier_endpoint_name = "${local.name}-llama-cuad-g6"
  classifier_endpoint_enabled      = var.classifier_model_data_url != ""
  classifier_environment = {
    HF_HUB_OFFLINE = "1"
    # Leave GPU headroom for all three LoRA adapters on the 24 GB L4.
    MODEL_INFERENCE_BATCH_SIZE     = "1"
    MMS_DEFAULT_WORKERS_PER_MODEL  = "1"
    PYTORCH_CUDA_ALLOC_CONF        = "expandable_segments:True"
    SAGEMAKER_CONTAINER_LOG_LEVEL  = "20"
    SAGEMAKER_MODEL_SERVER_WORKERS = "1"
    SAGEMAKER_PROGRAM              = "inference.py"
    SAGEMAKER_SUBMIT_DIRECTORY     = "/opt/ml/model/code"
    TOKENIZERS_PARALLELISM         = "false"
    TRANSFORMERS_OFFLINE           = "1"
  }
  # SageMaker model/config names are immutable. Include runtime inputs so a
  # replacement can be created before the previous resource is destroyed.
  classifier_model_revision = substr(sha256(jsonencode({
    artifact    = var.classifier_model_data_url
    image       = var.classifier_inference_image_uri
    environment = local.classifier_environment
  })), 0, 12)
  classifier_config_revision = substr(sha256(jsonencode({
    model_revision = local.classifier_model_revision
    instance_type  = var.classifier_endpoint_instance_type
  })), 0, 12)
  effective_classifier_endpoint_name = (
    local.classifier_endpoint_enabled
    ? local.managed_classifier_endpoint_name
    : var.classifier_endpoint_name
  )
}

check "classifier_endpoint_source" {
  assert {
    condition = !(
      local.classifier_endpoint_enabled
      && var.classifier_endpoint_name != ""
    )
    error_message = "Set either classifier_model_data_url for a managed endpoint or classifier_endpoint_name for an external endpoint, not both."
  }
}

# Keep the optional classifier's complete deployment boundary together: passing
# its execution role, managing versioned runtime resources, tags, and smoke tests.
resource "aws_iam_role_policy" "classifier_deploy" {
  count = local.classifier_endpoint_enabled && var.github_deploy_role_name != "" ? 1 : 0

  name = "classifier-runtime-passrole"
  role = var.github_deploy_role_name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "PassClassifierExecutionRole"
      Effect   = "Allow"
      Action   = "iam:PassRole"
      Resource = aws_iam_role.sagemaker_training.arn
      Condition = {
        StringEquals = { "iam:PassedToService" = "sagemaker.amazonaws.com" }
      }
      }, {
      Sid    = "ManageClassifierRuntime"
      Effect = "Allow"
      Action = [
        "sagemaker:CreateModel", "sagemaker:DescribeModel", "sagemaker:DeleteModel",
        "sagemaker:CreateEndpointConfig", "sagemaker:DescribeEndpointConfig", "sagemaker:DeleteEndpointConfig",
        "sagemaker:CreateEndpoint", "sagemaker:DescribeEndpoint", "sagemaker:UpdateEndpoint", "sagemaker:DeleteEndpoint",
        "sagemaker:AddTags", "sagemaker:ListTags", "sagemaker:DeleteTags",
        "sagemaker:InvokeEndpoint",
      ]
      Resource = [
        "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:model/${local.managed_classifier_endpoint_name}-*",
        "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint-config/${local.managed_classifier_endpoint_name}-*",
        "arn:${data.aws_partition.current.partition}:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:endpoint/${local.managed_classifier_endpoint_name}",
      ]
    }]
  })
}

resource "aws_sagemaker_model" "classifier" {
  count = local.classifier_endpoint_enabled ? 1 : 0

  depends_on = [aws_iam_role_policy.classifier_deploy]

  name                     = "${local.managed_classifier_endpoint_name}-${local.classifier_model_revision}"
  execution_role_arn       = aws_iam_role.sagemaker_training.arn
  enable_network_isolation = true

  primary_container {
    image          = var.classifier_inference_image_uri
    model_data_url = var.classifier_model_data_url
    environment    = local.classifier_environment
  }

  lifecycle {
    create_before_destroy = true

    precondition {
      condition = startswith(
        var.classifier_inference_image_uri,
        "${aws_ecr_repository.app.repository_url}@sha256:"
      )
      error_message = "A managed classifier endpoint requires the reviewed private ECR image by immutable digest."
    }

    precondition {
      condition = startswith(
        var.classifier_model_data_url,
        "s3://${aws_s3_bucket.sagemaker_training.id}/inference/"
      )
      error_message = "The managed classifier artifact must be in this deployment's encrypted training bucket."
    }
  }

  tags = merge(local.common_tags, {
    Name           = "${local.managed_classifier_endpoint_name}-model"
    SecurityDomain = var.security_domain
    ModelId        = var.classifier_endpoint_model_id
  })
}

resource "aws_sagemaker_endpoint_configuration" "classifier" {
  count = local.classifier_endpoint_enabled ? 1 : 0

  # G6 uses fixed local NVMe instance storage. SageMaker does not accept
  # customer-managed EBS volume size or KMS settings for this instance family;
  # the NVMe device is encrypted in hardware with per-instance keys.
  name = "${local.managed_classifier_endpoint_name}-${local.classifier_config_revision}"

  production_variants {
    variant_name                                      = "AllTraffic"
    model_name                                        = aws_sagemaker_model.classifier[0].name
    initial_instance_count                            = 1
    instance_type                                     = var.classifier_endpoint_instance_type
    initial_variant_weight                            = 1
    container_startup_health_check_timeout_in_seconds = 1800
    model_data_download_timeout_in_seconds            = 1800
  }

  lifecycle {
    create_before_destroy = true
  }

  tags = merge(local.common_tags, {
    Name           = "${local.managed_classifier_endpoint_name}-config"
    SecurityDomain = var.security_domain
    ModelId        = var.classifier_endpoint_model_id
  })
}

resource "aws_sagemaker_endpoint" "classifier" {
  count = local.classifier_endpoint_enabled ? 1 : 0

  name                 = local.managed_classifier_endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.classifier[0].name

  deployment_config {
    blue_green_update_policy {
      traffic_routing_configuration {
        type                     = "ALL_AT_ONCE"
        wait_interval_in_seconds = 0
      }

      maximum_execution_timeout_in_seconds = 3600
      termination_wait_in_seconds          = 60
    }
  }

  tags = merge(local.common_tags, {
    Name           = local.managed_classifier_endpoint_name
    SecurityDomain = var.security_domain
    ModelId        = var.classifier_endpoint_model_id
  })
}
