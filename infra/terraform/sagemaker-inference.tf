locals {
  managed_classifier_endpoint_name = "${local.name}-llama-cuad"
  classifier_endpoint_enabled      = var.classifier_model_data_url != ""
  classifier_model_revision        = substr(sha256(var.classifier_model_data_url), 0, 12)
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

resource "aws_sagemaker_model" "classifier" {
  count = local.classifier_endpoint_enabled ? 1 : 0

  name                     = "${local.managed_classifier_endpoint_name}-${local.classifier_model_revision}"
  execution_role_arn       = aws_iam_role.sagemaker_training.arn
  enable_network_isolation = true

  primary_container {
    image          = var.classifier_inference_image_uri
    model_data_url = var.classifier_model_data_url
    environment = {
      HF_HUB_OFFLINE                = "1"
      MODEL_INFERENCE_BATCH_SIZE    = "4"
      PYTORCH_CUDA_ALLOC_CONF       = "expandable_segments:True"
      SAGEMAKER_CONTAINER_LOG_LEVEL = "20"
      SAGEMAKER_PROGRAM             = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY    = "/opt/ml/model/code"
      TOKENIZERS_PARALLELISM        = "false"
      TRANSFORMERS_OFFLINE          = "1"
    }
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

  name        = "${local.managed_classifier_endpoint_name}-${local.classifier_model_revision}"
  kms_key_arn = aws_kms_key.j2.arn

  production_variants {
    variant_name                                      = "AllTraffic"
    model_name                                        = aws_sagemaker_model.classifier[0].name
    initial_instance_count                            = 1
    instance_type                                     = var.classifier_endpoint_instance_type
    initial_variant_weight                            = 1
    container_startup_health_check_timeout_in_seconds = 1800
    model_data_download_timeout_in_seconds            = 1800
    volume_size_in_gb                                 = 80
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
