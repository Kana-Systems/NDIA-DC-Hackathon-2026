locals {
  j2_name             = substr("${local.name}-${var.security_domain}-j2", 0, 40)
  j2_enterprise_index = "${var.security_domain}-j2-enterprise-intelligence-v1"

  j2_table_arns = [
    aws_dynamodb_table.j2_documents.arn,
    aws_dynamodb_table.j2_entities.arn,
    aws_dynamodb_table.j2_changes.arn,
    aws_dynamodb_table.j2_workflows.arn,
  ]

  j2_environment = [
    { name = "AWS_REGION", value = var.aws_region },
    { name = "SECURITY_DOMAIN", value = var.security_domain },
    { name = "DOCUMENT_REGISTRY_TABLE", value = aws_dynamodb_table.j2_documents.name },
    { name = "ENTITY_REGISTRY_TABLE", value = aws_dynamodb_table.j2_entities.name },
    { name = "CHANGE_EVENT_TABLE", value = aws_dynamodb_table.j2_changes.name },
    { name = "WORKFLOW_TABLE", value = aws_dynamodb_table.j2_workflows.name },
    { name = "J2_INGESTION_QUEUE_URL", value = aws_sqs_queue.j2_ingestion.url },
    { name = "J2_KMS_KEY_ARN", value = aws_kms_key.j2.arn },
    { name = "SOURCE_BUCKET", value = aws_s3_bucket.j2_sources.id },
    { name = "OPENSEARCH_ENDPOINT", value = "https://${aws_opensearch_domain.main.endpoint}" },
    { name = "ENTERPRISE_INDEX", value = local.j2_enterprise_index },
    { name = "DURABLE_INGESTION_ENABLED", value = "true" },
    { name = "GRAPH_CONNECTOR_SECRET_ARN", value = try(aws_secretsmanager_secret.graph_connector[0].arn, "") },
  ]
}

resource "aws_kms_key" "j2" {
  description             = "Customer-managed key for ${local.j2_name} data"
  enable_key_rotation     = true
  deletion_window_in_days = 30

  tags = merge(local.common_tags, {
    Name           = local.j2_name
    SecurityDomain = var.security_domain
  })
}

resource "aws_kms_alias" "j2" {
  name          = "alias/${local.j2_name}"
  target_key_id = aws_kms_key.j2.key_id
}

resource "aws_kms_alias" "j2_data" {
  name          = "alias/${local.j2_name}-data"
  target_key_id = aws_kms_key.j2.key_id
}

resource "aws_s3_bucket" "j2_sources" {
  bucket        = "${local.j2_name}-sources-${data.aws_caller_identity.current.account_id}"
  force_destroy = false

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_s3_bucket_versioning" "j2_sources" {
  bucket = aws_s3_bucket.j2_sources.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "j2_sources" {
  bucket = aws_s3_bucket.j2_sources.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.j2.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "j2_sources" {
  bucket                  = aws_s3_bucket.j2_sources.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "j2_sources" {
  bucket = aws_s3_bucket.j2_sources.id
  rule {
    id     = "expire-noncurrent-source-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

data "aws_iam_policy_document" "j2_sources" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    resources = [
      aws_s3_bucket.j2_sources.arn,
      "${aws_s3_bucket.j2_sources.arn}/*",
    ]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "j2_sources" {
  bucket = aws_s3_bucket.j2_sources.id
  policy = data.aws_iam_policy_document.j2_sources.json
}

resource "aws_sqs_queue" "j2_ingestion_dlq" {
  name                              = "${local.j2_name}-ingestion-dlq"
  message_retention_seconds         = 1209600
  kms_master_key_id                 = aws_kms_key.j2.arn
  kms_data_key_reuse_period_seconds = 300

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_sqs_queue" "j2_ingestion" {
  name                              = "${local.j2_name}-ingestion"
  visibility_timeout_seconds        = 900
  message_retention_seconds         = 345600
  receive_wait_time_seconds         = 20
  kms_master_key_id                 = aws_kms_key.j2.arn
  kms_data_key_reuse_period_seconds = 300

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.j2_ingestion_dlq.arn
    maxReceiveCount     = 5
  })

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_sqs_queue_redrive_allow_policy" "j2_ingestion_dlq" {
  queue_url = aws_sqs_queue.j2_ingestion_dlq.id

  redrive_allow_policy = jsonencode({
    redrivePermission = "byQueue"
    sourceQueueArns   = [aws_sqs_queue.j2_ingestion.arn]
  })
}

resource "aws_dynamodb_table" "j2_documents" {
  name         = "${local.j2_name}-documents"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "document_id"

  attribute {
    name = "document_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.j2.arn
  }

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_dynamodb_table" "j2_entities" {
  name         = "${local.j2_name}-entities"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "entity_id"

  attribute {
    name = "entity_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.j2.arn
  }

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_dynamodb_table" "j2_changes" {
  name         = "${local.j2_name}-changes"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "change_id"

  attribute {
    name = "change_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.j2.arn
  }

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_dynamodb_table" "j2_workflows" {
  name         = "${local.j2_name}-workflows"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "object_id"

  attribute {
    name = "object_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.j2.arn
  }

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_secretsmanager_secret" "graph_connector" {
  count = var.create_graph_connector_secret ? 1 : 0

  name                    = "${local.j2_name}/microsoft-graph"
  description             = "Optional Microsoft Graph connector credentials populated outside Terraform"
  kms_key_id              = aws_kms_key.j2.arn
  recovery_window_in_days = 30

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

data "aws_iam_policy_document" "j2_data_access" {
  statement {
    sid = "ReadWriteJ2Tables"
    actions = [
      "dynamodb:BatchGetItem",
      "dynamodb:BatchWriteItem",
      "dynamodb:ConditionCheckItem",
      "dynamodb:DeleteItem",
      "dynamodb:DescribeTable",
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:Query",
      "dynamodb:Scan",
      "dynamodb:UpdateItem",
    ]
    resources = local.j2_table_arns
  }

  statement {
    sid = "UseJ2IngestionQueues"
    actions = [
      "sqs:ChangeMessageVisibility",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:GetQueueUrl",
      "sqs:ReceiveMessage",
      "sqs:SendMessage",
    ]
    resources = [
      aws_sqs_queue.j2_ingestion.arn,
      aws_sqs_queue.j2_ingestion_dlq.arn,
    ]
  }

  statement {
    sid = "UseJ2DataKey"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
    ]
    resources = [aws_kms_key.j2.arn]
  }

  statement {
    sid = "ReadWriteJ2Sources"
    actions = [
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]
    resources = [
      aws_s3_bucket.j2_sources.arn,
      "${aws_s3_bucket.j2_sources.arn}/*",
    ]
  }

  statement {
    sid = "IndexJ2Knowledge"
    actions = [
      "es:ESHttpDelete",
      "es:ESHttpGet",
      "es:ESHttpHead",
      "es:ESHttpPost",
      "es:ESHttpPut",
    ]
    resources = ["${aws_opensearch_domain.main.arn}/*"]
  }

  statement {
    sid       = "EmbedJ2Knowledge"
    actions   = ["bedrock:InvokeModel"]
    resources = ["arn:${data.aws_partition.current.partition}:bedrock:${var.aws_region}::foundation-model/${var.embedding_model_id}"]
  }

  dynamic "statement" {
    for_each = aws_secretsmanager_secret.graph_connector

    content {
      sid       = "ReadGraphConnectorCredentials"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [statement.value.arn]
    }
  }
}

resource "aws_iam_role_policy" "task_j2_data_access" {
  name   = "j2-data-access"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.j2_data_access.json
}

resource "aws_iam_role" "fixture_ingestion" {
  name               = "${local.j2_name}-fixture-ingestion"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_iam_role_policy" "fixture_ingestion" {
  name   = "fixture-ingestion-data-access"
  role   = aws_iam_role.fixture_ingestion.id
  policy = data.aws_iam_policy_document.j2_data_access.json
}

resource "aws_ecs_task_definition" "fixture_ingestion" {
  family                   = "${local.j2_name}-fixture-ingestion"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.fixture_ingestion.arn

  container_definitions = jsonencode([{
    name      = "fixture-ingestion"
    image     = var.image_uri
    essential = true
    user      = "0"
    command   = ["python", "-m", "ingestion.cli"]
    environment = concat(local.j2_environment, [
      { name = "HOME", value = "/tmp/home" },
    ])
    mountPoints = [{
      sourceVolume  = "scratch"
      containerPath = "/tmp"
      readOnly      = false
    }]
    readonlyRootFilesystem = true
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "fixture-ingestion"
      }
    }
  }])

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "scratch"
  }

  lifecycle {
    precondition {
      condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
      error_message = "Fixture ingestion requires image_uri to reference an immutable ECR digest."
    }
  }

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

resource "aws_ecs_task_definition" "graph_ingestion" {
  count                    = var.create_graph_connector_secret ? 1 : 0
  family                   = "${local.j2_name}-graph-ingestion"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.fixture_ingestion.arn

  container_definitions = jsonencode([{
    name        = "graph-ingestion"
    image       = var.image_uri
    essential   = true
    user        = "0"
    command     = ["python", "-m", "ingestion.graph_cli"]
    environment = concat(local.j2_environment, [{ name = "HOME", value = "/tmp/home" }])
    mountPoints = [{
      sourceVolume  = "scratch"
      containerPath = "/tmp"
      readOnly      = false
    }]
    readonlyRootFilesystem = true
    linuxParameters = {
      initProcessEnabled = true
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "graph-ingestion"
      }
    }
  }])

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "scratch"
  }

  lifecycle {
    precondition {
      condition     = can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
      error_message = "Graph ingestion requires image_uri to reference an immutable ECR digest."
    }
  }

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

data "aws_iam_policy_document" "graph_scheduler_assume" {
  count = var.create_graph_connector_secret ? 1 : 0

  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:${data.aws_partition.current.partition}:scheduler:${var.aws_region}:${data.aws_caller_identity.current.account_id}:schedule-group/default"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_iam_role" "graph_scheduler" {
  count              = var.create_graph_connector_secret ? 1 : 0
  name               = "${local.j2_name}-graph-scheduler"
  assume_role_policy = data.aws_iam_policy_document.graph_scheduler_assume[0].json

  tags = merge(local.common_tags, {
    SecurityDomain = var.security_domain
  })
}

data "aws_iam_policy_document" "graph_scheduler" {
  count = var.create_graph_connector_secret ? 1 : 0

  statement {
    sid       = "RunGraphIngestion"
    actions   = ["ecs:RunTask"]
    resources = [aws_ecs_task_definition.graph_ingestion[0].arn]
  }
  statement {
    sid       = "PassGraphTaskRoles"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn, aws_iam_role.fixture_ingestion.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "graph_scheduler" {
  count  = var.create_graph_connector_secret ? 1 : 0
  name   = "run-graph-ingestion"
  role   = aws_iam_role.graph_scheduler[0].id
  policy = data.aws_iam_policy_document.graph_scheduler[0].json
}

resource "aws_scheduler_schedule" "graph_ingestion" {
  count               = var.create_graph_connector_secret ? 1 : 0
  name                = "${local.j2_name}-graph-ingestion"
  description         = "Run the configured sovereign Microsoft Graph delta connector"
  schedule_expression = var.graph_ingestion_schedule_expression

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.app.arn
    role_arn = aws_iam_role.graph_scheduler[0].arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.graph_ingestion[0].arn
      launch_type         = "FARGATE"
      platform_version    = "LATEST"
      task_count          = 1

      network_configuration {
        assign_public_ip = false
        security_groups  = [aws_security_group.app.id]
        subnets          = [for subnet in aws_subnet.private : subnet.id]
      }
    }
  }
}
