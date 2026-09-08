locals {
  sagemaker_training_name = "${local.name}-classifier-training"
}

resource "aws_s3_bucket" "sagemaker_training" {
  bucket        = "${local.name}-training-${data.aws_caller_identity.current.account_id}"
  force_destroy = false

  tags = merge(local.common_tags, {
    Name           = local.sagemaker_training_name
    SecurityDomain = var.security_domain
  })
}

resource "aws_s3_bucket_versioning" "sagemaker_training" {
  bucket = aws_s3_bucket.sagemaker_training.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sagemaker_training" {
  bucket = aws_s3_bucket.sagemaker_training.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.j2.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "sagemaker_training" {
  bucket                  = aws_s3_bucket.sagemaker_training.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "sagemaker_training" {
  bucket = aws_s3_bucket.sagemaker_training.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "sagemaker_training" {
  bucket = aws_s3_bucket.sagemaker_training.id

  rule {
    id     = "expire-incomplete-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "expire-checkpoints"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    expiration {
      days = 30
    }
  }
}

data "aws_iam_policy_document" "sagemaker_training_bucket" {
  statement {
    sid     = "DenyInsecureTransport"
    effect  = "Deny"
    actions = ["s3:*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    resources = [
      aws_s3_bucket.sagemaker_training.arn,
      "${aws_s3_bucket.sagemaker_training.arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "sagemaker_training" {
  bucket = aws_s3_bucket.sagemaker_training.id
  policy = data.aws_iam_policy_document.sagemaker_training_bucket.json
}

data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_training" {
  name                 = local.sagemaker_training_name
  assume_role_policy   = data.aws_iam_policy_document.sagemaker_assume.json
  max_session_duration = 3600

  tags = merge(local.common_tags, {
    Name           = local.sagemaker_training_name
    SecurityDomain = var.security_domain
  })
}

data "aws_iam_policy_document" "sagemaker_training" {
  statement {
    sid       = "ListTrainingBucket"
    actions   = ["s3:GetBucketLocation", "s3:ListBucket"]
    resources = [aws_s3_bucket.sagemaker_training.arn]
  }

  statement {
    sid = "ReadWriteTrainingArtifacts"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:DeleteObject",
      "s3:GetObject",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
    ]
    resources = ["${aws_s3_bucket.sagemaker_training.arn}/*"]
  }

  statement {
    sid = "UseTrainingDataKey"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:ReEncryptFrom",
      "kms:ReEncryptTo",
    ]
    resources = [aws_kms_key.j2.arn]
  }

  statement {
    sid       = "AuthenticateToTrainingImages"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PullTrainingImages"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = ["*"]
  }

  statement {
    sid = "WriteTrainingLogs"
    actions = [
      "cloudwatch:PutMetricData",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_training" {
  name   = "classifier-training"
  role   = aws_iam_role.sagemaker_training.id
  policy = data.aws_iam_policy_document.sagemaker_training.json
}

data "aws_iam_policy_document" "sagemaker_submitter" {
  statement {
    sid       = "CreateApprovedTrainingJobs"
    actions   = ["sagemaker:CreateTrainingJob"]
    resources = ["*"]

    condition {
      test     = "ForAllValues:StringEquals"
      variable = "sagemaker:InstanceTypes"
      values   = sort(tolist(var.sagemaker_allowed_training_instance_types))
    }
  }

  statement {
    sid = "ManageTrainingJobs"
    actions = [
      "sagemaker:AddTags",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:ListTrainingJobs",
      "sagemaker:StopTrainingJob",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "PassClassifierTrainingRole"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_training.arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  statement {
    sid       = "ListClassifierTrainingBucket"
    actions   = ["s3:GetBucketLocation", "s3:ListBucket"]
    resources = [aws_s3_bucket.sagemaker_training.arn]
  }

  statement {
    sid = "UploadClassifierTrainingData"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:ListMultipartUploadParts",
      "s3:PutObject",
    ]
    resources = ["${aws_s3_bucket.sagemaker_training.arn}/input/*"]
  }

  statement {
    sid = "EncryptClassifierTrainingData"
    actions = [
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
    ]
    resources = [aws_kms_key.j2.arn]
  }
}

resource "aws_iam_policy" "sagemaker_submitter" {
  name        = "${local.sagemaker_training_name}-submitter"
  description = "Submit and monitor approved SageMaker training jobs; attach only to developers."
  policy      = data.aws_iam_policy_document.sagemaker_submitter.json
  tags        = local.common_tags
}
