variable "aws_region" {
  description = "AWS GovCloud region."
  type        = string
  default     = "us-gov-west-1"

  validation {
    condition     = startswith(var.aws_region, "us-gov-")
    error_message = "aws_region must be an AWS GovCloud region."
  }
}

variable "project_name" {
  description = "Short lowercase name used in resource names."
  type        = string
  default     = "contract-review"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,23}$", var.project_name))
    error_message = "project_name must be 3-24 lowercase letters, numbers, or hyphens."
  }
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "demo"

  validation {
    condition     = var.environment == "demo"
    error_message = "This repository provisions only the demo environment."
  }
}

variable "security_domain" {
  description = "Lowercase security-domain namespace applied to J2 resources; it does not assert an accreditation or classification."
  type        = string
  default     = "demo"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,15}$", var.security_domain))
    error_message = "security_domain must be 2-16 lowercase letters, numbers, or hyphens."
  }
}

variable "create_graph_connector_secret" {
  description = "Create an empty Secrets Manager container for externally supplied Microsoft Graph connector credentials."
  type        = bool
  default     = false
}

variable "graph_ingestion_schedule_expression" {
  description = "EventBridge Scheduler expression for the optional Microsoft Graph delta connector."
  type        = string
  default     = "rate(15 minutes)"

  validation {
    condition     = can(regex("^(rate|cron)\\(", var.graph_ingestion_schedule_expression))
    error_message = "graph_ingestion_schedule_expression must be an EventBridge rate or cron expression."
  }
}

variable "hosted_zone_id" {
  description = "ID of an existing public Route53 hosted zone in the GovCloud account."
  type        = string

  validation {
    condition     = can(regex("^Z[A-Z0-9]+$", var.hosted_zone_id))
    error_message = "hosted_zone_id must be a Route53 hosted zone ID."
  }
}

variable "domain_name" {
  description = "Existing subdomain to assign to the ALB, for example review.example.mil."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+$", var.domain_name))
    error_message = "domain_name must be a lowercase fully qualified domain name."
  }
}

variable "image_uri" {
  description = "Immutable ECR image URI including an @sha256 digest. A placeholder is sufficient for the ECR-only bootstrap target."
  type        = string
  default     = "public.ecr.aws/docker/library/busybox:1.36"

  validation {
    condition     = var.image_uri == "public.ecr.aws/docker/library/busybox:1.36" || can(regex("@sha256:[0-9a-f]{64}$", var.image_uri))
    error_message = "image_uri must use an immutable sha256 digest."
  }
}

variable "container_port" {
  description = "Application container port."
  type        = number
  default     = 8080

  validation {
    condition     = var.container_port == 8080
    error_message = "The integrated application contract requires port 8080."
  }
}

variable "health_check_path" {
  description = "Unauthenticated health endpoint used by the ALB and deployment smoke test."
  type        = string
  default     = "/health"

  validation {
    condition     = var.health_check_path == "/health"
    error_message = "The integrated application health endpoint is /health."
  }
}

variable "judge_username" {
  description = "Username placed in the generated judge credential secret."
  type        = string
  default     = "judge"
  sensitive   = true
}

variable "bedrock_model_id" {
  description = "OpenAI model ID sent to the signed GovCloud bedrock-mantle Responses API."
  type        = string
  default     = "openai.gpt-5.6-terra"

  validation {
    condition     = var.bedrock_model_id == "openai.gpt-5.6-terra"
    error_message = "This demo is scoped to the GovCloud GPT-5.6 Terra model."
  }
}

variable "embedding_model_id" {
  description = "Bedrock embedding model used for OpenSearch vector retrieval."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"

  validation {
    condition     = var.embedding_model_id == "amazon.titan-embed-text-v2:0"
    error_message = "This deployment uses Titan Text Embeddings V2."
  }
}

variable "embedding_dimensions" {
  description = "Vector dimensions used by Titan and the OpenSearch index mapping."
  type        = number
  default     = 1024

  validation {
    condition     = var.embedding_dimensions == 1024
    error_message = "The integrated OpenSearch mapping requires 1024 dimensions."
  }
}

variable "opensearch_index" {
  description = "OpenSearch index containing normalized policy knowledge."
  type        = string
  default     = "government-contract-knowledge-v1"

  validation {
    condition     = var.opensearch_index == "government-contract-knowledge-v1"
    error_message = "The integrated application expects government-contract-knowledge-v1."
  }
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget in USD."
  type        = number
  default     = 250

  validation {
    condition     = var.monthly_budget_usd >= 10
    error_message = "monthly_budget_usd must be at least 10."
  }
}

variable "budget_alert_email" {
  description = "Email address that receives budget threshold notifications."
  type        = string

  validation {
    condition     = can(regex("^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", var.budget_alert_email))
    error_message = "budget_alert_email must be a valid email address."
  }
}

variable "desired_count" {
  description = "Number of Fargate tasks for the demo."
  type        = number
  default     = 1

  validation {
    condition     = var.desired_count >= 1 && var.desired_count <= 4
    error_message = "desired_count must be between 1 and 4."
  }
}

variable "allowed_ingress_cidrs" {
  description = "Explicit IPv4 CIDRs allowed to reach HTTPS; no public-access default is provided."
  type        = list(string)

  validation {
    condition     = length(var.allowed_ingress_cidrs) > 0 && alltrue([for cidr in var.allowed_ingress_cidrs : can(cidrnetmask(cidr))])
    error_message = "allowed_ingress_cidrs must contain valid CIDR blocks."
  }
}
