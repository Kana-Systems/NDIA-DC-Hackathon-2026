output "application_url" {
  description = "Public HTTPS URL."
  value       = "https://${var.domain_name}"
}

output "external_dns_cname_name" {
  description = "Hostname to create as a CNAME in the external DNS provider."
  value       = var.domain_name
}

output "external_dns_cname_target" {
  description = "ALB DNS name to use as the external CNAME target."
  value       = aws_lb.app.dns_name
}

output "acm_certificate_arn" {
  description = "Issued externally validated ACM certificate selected for the application."
  value       = data.aws_acm_certificate.app.arn
}

output "target_group_arn" {
  description = "Application target group checked by the deployment workflow."
  value       = aws_lb_target_group.app.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL used by CI."
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.app.name
}

output "ecs_service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.app.name
}

output "knowledge_index_task_definition_arn" {
  description = "Immutable-image task definition used to seed OpenSearch."
  value       = aws_ecs_task_definition.knowledge_index.arn
}

output "private_subnet_ids" {
  description = "Private subnet IDs used by one-shot Fargate tasks."
  value       = aws_subnet.private[*].id
}

output "app_security_group_id" {
  description = "Application security group permitted to reach OpenSearch."
  value       = aws_security_group.app.id
}

output "uploads_bucket_name" {
  description = "Encrypted temporary upload bucket."
  value       = aws_s3_bucket.uploads.id
}

output "artifacts_bucket_name" {
  description = "Encrypted model and report artifact bucket."
  value       = aws_s3_bucket.artifacts.id
}

output "judge_credentials_secret_arn" {
  description = "Secret ARN containing the generated judge credentials."
  value       = aws_secretsmanager_secret.judge.arn
}

output "opensearch_endpoint" {
  description = "Private OpenSearch endpoint."
  value       = "https://${aws_opensearch_domain.main.endpoint}"
}

output "security_domain" {
  description = "Namespace applied to the deployed J2 resources."
  value       = var.security_domain
}

output "j2_kms_key_arn" {
  description = "Customer-managed key ARN used by J2 data stores."
  value       = aws_kms_key.j2.arn
}

output "j2_source_bucket_name" {
  description = "Versioned KMS-encrypted source-object bucket."
  value       = aws_s3_bucket.j2_sources.id
}

output "j2_enterprise_index" {
  description = "ACL-filtered OpenSearch index for enterprise intelligence chunks."
  value       = local.j2_enterprise_index
}

output "j2_ingestion_queue_url" {
  description = "URL of the encrypted J2 ingestion queue."
  value       = aws_sqs_queue.j2_ingestion.url
}

output "j2_ingestion_dlq_url" {
  description = "URL of the encrypted J2 ingestion dead-letter queue."
  value       = aws_sqs_queue.j2_ingestion_dlq.url
}

output "j2_document_table_name" {
  description = "Name of the J2 document manifest table."
  value       = aws_dynamodb_table.j2_documents.name
}

output "j2_entity_table_name" {
  description = "Name of the J2 entity table."
  value       = aws_dynamodb_table.j2_entities.name
}

output "j2_change_table_name" {
  description = "Name of the J2 change-event table."
  value       = aws_dynamodb_table.j2_changes.name
}

output "graph_connector_secret_arn" {
  description = "ARN of the optional externally populated Microsoft Graph connector secret, or null."
  value       = try(aws_secretsmanager_secret.graph_connector[0].arn, null)
}

output "fixture_ingestion_task_definition_arn" {
  description = "Digest-pinned task definition used for deterministic fixture ingestion."
  value       = aws_ecs_task_definition.fixture_ingestion.arn
}

output "graph_ingestion_schedule_arn" {
  description = "ARN of the optional scheduled Graph delta ingestion job."
  value       = try(aws_scheduler_schedule.graph_ingestion[0].arn, null)
}

output "ndia_resource_group_arn" {
  description = "ARN of the tag-based AWS Resource Group for this deployment."
  value       = aws_resourcegroups_group.ndia.arn
}

output "sagemaker_training_bucket_name" {
  description = "Encrypted bucket for SageMaker training inputs, outputs, and checkpoints."
  value       = aws_s3_bucket.sagemaker_training.id
}

output "sagemaker_training_input_uri" {
  description = "S3 prefix for model-family-specific SageMaker training inputs."
  value       = "s3://${aws_s3_bucket.sagemaker_training.id}/input"
}

output "sagemaker_training_output_uri" {
  description = "S3 prefix where SageMaker writes model-family-specific artifacts."
  value       = "s3://${aws_s3_bucket.sagemaker_training.id}/output"
}

output "sagemaker_training_role_arn" {
  description = "Least-privilege execution role assumed by SageMaker training jobs."
  value       = aws_iam_role.sagemaker_training.arn
}

output "sagemaker_submitter_policy_arn" {
  description = "Policy to attach to an approved developer identity that submits training jobs."
  value       = aws_iam_policy.sagemaker_submitter.arn
}

output "classifier_endpoint_name" {
  description = "Private SageMaker classifier endpoint used by the application, or null when the packaged rollback is active."
  value       = local.effective_classifier_endpoint_name == "" ? null : local.effective_classifier_endpoint_name
}
