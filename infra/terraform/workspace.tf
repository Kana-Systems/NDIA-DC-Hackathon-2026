# Persistent Lens data is separated from the public model/corpus artifacts.
resource "aws_dynamodb_table" "lens_workspace" {
  name         = "${local.name}-${var.security_domain}-lens"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "scope"
  range_key    = "id"

  attribute {
    name = "scope"
    type = "S"
  }
  attribute {
    name = "id"
    type = "S"
  }
  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.j2.arn
  }
  point_in_time_recovery {
    enabled = true
  }
  deletion_protection_enabled = true
  tags                        = local.common_tags
}
