data "aws_ssm_parameter" "trusted_ingress_cidrs" {
  name = var.trusted_ingress_cidrs_parameter_name
}

locals {
  trusted_ingress_cidrs = try(
    tolist(jsondecode(nonsensitive(data.aws_ssm_parameter.trusted_ingress_cidrs.value))),
    []
  )
  effective_allowed_ingress_cidrs = distinct(
    concat(var.allowed_ingress_cidrs, local.trusted_ingress_cidrs)
  )
}

check "trusted_ingress_cidrs" {
  assert {
    condition = (
      length(local.trusted_ingress_cidrs) > 0
      && alltrue([
        for cidr in local.trusted_ingress_cidrs :
        can(cidrnetmask(cidr)) && try(tonumber(split("/", cidr)[1]) >= 27, false)
      ])
    )
    error_message = "The trusted ingress SSM parameter must contain a non-empty JSON array of IPv4 CIDRs no broader than /27."
  }
}

resource "aws_wafv2_ip_set" "trusted_ingress" {
  name               = "${local.name}-trusted-ingress"
  description        = "Approved public egress addresses for the GovCloud application"
  scope              = "REGIONAL"
  ip_address_version = "IPV4"
  addresses          = local.effective_allowed_ingress_cidrs

  tags = merge(local.common_tags, {
    Name = "${local.name}-trusted-ingress"
  })
}

resource "aws_wafv2_web_acl" "app" {
  name        = "${local.name}-web-acl"
  description = "Defense-in-depth controls for the contract review ALB"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  rule {
    name     = "TrustedIngressOnly"
    priority = 0

    action {
      block {}
    }

    statement {
      not_statement {
        statement {
          ip_set_reference_statement {
            arn = aws_wafv2_ip_set.trusted_ingress.arn
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-untrusted-ingress"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "TrustedSourceRateLimit"
    priority = 10

    action {
      block {}
    }

    statement {
      rate_based_statement {
        aggregate_key_type = "IP"
        limit              = var.waf_rate_limit
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "AmazonIpReputation"
    priority = 20

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesAmazonIpReputationList"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-ip-reputation"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "KnownBadInputs"
    priority = 30

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-known-bad-inputs"
      sampled_requests_enabled   = true
    }
  }

  rule {
    name     = "CommonWebExploits"
    priority = 40

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"

        # Contract uploads are intentionally larger than WAF's ALB body-inspection
        # window. Retain the signal without blocking every legitimate upload.
        rule_action_override {
          name = "SizeRestrictions_BODY"

          action_to_use {
            count {}
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-common-exploits"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.name}-web-acl"
    sampled_requests_enabled   = true
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-web-acl"
  })
}

resource "aws_wafv2_web_acl_association" "app" {
  resource_arn = aws_lb.app.arn
  web_acl_arn  = aws_wafv2_web_acl.app.arn
}

resource "aws_cloudwatch_log_group" "waf" {
  name              = "aws-waf-logs-${local.name}"
  retention_in_days = 30

  tags = merge(local.common_tags, {
    Name = "aws-waf-logs-${local.name}"
  })
}

resource "aws_wafv2_web_acl_logging_configuration" "app" {
  resource_arn            = aws_wafv2_web_acl.app.arn
  log_destination_configs = [aws_cloudwatch_log_group.waf.arn]

  redacted_fields {
    single_header {
      name = "authorization"
    }
  }

  redacted_fields {
    single_header {
      name = "cookie"
    }
  }

  logging_filter {
    default_behavior = "DROP"

    filter {
      behavior = "KEEP"

      condition {
        action_condition {
          action = "BLOCK"
        }
      }

      requirement = "MEETS_ANY"
    }
  }
}
