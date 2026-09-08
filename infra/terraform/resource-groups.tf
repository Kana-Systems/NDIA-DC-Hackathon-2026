resource "aws_resourcegroups_group" "ndia" {
  name        = "ndia"
  description = "Taggable resources managed for the NDIA contract-review demo."

  resource_query {
    type = "TAG_FILTERS_1_0"
    query = jsonencode({
      ResourceTypeFilters = ["AWS::AllSupported"]
      TagFilters = [
        {
          Key    = "Project"
          Values = [var.project_name]
        },
        {
          Key    = "Environment"
          Values = [var.environment]
        },
      ]
    })
  }

  tags = merge(local.common_tags, { Name = "ndia" })
}
