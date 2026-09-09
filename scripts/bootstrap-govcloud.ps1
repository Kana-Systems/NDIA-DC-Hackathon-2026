[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')]
    [string]$GitHubRepo,

    [Parameter(Mandatory)]
    [ValidatePattern('^[1-9][0-9]*$')]
    [string]$GitHubOwnerId,

    [Parameter(Mandatory)]
    [ValidatePattern('^[1-9][0-9]*$')]
    [string]$GitHubRepositoryId,

    [ValidatePattern('^us-gov-')]
    [string]$Region = 'us-gov-west-1',

    [ValidatePattern('^[a-z][a-z0-9-]{2,23}$')]
    [string]$Project = 'contract-review',

    [ValidatePattern('^[a-z][a-z0-9-]{1,15}$')]
    [string]$SecurityDomain = 'demo',

    [ValidatePattern('^[A-Za-z0-9._/-]+$')]
    [string]$StateKey,

    [ValidatePattern('^[A-Za-z0-9+=,.@_-]{1,64}$')]
    [string]$RoleName = 'contract-review-github'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:AWS_PAGER = ''

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw 'AWS CLI v2 is required.'
}

function Invoke-Aws {
    param([Parameter(Mandatory)][string[]]$Arguments)

    & aws @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "AWS CLI failed: aws $($Arguments -join ' ')"
    }
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory)][object]$Value,
        [Parameter(Mandatory)][string]$Path
    )

    $Value | ConvertTo-Json -Depth 20 | Set-Content -Path $Path -Encoding utf8
}

$IdentityJson = & aws sts get-caller-identity --output json
if ($LASTEXITCODE -ne 0) {
    throw 'Unable to read the active AWS identity.'
}
$Identity = $IdentityJson | ConvertFrom-Json
if ($Identity.Arn -notlike 'arn:aws-us-gov:*') {
    throw 'Active credentials are not for the aws-us-gov partition.'
}

$AccountId = [string]$Identity.Account
$RepoParts = $GitHubRepo.Split('/')
$ImmutableSubject = "repo:$($RepoParts[0])@$GitHubOwnerId/$($RepoParts[1])@$GitHubRepositoryId`:environment:govcloud-demo"
$StateKey = if ([string]::IsNullOrWhiteSpace($StateKey)) {
    "$Project/$SecurityDomain/demo.tfstate"
}
else {
    $StateKey
}
if ($StateKey -notmatch '/demo\.tfstate$') {
    throw 'StateKey must end in /demo.tfstate; this bootstrap supports only the demo environment.'
}
$Bucket = "$Project-tfstate-$AccountId-$Region"
$OidcArn = "arn:aws-us-gov:iam::$AccountId`:oidc-provider/token.actions.githubusercontent.com"
$RoleArn = "arn:aws-us-gov:iam::$AccountId`:role/$RoleName"
$TempRoot = Join-Path ([IO.Path]::GetTempPath()) "contract-review-bootstrap-$([guid]::NewGuid())"
New-Item -ItemType Directory -Path $TempRoot | Out-Null

try {
    & aws s3api head-bucket --bucket $Bucket 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Aws @(
            's3api', 'create-bucket',
            '--bucket', $Bucket,
            '--region', $Region,
            '--create-bucket-configuration', "LocationConstraint=$Region"
        )
    }

    Invoke-Aws @(
        's3api', 'put-bucket-versioning',
        '--bucket', $Bucket,
        '--versioning-configuration', 'Status=Enabled'
    )
    $EncryptionConfiguration = @{
        Rules = @(
            @{
                ApplyServerSideEncryptionByDefault = @{ SSEAlgorithm = 'AES256' }
                BucketKeyEnabled                   = $true
            }
        )
    }
    $EncryptionPath = Join-Path $TempRoot 'state-encryption.json'
    Write-JsonFile -Value $EncryptionConfiguration -Path $EncryptionPath
    Invoke-Aws @(
        's3api', 'put-bucket-encryption',
        '--bucket', $Bucket,
        '--server-side-encryption-configuration', "file://$EncryptionPath"
    )
    Invoke-Aws @(
        's3api', 'put-public-access-block',
        '--bucket', $Bucket,
        '--public-access-block-configuration',
        'BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true'
    )
    Invoke-Aws @(
        's3api', 'put-bucket-ownership-controls',
        '--bucket', $Bucket,
        '--ownership-controls', 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'
    )
    Invoke-Aws @(
        's3api', 'put-bucket-tagging',
        '--bucket', $Bucket,
        '--tagging', "TagSet=[{Key=Project,Value=$Project},{Key=SecurityDomain,Value=$SecurityDomain},{Key=ManagedBy,Value=Bootstrap}]"
    )

    $BucketPolicy = @{
        Version = '2012-10-17'
        Statement = @(
            @{
                Sid       = 'DenyInsecureTransport'
                Effect    = 'Deny'
                Principal = '*'
                Action    = 's3:*'
                Resource  = @("arn:aws-us-gov:s3:::$Bucket", "arn:aws-us-gov:s3:::$Bucket/*")
                Condition = @{ Bool = @{ 'aws:SecureTransport' = 'false' } }
            }
        )
    }
    $BucketPolicyPath = Join-Path $TempRoot 'bucket-policy.json'
    Write-JsonFile -Value $BucketPolicy -Path $BucketPolicyPath
    Invoke-Aws @(
        's3api', 'put-bucket-policy',
        '--bucket', $Bucket,
        '--policy', "file://$BucketPolicyPath"
    )

    $ProviderJson = & aws iam get-open-id-connect-provider --open-id-connect-provider-arn $OidcArn --output json 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Aws @(
            'iam', 'create-open-id-connect-provider',
            '--url', 'https://token.actions.githubusercontent.com',
            '--client-id-list', 'sts.amazonaws.com',
            '--thumbprint-list',
            '6938fd4d98bab03faadb97b34396831e3780aea1',
            '1b511abead59c6ce207077c0bf0e0043b1382612',
            '--tags', "Key=Project,Value=$Project"
        )
    }
    else {
        $Provider = $ProviderJson | ConvertFrom-Json
        if ($Provider.ClientIDList -notcontains 'sts.amazonaws.com') {
            Invoke-Aws @(
                'iam', 'add-client-id-to-open-id-connect-provider',
                '--open-id-connect-provider-arn', $OidcArn,
                '--client-id', 'sts.amazonaws.com'
            )
        }
        Invoke-Aws @(
            'iam', 'update-open-id-connect-provider-thumbprint',
            '--open-id-connect-provider-arn', $OidcArn,
            '--thumbprint-list',
            '6938fd4d98bab03faadb97b34396831e3780aea1',
            '1b511abead59c6ce207077c0bf0e0043b1382612'
        )
    }

    $TrustPolicy = @{
        Version = '2012-10-17'
        Statement = @(
            @{
                Effect    = 'Allow'
                Principal = @{ Federated = $OidcArn }
                Action    = 'sts:AssumeRoleWithWebIdentity'
                Condition = @{
                    StringEquals = @{
                        'token.actions.githubusercontent.com:aud'                 = 'sts.amazonaws.com'
                        'token.actions.githubusercontent.com:sub'                 = $ImmutableSubject
                        'token.actions.githubusercontent.com:environment'         = 'govcloud-demo'
                        'token.actions.githubusercontent.com:ref'                 = 'refs/heads/main'
                        'token.actions.githubusercontent.com:repository_id'       = $GitHubRepositoryId
                        'token.actions.githubusercontent.com:repository_owner_id' = $GitHubOwnerId
                    }
                }
            }
        )
    }
    $TrustPolicyPath = Join-Path $TempRoot 'trust-policy.json'
    Write-JsonFile -Value $TrustPolicy -Path $TrustPolicyPath

    & aws iam get-role --role-name $RoleName 2>$null
    if ($LASTEXITCODE -eq 0) {
        Invoke-Aws @(
            'iam', 'update-assume-role-policy',
            '--role-name', $RoleName,
            '--policy-document', "file://$TrustPolicyPath"
        )
    }
    else {
        Invoke-Aws @(
            'iam', 'create-role',
            '--role-name', $RoleName,
            '--assume-role-policy-document', "file://$TrustPolicyPath",
            '--max-session-duration', '3600',
            '--description', "GitHub OIDC deployment role for $Project",
            '--tags', "Key=Project,Value=$Project"
        )
    }

    $DeployPolicy = @{
        Version = '2012-10-17'
        Statement = @(
            @{
                Sid      = 'TerraformState'
                Effect   = 'Allow'
                Action   = @('s3:ListBucket', 's3:GetBucketLocation', 's3:GetBucketVersioning')
                Resource = "arn:aws-us-gov:s3:::$Bucket"
            },
            @{
                Sid      = 'TerraformStateObjects'
                Effect   = 'Allow'
                Action   = @('s3:GetObject', 's3:PutObject', 's3:DeleteObject')
                Resource = "arn:aws-us-gov:s3:::$Bucket/*"
            },
            @{
                Sid      = 'EcrAuthorization'
                Effect   = 'Allow'
                Action   = 'ecr:GetAuthorizationToken'
                Resource = '*'
            },
            @{
                Sid      = 'ProjectEcr'
                Effect   = 'Allow'
                Action   = 'ecr:*'
                Resource = "arn:aws-us-gov:ecr:$Region`:$AccountId`:repository/$Project-*"
            },
            @{
                Sid    = 'ProjectIam'
                Effect = 'Allow'
                Action = @(
                    'iam:CreateRole', 'iam:DeleteRole', 'iam:GetRole', 'iam:TagRole', 'iam:UntagRole',
                    'iam:ListRoleTags', 'iam:UpdateAssumeRolePolicy', 'iam:UpdateRole',
                    'iam:UpdateRoleDescription', 'iam:ListInstanceProfilesForRole',
                    'iam:PutRolePolicy', 'iam:GetRolePolicy', 'iam:DeleteRolePolicy', 'iam:ListRolePolicies',
                    'iam:AttachRolePolicy', 'iam:DetachRolePolicy', 'iam:ListAttachedRolePolicies'
                )
                Resource = "arn:aws-us-gov:iam::$AccountId`:role/$Project-*"
            },
            @{
                Sid    = 'ProjectManagedPolicies'
                Effect = 'Allow'
                Action = @(
                    'iam:CreatePolicy', 'iam:DeletePolicy', 'iam:GetPolicy',
                    'iam:CreatePolicyVersion', 'iam:DeletePolicyVersion', 'iam:GetPolicyVersion',
                    'iam:ListPolicyVersions', 'iam:ListEntitiesForPolicy',
                    'iam:TagPolicy', 'iam:UntagPolicy'
                )
                Resource = "arn:aws-us-gov:iam::$AccountId`:policy/$Project-*"
            },
            @{
                Sid       = 'PassProjectRoles'
                Effect    = 'Allow'
                Action    = 'iam:PassRole'
                Resource  = "arn:aws-us-gov:iam::$AccountId`:role/$Project-*"
                Condition = @{ StringEquals = @{ 'iam:PassedToService' = 'ecs-tasks.amazonaws.com' } }
            },
            @{
                Sid      = 'CreateRequiredServiceLinkedRoles'
                Effect   = 'Allow'
                Action   = 'iam:CreateServiceLinkedRole'
                Resource = '*'
                Condition = @{
                    StringEquals = @{
                        'iam:AWSServiceName' = @(
                            'application-autoscaling.amazonaws.com',
                            'ecs.amazonaws.com',
                            'ecs.application-autoscaling.amazonaws.com',
                            'es.amazonaws.com',
                            'elasticloadbalancing.amazonaws.com',
                            'opensearchservice.amazonaws.com'
                        )
                    }
                }
            },
            @{
                Sid    = 'ProvisionProjectInfrastructure'
                Effect = 'Allow'
                Action = @(
                    'acm:DescribeCertificate', 'acm:GetCertificate', 'acm:ListCertificates',
                    'application-autoscaling:*', 'cloudwatch:*', 'ec2:*',
                    'dynamodb:*', 'ecs:*', 'elasticloadbalancing:*', 'es:*', 'kms:*', 'logs:*',
                    'resource-groups:*', 'scheduler:*', 'secretsmanager:*', 'sqs:*'
                )
                Resource = '*'
            },
            @{
                Sid      = 'ProjectBuckets'
                Effect   = 'Allow'
                Action   = 's3:*'
                Resource = @(
                    "arn:aws-us-gov:s3:::$Project-*",
                    "arn:aws-us-gov:s3:::$Project-*/*"
                )
            }
        )
    }
    $DeployPolicyPath = Join-Path $TempRoot 'deploy-policy.json'
    Write-JsonFile -Value $DeployPolicy -Path $DeployPolicyPath
    Invoke-Aws @(
        'iam', 'put-role-policy',
        '--role-name', $RoleName,
        '--policy-name', "$Project-terraform-deploy",
        '--policy-document', "file://$DeployPolicyPath"
    )
    $ClassifierPullPolicyPath = Join-Path $PSScriptRoot 'iam/classifier-base-image-pull.json'
    Invoke-Aws @(
        'iam', 'put-role-policy',
        '--role-name', $RoleName,
        '--policy-name', 'classifier-base-image-pull',
        '--policy-document', "file://$ClassifierPullPolicyPath"
    )
}
finally {
    Remove-Item -Path $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

@"
Bootstrap complete.

GitHub Actions secrets (prefer the `govcloud-demo` environment):
  AWS_GOV_REGION=$Region
  AWS_GOV_ROLE_ARN=$RoleArn
  TF_STATE_BUCKET=$Bucket
  TF_STATE_KEY=$StateKey
  SECURITY_DOMAIN=$SecurityDomain
  CREATE_GRAPH_CONNECTOR_SECRET=false

Immutable GitHub identity:
  OWNER_ID=$GitHubOwnerId
  REPOSITORY_ID=$GitHubRepositoryId

Also configure:
  APP_DOMAIN                   externally managed hostname, such as ndia.kana.systems
  ALLOWED_INGRESS_CIDRS_JSON   reviewed JSON list, such as ["192.0.2.10/32"]

No static AWS access key is required.
"@ | Write-Host
