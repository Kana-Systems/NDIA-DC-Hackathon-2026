SHELL := /usr/bin/env bash
TF_DIR := infra/terraform
TF_VARS ?= terraform.tfvars

.PHONY: bootstrap tf-init tf-fmt tf-validate tf-plan tf-apply tf-destroy

bootstrap:
	@test -n "$(GITHUB_REPO)" || (echo "Set GITHUB_REPO=owner/repo" >&2; exit 2)
	@test -n "$(GITHUB_OWNER_ID)" || (echo "Set GITHUB_OWNER_ID to the numeric owner ID" >&2; exit 2)
	@test -n "$(GITHUB_REPOSITORY_ID)" || (echo "Set GITHUB_REPOSITORY_ID to the numeric repository ID" >&2; exit 2)
	pwsh -NoProfile -File scripts/bootstrap-govcloud.ps1 \
		-GitHubRepo "$(GITHUB_REPO)" \
		-GitHubOwnerId "$(GITHUB_OWNER_ID)" \
		-GitHubRepositoryId "$(GITHUB_REPOSITORY_ID)"

tf-init:
	@test -n "$(TF_STATE_BUCKET)" || (echo "Set TF_STATE_BUCKET" >&2; exit 2)
	terraform -chdir=$(TF_DIR) init \
		-backend-config="bucket=$(TF_STATE_BUCKET)" \
		-backend-config="key=$(or $(TF_STATE_KEY),contract-review/demo.tfstate)" \
		-backend-config="region=$(or $(AWS_REGION),us-gov-west-1)" \
		-backend-config="use_lockfile=true"

tf-fmt:
	terraform -chdir=$(TF_DIR) fmt -recursive

tf-validate:
	terraform -chdir=$(TF_DIR) validate

tf-plan:
	terraform -chdir=$(TF_DIR) plan -var-file="$(TF_VARS)" -out=tfplan

tf-apply:
	terraform -chdir=$(TF_DIR) apply tfplan

tf-destroy:
	@test "$(CONFIRM_DESTROY)" = "contract-review" || \
		(echo "Set CONFIRM_DESTROY=contract-review" >&2; exit 2)
	terraform -chdir=$(TF_DIR) destroy -var-file="$(TF_VARS)"
