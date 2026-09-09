"""Interactive local setup: secret input is hidden and never saved to the repo."""

import argparse
import getpass
import json
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingestion.graph import GraphClient  # noqa: E402


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--tenant-id")
    args.add_argument("--client-id")
    args.add_argument("--site-url")
    supplied = args.parse_args()
    tenant = supplied.tenant_id or input("Directory (tenant) ID: ").strip()
    app_id = supplied.client_id or input("Application (client) ID: ").strip()
    site_url = (
        supplied.site_url or input("Approved SharePoint SITE URL (not a file link): ").strip()
    )
    url = urlsplit(site_url)
    if url.scheme != "https" or not (url.hostname or "").endswith(".sharepoint.com"):
        raise ValueError("This setup command accepts a standard Microsoft 365 SharePoint site")
    secret_value = getpass.getpass("Client secret VALUE (hidden; not the secret ID): ")
    client = GraphClient(client_id=app_id, client_secret=secret_value, tenant_id=tenant)
    path = quote(url.path.rstrip("/") or "/", safe="/")
    site = client.get_json(f"/sites/{url.hostname}:{path}")
    drives, cursor = [], f"/sites/{quote(site['id'], safe='')}/drives"
    while cursor:
        page = client.get_json(cursor)
        drives.extend(page.get("value", []))
        cursor = page.get("@odata.nextLink", "")
    if not drives:
        raise ValueError(
            "No accessible library. Grant the app read access to this selected site first"
        )
    for index, drive in enumerate(drives, 1):
        print(f"{index}: {drive['name']}")
    selection = int(input("Library number to import: ")) - 1
    if selection < 0 or selection >= len(drives):
        raise ValueError("Invalid library selection")
    drive = drives[selection]
    if (
        input(
            "Confirm this library contains ONLY public/unclassified hackathon documents "
            "and may be shared with all judge-login users. Type APPROVE: "
        )
        != "APPROVE"
    ):
        raise ValueError("Library was not approved; no credentials stored")
    config = {
        "tenant_id": tenant,
        "client_id": app_id,
        "client_secret": secret_value,
        "site_id": site["id"],
        "drive_id": drive["id"],
        "graph_base_url": "https://graph.microsoft.com/v1.0",
        "approved_for_demo": True,
        "security_domain": "demo",
        "allowed_workspace_subjects": ["judge"],
    }
    region = "us-gov-west-1"
    aws = boto3.client("secretsmanager", region_name=region)
    name = "contract-review-demo/lens-sharepoint"
    try:
        existing = aws.describe_secret(SecretId=name)
    except aws.exceptions.ResourceNotFoundException:
        existing = None
    if existing:
        if (
            input("A connector secret already exists. Replace its configuration? Type REPLACE: ")
            != "REPLACE"
        ):
            raise ValueError("Existing configuration was not changed")
        result = aws.put_secret_value(SecretId=name, SecretString=json.dumps(config))
    else:
        result = aws.create_secret(
            Name=name,
            KmsKeyId="alias/contract-review-demo-demo-j2",
            Description="Approved Lens SharePoint library connector",
            SecretString=json.dumps(config),
            Tags=[
                {"Key": "Project", "Value": "contract-review"},
                {"Key": "Environment", "Value": "demo"},
            ],
        )
    print("SharePoint access verified and credentials stored securely in AWS.")
    print("Connector secret ARN (safe to share):", result["ARN"])
    print("Deployment must set LENS_GRAPH_SECRET_ARN to this ARN. No deployment was started.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never dump HTTP responses, tokens, secret payloads, or SDK request parameters.
        print(
            f"Setup stopped ({type(error).__name__}). Check app credentials, "
            "selected-site read grant, and AWS permissions.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
