"""Approved-library SharePoint ingestion for the shared-identity hackathon workspace."""

import json
import logging
from pathlib import PurePosixPath
from urllib.parse import quote, urlsplit

import requests
from fastapi import HTTPException

from app.parsers import DocumentParser
from ingestion.graph import GraphClient

logger = logging.getLogger(__name__)


def source_error(error, stage):
    """Keep signed URLs, credentials and document text out of sync diagnostics."""
    if isinstance(error, requests.RequestException):
        status = error.response.status_code if error.response is not None else None
        if status in {401, 403}:
            return "SharePoint denied access to this file. Check its library permissions."
        if status == 404:
            return "File was moved or removed during sync. Sync again to refresh the library."
        if status in {408, 429, 500, 502, 503, 504} or isinstance(
            error,
            (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError),
        ):
            return (
                "SharePoint download is temporarily unavailable after retries. Try syncing again."
            )
        return "SharePoint could not download this file. Check its availability and access."
    if isinstance(error, UnicodeError):
        return "Text encoding is unsupported. Save the file as UTF-8 and sync again."
    known = {
        "Document exceeds upload size limit": "File exceeds the configured upload size limit.",
        "Sensitivity-labelled files require a production access workflow": (
            "Sensitivity-labelled files require a production access workflow."
        ),
        "Document parsing timed out": (
            "Parsing timed out. Split or simplify the file and sync again."
        ),
    }
    if isinstance(error, ValueError) and str(error) in known:
        return known[str(error)]
    if stage == "parse":
        return "File could not be parsed. Check that it is readable and contains extractable text."
    return "File download failed. Check its availability and access, then sync again."


def configured_client(settings, principal):
    import boto3

    if not settings.graph_connector_secret_arn:
        raise HTTPException(409, "SharePoint credentials have not been configured in AWS")
    secret = boto3.client("secretsmanager", region_name=settings.aws_region).get_secret_value(
        SecretId=settings.graph_connector_secret_arn
    )
    config = json.loads(secret["SecretString"])
    required = ("tenant_id", "client_id", "client_secret", "site_id", "drive_id")
    if any(not config.get(key) for key in required):
        raise HTTPException(
            409, "SharePoint secret is missing tenant, app, site or library settings"
        )
    # App-only permission is not the individual user's permission. No inferred sharing.
    if (
        config.get("approved_for_demo") is not True
        or config.get("security_domain") != principal.security_domain
        or principal.subject not in config.get("allowed_workspace_subjects", [])
    ):
        raise HTTPException(
            403, "Administrator must explicitly approve this library for your workspace"
        )
    client = GraphClient(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        tenant_id=config["tenant_id"],
        graph_base_url=config.get("graph_base_url", "https://graph.microsoft.com/v1.0"),
    )
    return client, config


def check_connection(settings, principal):
    client, config = configured_client(settings, principal)
    site = client.get_json(f"/sites/{quote(config['site_id'], safe='')}")
    drive = client.get_json(f"/drives/{quote(config['drive_id'], safe='')}")
    # Check that this drive belongs to the explicitly configured site, not another readable site.
    drives, url, visited = [], f"/sites/{quote(config['site_id'], safe='')}/drives", set()
    while url:
        if url in visited:
            raise ValueError("Repeated Graph pagination URL")
        visited.add(url)
        page = client.get_json(url)
        drives.extend(page.get("value", []))
        url = page.get("@odata.nextLink", "")
    if not any(d["id"] == config["drive_id"] for d in drives):
        raise HTTPException(403, "The configured library does not belong to the approved site")
    return (
        client,
        config,
        {
            "site": site.get("displayName", "Approved site"),
            "library": drive.get("name", "Approved library"),
            "url": drive.get("webUrl", ""),
            "status": "connected",
        },
    )


def source_files(connection, settings, principal):
    from app.workspace import parsed_text

    client, config, _ = check_connection(settings, principal)
    drive = quote(config["drive_id"], safe="")
    folder = connection.get("folder", "").strip("/")
    if ".." in PurePosixPath(folder).parts or ":" in folder or "\\" in folder:
        raise HTTPException(422, "Choose a relative folder within the approved document library")
    root = f"/drives/{drive}/root"
    if folder:
        root += ":/" + quote(folder, safe="/") + ":"
    queue, visited = [root + "/children?$top=200"], set()
    parser = DocumentParser(settings)
    while queue:
        url = queue.pop(0)
        if url in visited:
            raise ValueError("Repeated Graph pagination URL")
        visited.add(url)
        page = client.get_json(url)
        if page.get("@odata.nextLink"):
            queue.append(page["@odata.nextLink"])
        for item in page.get("value", []):
            if "remoteItem" in item:
                continue  # Do not follow shortcuts outside the approved library.
            key = item["id"]
            if "folder" in item:
                queue.append(f"/drives/{drive}/items/{quote(key, safe='')}/children?$top=200")
                continue
            name = item.get("name", "")
            if PurePosixPath(name).suffix.lower() not in {".txt", ".md", ".pdf", ".docx"}:
                continue
            stage = "download"
            try:
                label = item.get("sensitivityLabel")
                if label:
                    raise ValueError(
                        "Sensitivity-labelled files require a production access workflow"
                    )
                if item.get("size", 0) > settings.max_upload_bytes:
                    raise ValueError("Document exceeds upload size limit")
                raw = client.download(
                    f"/drives/{drive}/items/{quote(key, safe='')}/content",
                    settings.max_upload_bytes,
                )
                stage = "parse"
                doc = (
                    parser.parse_isolated(name, raw)
                    if name.lower().endswith((".pdf", ".docx"))
                    else parsed_text(name, raw.decode("utf-8-sig"))
                )
                source_url = item.get("webUrl", "")
                if urlsplit(source_url).scheme != "https":
                    source_url = ""
                yield (
                    key,
                    doc,
                    None,
                    {
                        "raw_bytes": raw,
                        "source_url": source_url,
                        "source_version": item.get("eTag", ""),
                        "source_provider": "sharepoint",
                    },
                )
            except Exception as error:
                response = getattr(error, "response", None)
                logger.warning(
                    "SharePoint file sync failed: item=%s stage=%s error=%s status=%s",
                    key,
                    stage,
                    type(error).__name__,
                    response.status_code if response is not None else None,
                )
                yield (
                    key,
                    None,
                    source_error(error, stage),
                    {"source_name": name},
                )
