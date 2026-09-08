"""Sovereign-cloud-aware Microsoft Graph delta ingestion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import requests

from ingestion.models import ChangeEvent, ConnectorBatch, SourceDocument


class GraphClient:
    """Small requests-based Graph client with externally supplied credentials."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        graph_base_url: str = "https://graph.microsoft.com/v1.0",
        token_url: str | None = None,
        session: requests.Session | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.graph_base_url = graph_base_url.rstrip("/")
        self.token_url = token_url or (
            f"https://login.microsoftonline.com/{quote(tenant_id, safe='')}/oauth2/v2.0/token"
        )
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self._access_token: str | None = None

    def _token(self) -> str:
        if self._access_token is None:
            response = self.session.post(
                self.token_url,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                    "scope": f"{self.graph_base_url.split('/v1.0', 1)[0]}/.default",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            self._access_token = payload["access_token"]
        return self._access_token

    def get_json(self, url_or_path: str) -> dict[str, Any]:
        response = self._get(url_or_path)
        return response.json()

    def get_bytes(self, url_or_path: str) -> bytes:
        return self._get(url_or_path).content

    def _get(self, url_or_path: str) -> requests.Response:
        url = (
            url_or_path
            if url_or_path.startswith(("https://", "http://"))
            else f"{self.graph_base_url}/{url_or_path.lstrip('/')}"
        )
        response = self.session.get(
            url,
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 401:
            self._access_token = None
            response = self.session.get(
                url,
                headers={"Authorization": f"Bearer {self._token()}"},
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        return response


def _identity_principals(item: Mapping[str, Any]) -> tuple[str, ...]:
    principals: set[str] = set()
    for permission in item.get("permissions", ()):
        identities = permission.get("grantedToIdentitiesV2") or ()
        granted = permission.get("grantedToV2")
        if granted:
            identities = (*identities, granted)
        for identity_set in identities:
            for kind in ("user", "group", "application", "siteUser", "siteGroup"):
                identity = identity_set.get(kind) or {}
                value = identity.get("id") or identity.get("email")
                if value:
                    principals.add(f"{kind}:{value}")
        scope = (permission.get("link") or {}).get("scope")
        if scope in {"anonymous", "organization"}:
            principals.add(f"link:{scope}")
    return tuple(sorted(principals))


class GraphDeltaConnector:
    """Read all pages of a drive delta feed and emit upserts/tombstones."""

    def __init__(
        self,
        client: GraphClient,
        *,
        drive_id: str,
        corpus: str,
        security_label: str = "OFFICIAL",
        default_acl_principals: tuple[str, ...] = (),
        fetch_permissions: bool = True,
    ) -> None:
        self.client = client
        self.drive_id = drive_id
        self.corpus = corpus
        self.security_label = security_label
        self.default_acl_principals = default_acl_principals
        self.fetch_permissions = fetch_permissions

    def changes(self, cursor: str | None = None) -> ConnectorBatch:
        url = cursor or f"/drives/{quote(self.drive_id, safe='')}/root/delta"
        events: list[ChangeEvent] = []
        final_cursor = ""
        while url:
            page = self.client.get_json(url)
            for item in page.get("value", ()):
                event = self._event(item)
                if event is not None:
                    events.append(event)
            next_url = page.get("@odata.nextLink")
            delta_url = page.get("@odata.deltaLink")
            if delta_url:
                final_cursor = str(delta_url)
            url = str(next_url) if next_url else ""
        if not final_cursor:
            raise ValueError("Graph delta response did not contain @odata.deltaLink")
        return ConnectorBatch(tuple(events), final_cursor)

    def _event(self, item: Mapping[str, Any]) -> ChangeEvent | None:
        if "folder" in item:
            return None
        document_id = str(item["id"])
        provenance = {
            "connector": "microsoft-graph",
            "drive_id": self.drive_id,
            "item_id": document_id,
            "name": item.get("name", ""),
            "web_url": item.get("webUrl", ""),
            "parent_path": (item.get("parentReference") or {}).get("path", ""),
        }
        version = str(
            item.get("eTag") or item.get("cTag") or item.get("lastModifiedDateTime") or ""
        )
        if "deleted" in item:
            return ChangeEvent.delete(
                self.corpus,
                document_id,
                version=version,
                provenance=provenance,
            )
        if "permissions" not in item and self.fetch_permissions:
            permission_page = self.client.get_json(
                f"/drives/{quote(self.drive_id, safe='')}/items/"
                f"{quote(document_id, safe='')}/permissions"
            )
            item = {**item, "permissions": permission_page.get("value", ())}
        content = self.client.get_bytes(
            f"/drives/{quote(self.drive_id, safe='')}/items/{quote(document_id, safe='')}/content"
        )
        encoding = str((item.get("file") or {}).get("encoding") or "utf-8")
        text = content.decode(encoding)
        label = str((item.get("sensitivityLabel") or {}).get("displayName") or self.security_label)
        acl = _identity_principals(item) or self.default_acl_principals
        return ChangeEvent.upsert(
            SourceDocument(
                corpus=self.corpus,
                document_id=document_id,
                version=version or str(len(content)),
                text=text,
                security_label=label,
                acl_principals=acl,
                provenance=provenance,
            )
        )

    def poll(self, cursor: str | None = None) -> ConnectorBatch:
        return self.changes(cursor)
