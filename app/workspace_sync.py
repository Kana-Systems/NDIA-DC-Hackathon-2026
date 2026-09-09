"""Periodic reconciliation with persisted per-connection leases."""

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import HTTPException

from app.models import PrincipalContext

logger = logging.getLogger(__name__)


def elapsed(value):
    return (
        (datetime.now(UTC) - datetime.fromisoformat(value)).total_seconds()
        if value
        else float("inf")
    )


def claim(db, principal, connection):
    if connection.get("status") == "syncing" and elapsed(connection.get("sync_started_at")) < 3600:
        raise HTTPException(409, "A sync is already running")
    updated = dict(
        connection,
        status="syncing",
        sync_started_at=datetime.now(UTC).isoformat(),
        sync_lease=uuid4().hex,
    )
    return db.save(principal, "connection", updated, connection["id"], connection["revision"])


def reconcile(db, settings):
    from app.workspace import sync_connection

    for connection in db.connections_for_worker():
        if connection["provider"] != "sharepoint":
            continue
        if elapsed(connection.get("last_sync")) < settings.workspace_sync_seconds:
            continue
        principal = PrincipalContext(
            subject=connection["owner"], security_domain=connection["security_domain"]
        )
        try:
            lease = claim(db, principal, connection)
        except HTTPException as error:
            if error.status_code != 409:
                raise
            continue
        sync_connection(db, principal, connection["id"], settings, lease["sync_lease"])


async def worker(db, settings, stop):
    while not stop.is_set():
        try:
            await asyncio.to_thread(reconcile, db, settings)
        except Exception:
            logger.warning("Workspace reconciliation failed; retrying on the next poll")
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=30)
