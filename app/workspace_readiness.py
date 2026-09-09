"""One source of truth for review readiness, decisions, and portable exports.

Readiness is computed from current records, never persisted as an authorization.
The evaluator is request-scoped so source metadata is read once per response.
"""

from fastapi import HTTPException

from app.workspace_store import now


class ReadinessEvaluator:
    def __init__(self, db, principal):
        self.db = db
        self.principal = principal
        self.documents = {}

    def document(self, record_id):
        if record_id not in self.documents:
            try:
                doc = self.db.get(self.principal, record_id, hydrate=False)
                self.documents[record_id] = doc if doc["kind"] == "document" else None
            except HTTPException as error:
                if error.status_code != 404:
                    raise
                self.documents[record_id] = None
        return self.documents[record_id]

    def evaluate(self, item):
        blockers = []

        def block(code, message):
            if not any(b["code"] == code for b in blockers):
                blockers.append({"code": code, "message": message})

        def check_source(record_id, version):
            doc = self.document(record_id)
            if doc is None or doc.get("deleted"):
                block(
                    "source_unavailable",
                    "A supporting source is unavailable. Create a fresh draft.",
                )
            elif not self.db.source_available(self.principal, doc):
                block("source_not_ready", "Sync or index the supporting source successfully first.")
            elif not version or doc["version"] != version:
                block(
                    "source_changed", "A source changed. Run a new review or create a fresh draft."
                )
            return doc

        kind = item["kind"]
        if kind == "review":
            doc = check_source(item["document_id"], item["document_version"])
            if doc and doc.get("metadata") != item.get("metadata_snapshot"):
                block("metadata_changed", "Acquisition details changed. Run a new review.")
            evidence = item["analysis"]["report"].get("evidence", [])
        elif kind == "structured-record":
            content = item["content"]
            evidence = content["evidence"]
            ids = {e["evidence_id"] for e in evidence}
            if not content["statements"] or any(
                s["grounding_status"] != "verified"
                or not s["citation_ids"]
                or not set(s["citation_ids"]).issubset(ids)
                for s in content["statements"]
            ):
                block(
                    "unresolved_citations", "Resolve uncited statements before approval or export."
                )
            if item.get("document_id"):
                # Older context-bound records lack a snapshot: require regeneration.
                check_source(item["document_id"], item.get("document_version"))
        elif kind == "entity":
            evidence = []
            for link in item["links"]:
                check_source(link["document_id"], link["version"])
        else:
            block("unsupported_record", "This record does not support a review decision.")
            evidence = []
        for link in evidence:
            if link["evidence_id"].startswith("workspace:"):
                check_source(link["document_id"], link["version"])
        return {
            "can_approve": not blockers,
            "can_export": not blockers
            and item.get("decision") == "approved"
            and kind in {"review", "structured-record"},
            "blockers": blockers,
            "checked_at": now(),
        }

    def present(self, item):
        return {**item, "readiness": self.evaluate(item)}

    def require_current(self, item):
        status = self.evaluate(item)
        if status["blockers"]:
            raise HTTPException(409, status["blockers"][0]["message"])
