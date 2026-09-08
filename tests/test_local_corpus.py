import sqlite3

from app.local_retrieval import FederalCorpusRetrieval
from knowledge.ingest_dita import parse_topic


def test_main_clause_not_lost_when_alternate_section_exists(tmp_path):
    path = tmp_path / "clause.dita"
    path.write_text(
        '<dita><concept id="clause"><title>Clause</title><conbody>'
        "<p>Main payment requirement.</p><ol><li>Thirty days.</li></ol>"
        '<section id="alt"><title>Alternate I</title><p>Alternate rule.</p>'
        "</section></conbody></concept></dita>"
    )
    records = parse_topic(path, authority="FAR", document_id="test", document_title="Test")
    assert len(records) == 2
    assert "Main payment requirement" in records[0].text
    assert "Thirty days" in records[0].text
    assert "Alternate rule" not in records[0].text
    assert "Alternate rule" in records[1].text


def test_official_dita_concept_wrapper_is_extracted(tmp_path):
    source = tmp_path / "252.999-1.dita"
    source.write_text(
        '<dita><concept id="clause"><title>Synthetic clause</title>'
        "<conbody><p>As prescribed in the synthetic prescription.</p></conbody>"
        "</concept></dita>",
    )
    records = parse_topic(
        source,
        authority="DFARS",
        document_id="test",
        document_title="Synthetic",
    )
    assert len(records) == 1
    assert "synthetic prescription" in records[0].text


def test_retrieval_keeps_source_version_and_readonly_database(tmp_path):
    path = tmp_path / "corpus.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5("
            "evidence_id UNINDEXED, heading, text, authority UNINDEXED, "
            "url UNINDEXED, version UNINDEXED, document_id UNINDEXED, "
            "locator UNINDEXED, content_sha256 UNINDEXED)"
        )
        connection.execute(
            "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "known-id",
                "Prompt payment",
                "Payment is described in this synthetic passage.",
                "FAR",
                "https://example.test/source",
                "pinned-commit",
                "doc-1",
                "loc-1",
                "hash",
            ),
        )
    retrieval = FederalCorpusRetrieval(str(path))
    results = retrieval.retrieve(['payment " OR *'])
    assert len(results) == 1
    assert results[0].version == "pinned-commit"
    assert results[0].evidence_id == "known-id"
    assert retrieval.retrieve(["zxqvnotindata"]) == []
