"""Against a real pgvector database. Skipped unless TEST_DATABASE_URL is set (CI starts one).

The tests rebuild the tables, so point TEST_DATABASE_URL at a database you don't serve from.
"""
import os
from datetime import date

import pytest

from rag import store
from rag.ingest import manifest
from rag.models import FakeEmbedder
from rag.pipeline import ensure_index, index, retrieve

pytestmark = pytest.mark.skipif("TEST_DATABASE_URL" not in os.environ, reason="needs a pgvector database")


@pytest.fixture(scope="module")
def conn():
    connection = store.connect(os.environ["TEST_DATABASE_URL"])
    index(connection, FakeEmbedder())
    yield connection
    connection.close()


def test_index_records_its_embedder_and_is_not_replaced_by_another_one(conn):
    assert store.meta(conn)["embedder"] == "fake-hashing"
    assert ensure_index(conn, FakeEmbedder()) is False

    class Other(FakeEmbedder):
        name = "other-model"
    with pytest.raises(SystemExit, match="built with fake-hashing"):
        ensure_index(conn, Other())


def test_versions_must_arrive_in_order(conn):
    row = {**next(m for m in manifest() if m["document_id"] == "employee-handbook" and m["version"] == 2),
           "uploaded_at": "2026-09-01"}
    with pytest.raises(ValueError, match="arrived after v3"):
        store.add_version(conn, row)


def test_ingest_never_deletes_and_closes_previous_versions(conn):
    stored = {(r["document_id"], r["version"]): r for r in store.versions(conn)}
    for m in manifest():
        r = stored[(m["document_id"], m["version"])]
        assert r["effective_from"].isoformat() == m["effective_from"]
        assert (r["effective_to"] and r["effective_to"].isoformat()) == m["effective_to"]
        assert r["status"] == m["status"]


def versions_for(rows, document_id):
    return {r["version"] for r in rows if r["document_id"] == document_id}


def test_modes_filter_versions(conn):
    question = "How many unused vacation days can be carried over?"
    embedder = FakeEmbedder()
    # v3 was uploaded on 10 Feb 2026 but takes effect on 1 March: on 20 Feb, v2 is the policy.
    aware = retrieve(conn, embedder, question, "aware", date(2026, 2, 20), k=20)
    assert versions_for(aware, "employee-handbook") == {2}
    current = retrieve(conn, embedder, question, "current", date(2026, 2, 20), k=20)
    assert versions_for(current, "employee-handbook") == {3}
    everything = retrieve(conn, embedder, question, "all", date(2026, 2, 20), k=40)
    assert versions_for(everything, "employee-handbook") == {1, 2, 3}
    assert retrieve(conn, embedder, question, "aware", date(2023, 6, 1), k=5) == []  # nothing was in effect yet


def test_lexical_search_uses_the_as_of_versions(conn):
    rows = retrieve(conn, FakeEmbedder(), "vacation days carried over", "lexical", date(2024, 6, 1), k=10)
    assert rows and versions_for(rows, "employee-handbook") == {1}


@pytest.mark.parametrize("question", ["email it-help@brackwater.example about https://intranet/x?a=b", "the of and",
                                      "O'Brien & co | !(x) <-> y:*", "carry-over"])
def test_lexical_search_accepts_any_text(conn, question):
    retrieve(conn, FakeEmbedder(), question, "lexical", date(2026, 10, 1), k=5)


def test_contradictions_use_versions_effective_on_the_requested_date(conn):
    before = store.chunks_of_current_versions(conn, date(2026, 2, 20))
    after = store.chunks_of_current_versions(conn, date(2026, 3, 1))
    assert versions_for(before, "employee-handbook") == {2}
    assert versions_for(after, "employee-handbook") == {3}
    assert store.chunks_of_current_versions(conn, date(2023, 1, 1)) == []
