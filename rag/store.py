"""pgvector storage. Exact scan (no ANN index), so a filtered query always returns k rows."""
import os
from datetime import date

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .ingest import Chunk

DIM = 384  # bge-small-en-v1.5
DEFAULT_URL = "postgresql://postgres:postgres@localhost:5432/rag"

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
DROP TABLE IF EXISTS chunks, doc_versions, index_meta;
CREATE TABLE index_meta (
    embedder       text NOT NULL,               -- query vectors must come from the same model
    corpus         text NOT NULL                -- fingerprint of the files and chunking settings
);
CREATE TABLE doc_versions (
    id             serial PRIMARY KEY,
    document_id    text NOT NULL,
    title          text NOT NULL,
    version        int  NOT NULL,
    effective_from date NOT NULL,
    effective_to   date,                        -- NULL = still in effect
    status         text NOT NULL CHECK (status IN ('current', 'superseded')),
    uploaded_at    date NOT NULL,
    path           text NOT NULL,
    UNIQUE (document_id, version),
    CHECK (effective_to IS NULL OR effective_to > effective_from)
);
CREATE TABLE chunks (
    id             serial PRIMARY KEY,
    doc_version_id int  NOT NULL REFERENCES doc_versions (id),
    section        text NOT NULL,
    ord            int  NOT NULL,
    content        text NOT NULL,
    embedding      vector({DIM}) NOT NULL,
    tsv            tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON chunks (doc_version_id);
"""

# Historical filtering uses the date range only, never status: a version uploaded early is
# marked superseded-on-upload but stays in effect until its successor's effective date.
FILTERS = {
    "all": "TRUE",
    "current": "v.status = 'current'",
    "aware": "v.effective_from <= %(as_of)s AND (v.effective_to IS NULL OR v.effective_to > %(as_of)s)",
}

COLUMNS = """v.document_id, v.version, v.title, v.effective_from, v.effective_to, c.section, c.content"""


def connect(url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(url or os.environ.get("DATABASE_URL", DEFAULT_URL), autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def reset(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def set_meta(conn: psycopg.Connection, embedder: str, corpus: str) -> None:
    conn.execute("INSERT INTO index_meta VALUES (%s, %s)", (embedder, corpus))


def meta(conn: psycopg.Connection) -> dict | None:
    """What the index was built with, or None if there is no index."""
    if conn.execute("SELECT to_regclass('index_meta')").fetchone()[0] is None:
        return None
    rows = _rows(conn.execute("SELECT embedder, corpus FROM index_meta"))
    return rows[0] if rows else None


def add_version(conn: psycopg.Connection, row: dict) -> int:
    """Insert a document version. Never deletes: the open previous version is closed instead.

    Versions must arrive in order: inserting v2 after v3 would leave two versions open.
    """
    with conn.transaction():
        latest = conn.execute("SELECT max(version) FROM doc_versions WHERE document_id = %(document_id)s", row).fetchone()[0]
        if latest is not None and latest >= row["version"]:
            raise ValueError(f"{row['document_id']} v{row['version']} arrived after v{latest}")
        conn.execute(
            """UPDATE doc_versions SET effective_to = %(effective_from)s, status = 'superseded'
               WHERE document_id = %(document_id)s AND effective_to IS NULL AND version < %(version)s""", row)
        return conn.execute(
            """INSERT INTO doc_versions (document_id, title, version, effective_from, status, uploaded_at, path)
               VALUES (%(document_id)s, %(title)s, %(version)s, %(effective_from)s, 'current', %(uploaded_at)s, %(path)s)
               RETURNING id""", row).fetchone()[0]


def add_chunks(conn: psycopg.Connection, version_id: int, chunks: list[Chunk], vectors: np.ndarray) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_version_id, section, ord, content, embedding) VALUES (%s, %s, %s, %s, %s)",
            [(version_id, c.section, c.ord, c.content, v) for c, v in zip(chunks, vectors)])


def search(conn: psycopg.Connection, vector: np.ndarray, k: int, mode: str, as_of: date) -> list[dict]:
    """Top-k chunks by cosine similarity among the versions `mode` allows."""
    return _rows(conn.execute(
        f"""SELECT {COLUMNS}, 1 - (c.embedding <=> %(v)s) AS score
            FROM chunks c JOIN doc_versions v ON v.id = c.doc_version_id
            WHERE {FILTERS[mode]} ORDER BY c.embedding <=> %(v)s, c.id LIMIT %(k)s""",
        {"v": vector, "k": k, "as_of": as_of}))


def lexical_search(conn: psycopg.Connection, question: str, k: int, mode: str, as_of: date) -> list[dict]:
    """Postgres full-text baseline: any question term may match (OR), ranked by ts_rank.

    plainto_tsquery parses any text (URLs, emails, punctuation); its AND query is turned into an OR
    of the same already-stemmed lexemes.
    """
    return _rows(conn.execute(
        f"""WITH q AS (SELECT replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS query)
            SELECT {COLUMNS}, ts_rank(c.tsv, q.query) AS score
            FROM chunks c JOIN doc_versions v ON v.id = c.doc_version_id, q
            WHERE {FILTERS[mode]} AND c.tsv @@ q.query ORDER BY score DESC, c.id LIMIT %(k)s""",
        {"q": question, "k": k, "as_of": as_of}))


def chunks_of_current_versions(conn: psycopg.Connection) -> list[dict]:
    """Every chunk of a document version that is in effect now, with its embedding as a numpy array."""
    rows = _rows(conn.execute(
        f"""SELECT {COLUMNS}, c.embedding FROM chunks c JOIN doc_versions v ON v.id = c.doc_version_id
            WHERE v.effective_to IS NULL ORDER BY c.id"""))
    return [{**r, "embedding": r["embedding"].to_numpy()} for r in rows]


def versions(conn: psycopg.Connection) -> list[dict]:
    return _rows(conn.execute("SELECT document_id, version, effective_from, effective_to, status FROM doc_versions"))


def _rows(cursor: psycopg.Cursor) -> list[dict]:
    names = [d.name for d in cursor.description]
    return [dict(zip(names, r)) for r in cursor]
