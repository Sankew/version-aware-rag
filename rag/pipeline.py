"""Index the corpus and retrieve in one of four modes.

    uv run --group models python -m rag.pipeline index
"""
from datetime import date

from . import store
from .ingest import fingerprint, load_corpus

MODES = ("all", "current", "aware", "lexical")  # lexical = full-text baseline over the as-of versions


def index(conn, embedder) -> int:
    """Rebuild the tables from the manifest, ingesting files in upload order."""
    store.reset(conn)
    total = 0
    for row, chunks in load_corpus(embedder.count):
        version_id = store.add_version(conn, row)
        store.add_chunks(conn, version_id, chunks, embedder.embed([c.content for c in chunks]))
        total += len(chunks)
    store.set_meta(conn, embedder.name, fingerprint())
    return total


def ensure_index(conn, embedder, reindex: bool = False) -> bool:
    """Build the index if it's missing or the corpus changed. True if rebuilt.

    An index built with a different embedder is someone else's (the API's, say): it is replaced
    only with reindex=True.
    """
    built = store.meta(conn)
    if built == {"embedder": embedder.name, "corpus": fingerprint()}:
        return False
    if built and built["embedder"] != embedder.name and not reindex:
        raise SystemExit(f"this database's index was built with {built['embedder']}, not {embedder.name}. "
                         "Use another database (DATABASE_URL) or pass --reindex to replace it.")
    index(conn, embedder)
    return True


def retrieve(conn, embedder, question: str, mode: str, as_of: date, k: int = 5) -> list[dict]:
    if mode == "lexical":
        return store.lexical_search(conn, question, k, "aware", as_of)
    return store.search(conn, embedder.embed([question], query=True)[0], k, mode, as_of)


if __name__ == "__main__":
    import sys
    from .models import Embedder
    if sys.argv[1:] != ["index"]:
        raise SystemExit(__doc__)
    print(index(store.connect(), Embedder()), "chunks indexed")
