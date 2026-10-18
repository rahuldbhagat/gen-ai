"""Load demo_notes.txt, split it into chunks, embed them and store them in Postgres.

Run it once, and again whenever the notes change:
    uv run ingest.py            # add or update chunks; safe to run many times
    uv run ingest.py --rebuild  # empty the table first; use after deleting or editing text 
"""

import argparse
import logging
import sys
import uuid

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAIError
from sqlalchemy.exc import SQLAlchemyError

from vector_store import (
    SCHEMA_NAME,
    TABLE_NAME,
    close_engine,
    create_embeddings,
    create_engine,
    ensure_table,
    open_store,
)

SOURCE_FILE = "demo_notes.txt"

# Fixed namespace for chunk IDs. Changing it changes every ID, so the next
# ingest would add a second copy of everything.
CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-demo/demo-notes-chunks")

log = logging.getLogger("ingest")


def chunk_id(chunk: Document) -> str:
    """Same source + same text -> same ID, so re-running updates instead of duplicating."""
    key = f"{chunk.metadata.get('source', '')}\n{chunk.page_content}"
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, key))


def load_chunks(path: str) -> list[Document]:
    # Explicit UTF-8: on Windows the default encoding is not UTF-8.
    documents = TextLoader(path, encoding="utf-8").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(documents)


def unique_by_id(chunks: list[Document]) -> tuple[list[Document], list[str]]:
    # Identical chunks would get the same ID; embed each one only once.
    by_id = {chunk_id(chunk): chunk for chunk in chunks}
    return list(by_id.values()), list(by_id.keys())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop and recreate the table before loading (removes all stored chunks)",
    )
    args = parser.parse_args()
    load_dotenv()

    try:
        chunks, ids = unique_by_id(load_chunks(SOURCE_FILE))
    except RuntimeError as exc:  # TextLoader wraps file and decode errors in RuntimeError
        log.error("Could not read %s: %s", SOURCE_FILE, exc.__cause__ or exc)
        return 1
    if not chunks:
        log.error("%s produced no chunks. Is the file empty?", SOURCE_FILE)
        return 1

    try:
        embeddings = create_embeddings()
        engine = create_engine()
    except RuntimeError as exc:  # missing environment variable
        log.error("%s", exc)
        return 1

    try:
        created = ensure_table(engine, rebuild=args.rebuild)
        log.info(
            "Table %s.%s %s",
            SCHEMA_NAME,
            TABLE_NAME,
            "created" if created else "already exists",
        )
        store = open_store(engine, embeddings)
        store.add_documents(chunks, ids=ids)
        log.info("Stored %d chunks from %s", len(chunks), SOURCE_FILE)
        return 0
    except (SQLAlchemyError, OSError) as exc:
        log.error(
            "Database error: %s\nIs the container running? Check with: docker compose ps",
            exc,
        )
        return 1
    except OpenAIError as exc:
        # Rows are saved one at a time, so some may already be stored.
        # Re-running is safe: stored chunks are updated, not duplicated.
        log.error("OpenAI embedding call failed: %s\nFix the problem and run again.", exc)
        return 1
    finally:
        close_engine(engine)


if __name__ == "__main__":
    sys.exit(main())
