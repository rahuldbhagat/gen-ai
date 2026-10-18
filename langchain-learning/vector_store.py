"""Shared Postgres vector-store setup used by ingest.py and app.py.

Everything both scripts must agree on lives here: where the table is,
how big each vector is, and which embedding model makes the vectors.
"""

import asyncio
import os

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from sqlalchemy.engine import URL
from sqlalchemy.exc import ProgrammingError

SCHEMA_NAME = "dev_db"
TABLE_NAME = "demo_notes_chunks"

# The table column is created with a fixed vector size. If you change the
# embedding model or its dimensions, you must create a new table (ingest.py --rebuild).
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

# SQLSTATE for "relation already exists".
_DUPLICATE_TABLE_SQLSTATE = "42P07"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_connection_url() -> URL:
    # URL.create escapes special characters in the password for us.
    # asyncpg (not psycopg) on purpose: psycopg's async mode does not work with
    # Windows' default asyncio event loop, and PGEngine is async underneath.
    return URL.create(
        drivername="postgresql+asyncpg",
        username=require_env("POSTGRES_USER"),
        password=require_env("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=require_env("POSTGRES_DB"),
    )


def create_engine() -> PGEngine:
    return PGEngine.from_connection_string(url=build_connection_url())


def close_engine(engine: PGEngine) -> None:
    asyncio.run(engine.close())


def create_embeddings() -> OpenAIEmbeddings:
    # Passing dimensions keeps the vectors tied to VECTOR_SIZE.
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        dimensions=VECTOR_SIZE,
        api_key=require_env("OPENAI_API_KEY"),
    )


def ensure_table(engine: PGEngine, *, rebuild: bool = False) -> bool:
    """Create the vector table if needed. Returns True if it was created.

    rebuild=True drops the table first, which removes all stored chunks.
    """
    try:
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            schema_name=SCHEMA_NAME,
            vector_size=VECTOR_SIZE,
            overwrite_existing=rebuild,
        )
        return True
    except ProgrammingError as exc:
        if _sqlstate(exc) == _DUPLICATE_TABLE_SQLSTATE:
            return False
        raise


def open_store(engine: PGEngine, embeddings: OpenAIEmbeddings) -> PGVectorStore:
    try:
        return PGVectorStore.create_sync(
            engine=engine,
            table_name=TABLE_NAME,
            schema_name=SCHEMA_NAME,
            embedding_service=embeddings,
        )
    except ValueError as exc:
        # The library raises ValueError when the table or its columns are missing.
        raise RuntimeError(
            f"Vector table {SCHEMA_NAME}.{TABLE_NAME} is missing or has the wrong "
            f"columns ({exc}). Run `uv run ingest.py` first."
        ) from exc


def _sqlstate(exc: ProgrammingError) -> str | None:
    original = exc.orig
    return getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
