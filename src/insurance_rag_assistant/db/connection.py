from psycopg import Connection
from psycopg.rows import TupleRow
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from insurance_rag_assistant.config import settings


def _configure(conn: Connection[TupleRow]) -> None:
    register_vector(conn)


def create_pool(
    database_url: str | None = None,
    min_size: int = 1,
    max_size: int = 5,
) -> ConnectionPool[Connection[TupleRow]]:
    """Create a pool whose connections understand the pgvector type.

    The `vector` extension must already exist (run the migrations first).
    """
    return ConnectionPool(
        conninfo=database_url or settings.database_url,
        min_size=min_size,
        max_size=max_size,
        configure=_configure,
        open=True,
    )  # type: ignore[return-value]
