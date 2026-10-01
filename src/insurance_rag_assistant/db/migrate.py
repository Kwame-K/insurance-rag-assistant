from pathlib import Path

import psycopg

from insurance_rag_assistant.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def apply_migrations(database_url: str | None = None) -> list[str]:
    """Apply pending SQL migrations in filename order; return applied names."""
    url = database_url or settings.database_url
    applied: list[str] = []

    with psycopg.connect(url, autocommit=False) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)

        conn.commit()

    return applied


def main() -> None:
    applied = apply_migrations()
    print(f"Applied migrations: {applied or 'none (up to date)'}")


if __name__ == "__main__":
    main()
