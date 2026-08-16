"""Customer database: a real SQLite lookup the agent depends on."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "customers.db"


def ensure_db() -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, "
        "plan TEXT, price TEXT)"
    )
    connection.execute(
        "INSERT OR REPLACE INTO users VALUES (1, 'Priya', 'premium', '$19.99'), "
        "(2, 'Marco', 'basic', '$9.99')"
    )
    connection.commit()
    connection.close()


def fetch_user(user_id: int) -> dict:
    ensure_db()
    connection = sqlite3.connect(DB_PATH)
    row = connection.execute(
        "SELECT name, plan, price FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    connection.close()
    if row is None:
        return {"name": "unknown", "plan": "none", "price": "n/a"}
    return {"name": row[0], "plan": row[1], "price": row[2]}
