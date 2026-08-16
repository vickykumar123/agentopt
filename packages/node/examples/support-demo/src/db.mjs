/** Customer database: a real SQLite lookup the agent depends on. */

import { DatabaseSync } from "node:sqlite";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const DB_PATH = join(dirname(fileURLToPath(import.meta.url)), "..", "customers.db");

function ensureDb() {
  const db = new DatabaseSync(DB_PATH);
  db.exec(
    "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, plan TEXT, price TEXT)",
  );
  db.exec(
    "INSERT OR REPLACE INTO users VALUES (1, 'Priya', 'premium', '$19.99'), " +
      "(2, 'Marco', 'basic', '$9.99')",
  );
  db.close();
}

export function fetchUser(userId) {
  ensureDb();
  const db = new DatabaseSync(DB_PATH);
  const row = db.prepare("SELECT name, plan, price FROM users WHERE id = ?").get(userId);
  db.close();
  if (!row) return { name: "unknown", plan: "none", price: "$0" };
  return { name: row.name, plan: row.plan, price: row.price };
}
