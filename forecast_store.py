import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATABASE = Path(__file__).resolve().with_name("forecast_accounts.sqlite3")
PASSWORD_ROUNDS = 600_000
MAX_PASSWORD_LENGTH = 256
HISTORY_PAGE_SIZE = 20


class AccountExists(ValueError):
    pass


def password_hash(password: str, salt: bytes, rounds: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)


class ForecastStore:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as db:
            db.executescript("""
                BEGIN;
                CREATE TABLE IF NOT EXISTS forecast_users (
                    id INTEGER PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL,
                    password_rounds INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES forecast_users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model TEXT NOT NULL,
                    lookback INTEGER NOT NULL,
                    horizon INTEGER NOT NULL,
                    points TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS forecast_runs_user_id_idx
                    ON forecast_runs (user_id, id DESC);
                COMMIT;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    def register(self, email: str, password: str) -> tuple[int, str]:
        email = email.strip().lower()
        if len(email) > 254 or not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            raise ValueError("Enter a valid email address.")
        if not 12 <= len(password) <= MAX_PASSWORD_LENGTH:
            raise ValueError(f"Use a password between 12 and {MAX_PASSWORD_LENGTH} characters.")

        salt = secrets.token_bytes(16)
        digest = password_hash(password, salt, PASSWORD_ROUNDS)
        with self.connect() as db:
            cursor = db.execute(
                """INSERT INTO forecast_users (email, password_salt, password_hash, password_rounds)
                   VALUES (?, ?, ?, ?) ON CONFLICT(email) DO NOTHING""",
                (email, salt, digest, PASSWORD_ROUNDS),
            )
            if cursor.rowcount == 0:
                raise AccountExists("An account already exists for this email. Sign in instead.")
            return cursor.lastrowid, email

    def authenticate(self, email: str, password: str) -> tuple[int, str] | None:
        email = email.strip().lower()
        if len(email) > 254 or not 1 <= len(password) <= MAX_PASSWORD_LENGTH:
            return None

        with self.connect() as db:
            user = db.execute(
                "SELECT * FROM forecast_users WHERE email = ?", (email,)
            ).fetchone()

        if user is None:
            # Do comparable work for an unknown email and an incorrect password.
            password_hash(password, bytes(16), PASSWORD_ROUNDS)
            return None

        digest = password_hash(password, user["password_salt"], user["password_rounds"])
        if hmac.compare_digest(digest, user["password_hash"]):
            return user["id"], user["email"]
        return None

    def save_forecast(
        self,
        user_id: int,
        symbol: str,
        model: str,
        lookback: int,
        points: list[tuple[str, float]],
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        serialized_points = json.dumps(points, allow_nan=False)
        with self.connect() as db:
            db.execute(
                """INSERT INTO forecast_runs
                   (user_id, created_at, symbol, model, lookback, horizon, points)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, created_at, symbol, model, lookback, len(points), serialized_points),
            )

    def history(self, user_id: int, page: int) -> list[sqlite3.Row]:
        with self.connect() as db:
            return db.execute(
                """SELECT created_at, symbol, model, lookback, horizon, points
                   FROM forecast_runs WHERE user_id = ? ORDER BY id DESC LIMIT ? OFFSET ?""",
                (user_id, HISTORY_PAGE_SIZE + 1, page * HISTORY_PAGE_SIZE),
            ).fetchall()
