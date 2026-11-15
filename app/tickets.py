import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import AiIntent, IncomingMessage

DB_PATH=Path(os.getenv("DB_PATH", "tickets.db"))

def _get_connection() -> sqlite3.connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """ticket table creation"""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL DEFAULT 'open',
                priority TEXT NOT NULL,
                category TEXT NOT NULL,
                guest_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                original_message TEXT NOT NULL,
                internal_summary TEXT NOT NULL,
                suggested_reply TEXT NOT NULL,
                reservation_json TEXT,
                provider_used TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

def create_ticket(
    message: IncomingMessage,
    decision: AiIntent,
    provider_used: str,
) -> int:
    init_db()

    category_val = (
        decision.category.value
        if hasattr(decision.category, "value")
        else str(decision.category)
    )
    reservation_json = (
        decision.reservation_data.model_dump_json()
        if decision.reservation_data
        else None
    )
    created_at = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO tickets (
                status, priority, category, guest_name, phone,
                original_message, internal_summary, suggested_reply,
                reservation_json, provider_used, created_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "open",
                decision.priority,
                category_val,
                message.guest_name,
                message.phone,
                message.text,
                decision.internal_summary,
                decision.guest_response,
                reservation_json,
                provider_used,
                created_at,
            ),
        )
        return int(cursor.lastrowid)

def list_tickets(status: str | None = None) -> list[dict]:
    """return sorted tickets by newest"""
    init_db()
    with _get_connection() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status = ? ORDER BY id DESC",
                (status,),

            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tickets ORDER BY id DESC"
            ).fetchall()

    results = []
    for row in rows:
        item = dict(row)
        item["reservation_data"] = (
            json.loads(item["reservation_json"]) if item["reservation_json"] else None
        )
        del item["reservation_json"]
        results.append(item)
    return results

def resolve_ticket(ticket_id: int) -> bool:
    """mark tickets as solved by id, return true if successful"""
    init_db()
    with _get_connection() as conn:
        cursor = conn.execute(
            "UPDATE tickets SET status = 'resolved' WHERE id = ?",
            (ticket_id,),
        )
        return cursor.rowcount > 0

