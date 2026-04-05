#!/usr/bin/env python3
# ============================================================
# Script: Export Conversations to SQLite
#
# Extracts conversations from Claude Code and Cursor into a
# portable SQLite database. No LLM calls, no API keys.
#
# Two tables:
#   conversations — one row per session
#   messages      — one row per message, ordered
#
# Usage:
#   python export_conversations.py                            # all sources
#   python export_conversations.py -o memories.db
#   python export_conversations.py --source cursor --name "interview"
#   python export_conversations.py --source claude --project vibma
# ============================================================

import argparse
import json
import os
import re
import sqlite3
import sys
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("ombre_brain.export")

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"
CURSOR_DB = Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb"

SCHEMA_VERSION = 2

NOISE_PATTERNS = [
    re.compile(r"^<(system-reminder|antml:|functions).*", re.MULTILINE),
    re.compile(r"^\s*\{\"type\":\s*\"tool", re.MULTILINE),
    re.compile(r"^Co-Authored-By:.*", re.MULTILINE),
]
MSG_CONTENT_CAP = 800


# ── Schema ───────────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id            TEXT PRIMARY KEY,
            source        TEXT NOT NULL,           -- 'claude' or 'cursor'
            name          TEXT DEFAULT '',
            project       TEXT DEFAULT '',
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            status        TEXT DEFAULT 'pending',  -- pending, processed, failed
            processed_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            position        INTEGER NOT NULL,       -- 0-based order in conversation
            role            TEXT NOT NULL,           -- 'user' or 'assistant'
            content         TEXT NOT NULL,
            timestamp       TEXT,                    -- ISO timestamp of this message
            FOREIGN KEY (conversation_id) REFERENCES conversations(id),
            UNIQUE (conversation_id, position)
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, position);
    """)

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),)
    )
    conn.commit()
    return conn


# ── Noise filter ─────────────────────────────────────────────

def is_noise(text: str) -> bool:
    for pattern in NOISE_PATTERNS:
        if pattern.search(text):
            return True
    lines = text.split("\n")
    if len(lines) > 5:
        code_lines = sum(
            1 for l in lines
            if l.startswith(("  ", "\t", "{", "}"))
            or l.strip().startswith(("//", "#", "/*", "*"))
        )
        if code_lines / len(lines) > 0.6:
            return True
    return False


# ── Claude ───────────────────────────────────────────────────

def export_claude(conn: sqlite3.Connection, project_filter: str = None, min_messages: int = 4) -> int:
    if not PROJECTS_DIR.exists():
        logger.warning(f"Claude projects dir not found: {PROJECTS_DIR}")
        return 0

    exported = 0
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_filter and project_filter not in project_dir.name:
            continue

        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            session_id = jsonl_file.stem

            if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (session_id,)).fetchone():
                continue

            messages, first_ts, last_ts = _extract_claude(jsonl_file)
            if len(messages) < min_messages:
                continue

            first_user = next((m["content"][:80] for m in messages if m["role"] == "user"), "")

            conn.execute(
                "INSERT INTO conversations (id, source, name, project, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, "claude", first_user, project_dir.name,
                 first_ts or datetime.now().isoformat(),
                 last_ts or datetime.now().isoformat(),
                 len(messages))
            )
            for i, msg in enumerate(messages):
                conn.execute(
                    "INSERT OR IGNORE INTO messages (conversation_id, position, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (session_id, i, msg["role"], msg["content"], msg.get("timestamp", ""))
                )
            conn.commit()
            exported += 1
            logger.info(f"  [claude] {project_dir.name}/{session_id[:8]}... — {len(messages)} msgs")

    return exported


def _extract_claude(jsonl_path: Path) -> tuple[list[dict], str, str]:
    messages = []
    first_ts = ""
    last_ts = ""

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") not in ("user", "assistant"):
                continue

            ts = entry.get("timestamp", "")
            if ts and not first_ts:
                first_ts = ts
            if ts:
                last_ts = ts

            msg_raw = entry.get("message", "")
            if isinstance(msg_raw, str):
                try:
                    msg_raw = json.loads(msg_raw)
                except (json.JSONDecodeError, ValueError):
                    if msg_raw.strip():
                        messages.append({"role": entry["type"], "content": msg_raw[:MSG_CONTENT_CAP], "timestamp": ts})
                    continue

            if not isinstance(msg_raw, dict):
                continue

            role = msg_raw.get("role", entry["type"])
            content = msg_raw.get("content", "")

            if isinstance(content, list):
                texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = "\n".join(texts)
            elif not isinstance(content, str):
                continue

            content = content.strip()
            if not content or is_noise(content):
                continue

            messages.append({"role": role, "content": content[:MSG_CONTENT_CAP], "timestamp": ts})

    return messages, first_ts, last_ts


# ── Cursor ───────────────────────────────────────────────────

def export_cursor(conn: sqlite3.Connection, name_filter: str = None, composer_id: str = None, min_messages: int = 4) -> int:
    if not CURSOR_DB.exists():
        logger.warning(f"Cursor DB not found: {CURSOR_DB}")
        return 0

    cursor_conn = sqlite3.connect(f"file:{CURSOR_DB}?mode=ro", uri=True)
    try:
        rows = cursor_conn.execute(
            "SELECT key, value FROM cursorDiskKV WHERE key LIKE 'composerData:%'"
        ).fetchall()
    finally:
        cursor_conn.close()

    exported = 0
    for key, value in rows:
        cid = key.replace("composerData:", "")
        if composer_id and composer_id not in cid:
            continue

        try:
            data = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            continue

        name = data.get("name", "")
        if name_filter and name_filter.lower() not in name.lower():
            continue

        headers = data.get("fullConversationHeadersOnly", [])
        if not headers:
            continue

        if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone():
            continue

        created_ts = data.get("createdAt", 0)
        updated_ts = data.get("lastUpdatedAt", created_ts)
        created = datetime.fromtimestamp(created_ts / 1000).isoformat(timespec="seconds") if created_ts else datetime.now().isoformat()
        updated = datetime.fromtimestamp(updated_ts / 1000).isoformat(timespec="seconds") if updated_ts else created

        messages = _extract_cursor(cid, headers)
        if len(messages) < min_messages:
            continue

        conn.execute(
            "INSERT INTO conversations (id, source, name, project, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, "cursor", name, "", created, updated, len(messages))
        )
        for i, msg in enumerate(messages):
            conn.execute(
                "INSERT OR IGNORE INTO messages (conversation_id, position, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (cid, i, msg["role"], msg["content"], msg.get("timestamp", ""))
            )
        conn.commit()
        exported += 1
        logger.info(f"  [cursor] {name or cid[:8]}... — {len(messages)} msgs")

    return exported


def _extract_cursor(session_id: str, bubble_headers: list[dict]) -> list[dict]:
    messages = []
    cursor_conn = sqlite3.connect(f"file:{CURSOR_DB}?mode=ro", uri=True)
    try:
        for header in bubble_headers:
            bubble_id = header.get("bubbleId", "")
            msg_type = header.get("type")
            if msg_type not in (1, 2):
                continue

            row = cursor_conn.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                (f"bubbleId:{session_id}:{bubble_id}",)
            ).fetchone()
            if not row:
                continue

            try:
                bubble = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                continue

            text = bubble.get("text", "").strip()
            if not text or is_noise(text):
                continue

            role = "user" if msg_type == 1 else "assistant"
            ts = bubble.get("createdAt", "")
            messages.append({"role": role, "content": text[:MSG_CONTENT_CAP], "timestamp": ts})
    finally:
        cursor_conn.close()
    return messages


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export conversations to SQLite")
    parser.add_argument("-o", "--output", default="conversations.db")
    parser.add_argument("--source", choices=["claude", "cursor", "all"], default="all")
    parser.add_argument("--project", default=None, help="Filter Claude by project")
    parser.add_argument("--name", default=None, help="Filter Cursor by name")
    parser.add_argument("--id", default=None, help="Export specific conversation ID")
    parser.add_argument("--min-messages", type=int, default=6)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S",
    )

    conn = init_db(args.output)
    total = 0

    if args.source in ("claude", "all"):
        logger.info("Exporting Claude Code conversations...")
        total += export_claude(conn, args.project, args.min_messages)

    if args.source in ("cursor", "all"):
        logger.info("Exporting Cursor conversations...")
        total += export_cursor(conn, args.name, args.id, args.min_messages)

    conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM conversations WHERE status = 'pending'").fetchone()[0]
    conn.close()

    logger.info(f"\nExported {total} new conversation(s) to {args.output}")
    logger.info(f"Total: {conv_count} conversations, {msg_count} messages ({pending} pending)")


if __name__ == "__main__":
    main()
