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
    """Filter only system/tool junk. Keep code-heavy responses —
    they often contain valuable explanations around the code."""
    for pattern in NOISE_PATTERNS:
        if pattern.search(text):
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


# ── Gemini (Google Takeout HTML) ──────────────────────────────

def export_gemini(conn: sqlite3.Connection, html_path: str, min_messages: int = 4, gap_minutes: int = 30) -> int:
    """Import Gemini conversations from Google Takeout MyActivity.html.

    Gemini doesn't thread conversations. We group entries by time proximity:
    entries within gap_minutes of each other become one conversation.
    """
    path = Path(html_path)
    if path.is_dir():
        candidates = list(path.rglob("MyActivity.html"))
        if not candidates:
            logger.warning(f"No MyActivity.html found in {html_path}")
            return 0
        html_file = candidates[0]
    elif path.is_file():
        html_file = path
    else:
        logger.warning(f"Path not found: {html_path}")
        return 0

    from html import unescape
    from datetime import timedelta
    import hashlib

    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse each outer-cell block
    outer_blocks = content.split('class="outer-cell')[1:]
    entries = []

    for block in outer_blocks:
        m = re.search(
            r'class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)(?=<div class="content-cell)',
            block, re.DOTALL
        )
        if not m:
            continue

        raw = m.group(1)
        clean = re.sub(r'<br\s*/?>', '\n', raw)
        clean = re.sub(r'<[^>]+>', '', clean)
        lines = [unescape(l.strip()) for l in clean.split('\n') if l.strip()]

        if not lines or not lines[0].startswith('Prompted'):
            continue

        user_text = lines[0].replace('Prompted', '', 1).strip()

        # Find date line and split user/assistant
        date_str = ''
        parsed_dt = None
        response_start = 1
        for i, line in enumerate(lines[1:], 1):
            if re.match(r'[A-Z][a-z]{2} \d{1,2}, \d{4},', line):
                date_str = line
                response_start = i + 1
                try:
                    # "Mar 31, 2026, 11:07:57 PM EDT"
                    clean_date = re.sub(r'\s+[A-Z]{2,4}$', '', date_str)
                    parsed_dt = datetime.strptime(clean_date, "%b %d, %Y, %I:%M:%S %p")
                except ValueError:
                    pass
                break

        response = '\n'.join(lines[response_start:])

        if user_text and parsed_dt:
            entries.append({
                'user': user_text[:MSG_CONTENT_CAP],
                'assistant': response[:MSG_CONTENT_CAP] if response else '',
                'date': parsed_dt,
                'date_str': parsed_dt.isoformat(timespec="seconds") if parsed_dt else '',
            })

    if not entries:
        logger.warning("No Gemini entries parsed")
        return 0

    # Sort chronologically
    entries.sort(key=lambda e: e['date'])

    # Group into conversations by time gap
    conversations = []
    current_conv = [entries[0]]

    for entry in entries[1:]:
        if (entry['date'] - current_conv[-1]['date']) <= timedelta(minutes=gap_minutes):
            current_conv.append(entry)
        else:
            conversations.append(current_conv)
            current_conv = [entry]
    conversations.append(current_conv)

    logger.info(f"  Parsed {len(entries)} entries → {len(conversations)} conversations (gap={gap_minutes}min)")

    exported = 0
    for conv_entries in conversations:
        # Build messages
        messages = []
        for entry in conv_entries:
            messages.append({"role": "user", "content": entry["user"], "timestamp": entry["date_str"]})
            if entry["assistant"]:
                messages.append({"role": "assistant", "content": entry["assistant"], "timestamp": entry["date_str"]})

        if len(messages) < min_messages:
            continue

        # Generate stable ID from first entry's date
        first_date = conv_entries[0]['date_str']
        conv_id = "gemini-" + hashlib.md5(first_date.encode()).hexdigest()[:16]

        if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone():
            continue

        name = conv_entries[0]['user'][:80]
        created = conv_entries[0]['date_str']
        updated = conv_entries[-1]['date_str']

        conn.execute(
            "INSERT INTO conversations (id, source, name, project, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conv_id, "gemini", name, "", created, updated, len(messages))
        )
        for i, msg in enumerate(messages):
            conn.execute(
                "INSERT OR IGNORE INTO messages (conversation_id, position, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                (conv_id, i, msg["role"], msg["content"], msg.get("timestamp", ""))
            )
        conn.commit()
        exported += 1
        logger.info(f"  [gemini] {name[:60]}... — {len(messages)} msgs")

    return exported


# ── Codex (OpenAI CLI) ────────────────────────────────────────

CODEX_DIR = Path.home() / ".codex" / "sessions"

def export_codex(conn: sqlite3.Connection, min_messages: int = 4) -> int:
    """Import Codex CLI conversations from ~/.codex/sessions/."""
    if not CODEX_DIR.exists():
        logger.warning(f"Codex sessions dir not found: {CODEX_DIR}")
        return 0

    exported = 0
    for jsonl_file in sorted(CODEX_DIR.rglob("*.jsonl")):
        # Extract session ID from filename: rollout-DATE-UUID.jsonl
        fname = jsonl_file.stem
        parts = fname.split("-", 2)
        if len(parts) < 2:
            continue
        # UUID is the last part after the date
        session_id = fname.rsplit("-", 5)
        session_id = "-".join(session_id[-5:]) if len(session_id) >= 5 else fname

        if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (session_id,)).fetchone():
            continue

        messages = []
        first_ts = ""
        last_ts = ""

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                ts = entry.get("timestamp", "")
                if ts and not first_ts:
                    first_ts = ts
                if ts:
                    last_ts = ts

                if entry.get("type") != "response_item":
                    continue

                payload = entry.get("payload", "")
                if isinstance(payload, str):
                    try:
                        payload = eval(payload)
                    except Exception:
                        continue
                if not isinstance(payload, dict):
                    continue

                role = payload.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                content = payload.get("content")
                if not content or not isinstance(content, list):
                    continue

                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text", "").strip()
                    btype = block.get("type", "")
                    if btype not in ("input_text", "output_text"):
                        continue
                    # Skip system/agent instructions
                    if text.startswith(("<permissions", "<environment", "# AGENTS.md")):
                        continue
                    if text and not is_noise(text):
                        messages.append({
                            "role": role,
                            "content": text[:MSG_CONTENT_CAP],
                            "timestamp": ts,
                        })

        if len(messages) < min_messages:
            continue

        name = next((m["content"][:80] for m in messages if m["role"] == "user"), "")

        conn.execute(
            "INSERT INTO conversations (id, source, name, project, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, "codex", name, "",
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
        logger.info(f"  [codex] {name[:60]}... — {len(messages)} msgs")

    return exported


# ── Claude.ai web export ─────────────────────────────────────

def export_claude_web(conn: sqlite3.Connection, json_path: str, min_messages: int = 4) -> int:
    """Import Claude.ai data export (conversations.json from Settings > Export)."""
    path = Path(json_path)

    # Accept a directory (the batch folder) or the JSON file directly
    if path.is_dir():
        candidates = list(path.glob("conversations.json"))
        if not candidates:
            candidates = list(path.glob("*.json"))
        files = candidates
    elif path.is_file():
        files = [path]
    else:
        logger.warning(f"Path not found: {json_path}")
        return 0

    exported = 0
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {f}: {e}")
            continue

        items = data if isinstance(data, list) else [data]

        for item in items:
            conv_id = item.get("uuid", "")
            if not conv_id:
                continue

            if conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conv_id,)).fetchone():
                continue

            chat_messages = item.get("chat_messages", [])
            name = item.get("name", "")
            created = item.get("created_at", "")
            updated = item.get("updated_at", created)

            messages = []
            for msg in chat_messages:
                sender = msg.get("sender", "")
                if sender not in ("human", "assistant"):
                    continue
                role = "user" if sender == "human" else "assistant"

                msg_ts = msg.get("created_at", "")
                texts = []
                for block in msg.get("content", []):
                    if block.get("type") == "text" and block.get("text", "").strip():
                        texts.append(block["text"][:MSG_CONTENT_CAP])
                        if not msg_ts and block.get("start_timestamp"):
                            msg_ts = block["start_timestamp"]

                content = "\n".join(texts)
                if not content:
                    continue

                messages.append({"role": role, "content": content, "timestamp": msg_ts})

            if len(messages) < min_messages:
                continue

            conn.execute(
                "INSERT INTO conversations (id, source, name, project, created_at, updated_at, message_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conv_id, "claude-web", name, "",
                 created or datetime.now().isoformat(),
                 updated or datetime.now().isoformat(),
                 len(messages))
            )
            for i, msg in enumerate(messages):
                conn.execute(
                    "INSERT OR IGNORE INTO messages (conversation_id, position, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (conv_id, i, msg["role"], msg["content"], msg.get("timestamp", ""))
                )
            conn.commit()
            exported += 1
            logger.info(f"  [claude-web] {name or conv_id[:8]}... — {len(messages)} msgs")

    return exported


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export conversations to SQLite")
    parser.add_argument("-o", "--output", default="conversations.db")
    parser.add_argument("--source", choices=["claude", "cursor", "codex", "claude-web", "gemini", "all"], default="all")
    parser.add_argument("--project", default=None, help="Filter Claude by project")
    parser.add_argument("--name", default=None, help="Filter Cursor by name")
    parser.add_argument("--id", default=None, help="Export specific conversation ID")
    parser.add_argument("--json", default=None, help="Claude.ai export file or batch directory")
    parser.add_argument("--html", default=None, help="Gemini Takeout HTML file or Takeout directory")
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

    if args.source in ("codex", "all"):
        logger.info("Exporting Codex conversations...")
        total += export_codex(conn, args.min_messages)

    if args.source == "claude-web" or (args.source == "all" and args.json):
        json_path = args.json
        if not json_path:
            logger.error("--json path required for claude-web source")
            sys.exit(1)
        logger.info(f"Exporting Claude.ai web conversations from {json_path}...")
        total += export_claude_web(conn, json_path, args.min_messages)

    if args.source == "gemini" or (args.source == "all" and args.html):
        html_path = args.html
        if not html_path:
            logger.error("--html path required for gemini source")
            sys.exit(1)
        logger.info(f"Exporting Gemini conversations from {html_path}...")
        total += export_gemini(conn, html_path, args.min_messages)

    conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM conversations WHERE status = 'pending'").fetchone()[0]
    conn.close()

    logger.info(f"\nExported {total} new conversation(s) to {args.output}")
    logger.info(f"Total: {conv_count} conversations, {msg_count} messages ({pending} pending)")


if __name__ == "__main__":
    main()
