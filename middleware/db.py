"""SQLite database with FTS5 for memory storage and retrieval."""

import sqlite3
import os
import time
from typing import Optional


def get_connection(db_path: str) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: str) -> None:
    """Initialize the database schema."""
    conn = get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic TEXT NOT NULL,
                fact TEXT NOT NULL,
                source_session TEXT,
                source_channel TEXT,
                importance INTEGER DEFAULT 5,
                created_at REAL NOT NULL,
                last_accessed_at REAL
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                channel TEXT,
                user_id TEXT,
                summary TEXT NOT NULL,
                message_count INTEGER,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
            CREATE INDEX IF NOT EXISTS idx_summaries_user ON summaries(user_id);
        """)

        # Create FTS5 virtual table (separate because CREATE VIRTUAL TABLE
        # doesn't support IF NOT EXISTS in all SQLite versions)
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE memories_fts USING fts5(
                    user_id,
                    topic,
                    fact,
                    content=memories,
                    content_rowid=id,
                    tokenize='porter unicode61'
                )
            """)
        except sqlite3.OperationalError:
            pass  # Already exists

        # Triggers to keep FTS5 in sync with memories table
        for trigger_sql in [
            """CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, user_id, topic, fact)
                VALUES (new.id, new.user_id, new.topic, new.fact);
            END""",
            """CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, user_id, topic, fact)
                VALUES ('delete', old.id, old.user_id, old.topic, old.fact);
            END""",
            """CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, user_id, topic, fact)
                VALUES ('delete', old.id, old.user_id, old.topic, old.fact);
                INSERT INTO memories_fts(rowid, user_id, topic, fact)
                VALUES (new.id, new.user_id, new.topic, new.fact);
            END""",
        ]:
            try:
                conn.execute(trigger_sql)
            except sqlite3.OperationalError:
                pass  # Already exists

        # Add embedding column (ALTER TABLE doesn't support IF NOT EXISTS)
        try:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # Column already exists

        conn.commit()
    finally:
        conn.close()


def store_memories(db_path: str, memories: list[dict]) -> tuple[int, list[int]]:
    """Store extracted memories. Returns (count, list_of_row_ids)."""
    if not memories:
        return 0, []
    conn = get_connection(db_path)
    try:
        now = time.time()
        count = 0
        row_ids: list[int] = []
        for mem in memories:
            cursor = conn.execute(
                """INSERT INTO memories (user_id, topic, fact, source_session,
                   source_channel, importance, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem.get("user_id"),
                    mem["topic"],
                    mem["fact"],
                    mem.get("source_session"),
                    mem.get("source_channel"),
                    mem.get("importance", 5),
                    now,
                ),
            )
            row_ids.append(cursor.lastrowid)
            count += 1
        conn.commit()
        return count, row_ids
    finally:
        conn.close()


def store_summary(db_path: str, session_id: str, channel: Optional[str],
                  user_id: Optional[str], summary: str,
                  message_count: int) -> int:
    """Store a session summary. Returns the summary ID."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO summaries (session_id, channel, user_id, summary,
               message_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, channel, user_id, summary, message_count, time.time()),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def search_memories(db_path: str, query: str,
                    user_id: Optional[str] = None,
                    limit: int = 20) -> list[dict]:
    """Search memories using FTS5. Returns matching memories ranked by relevance."""
    conn = get_connection(db_path)
    try:
        # Build FTS5 query - tokenize into individual words, strip punctuation
        import re
        words = re.findall(r'\w+', query)
        if not words:
            return []
        # Quote each word to avoid FTS5 syntax issues, join with OR for broad matching
        fts_query = " OR ".join(f'"{w}"' for w in words)

        if user_id:
            rows = conn.execute(
                """SELECT m.id, m.user_id, m.topic, m.fact, m.importance,
                          m.created_at, m.source_session, m.source_channel,
                          rank
                   FROM memories_fts fts
                   JOIN memories m ON m.id = fts.rowid
                   WHERE memories_fts MATCH ?
                     AND m.user_id = ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT m.id, m.user_id, m.topic, m.fact, m.importance,
                          m.created_at, m.source_session, m.source_channel,
                          rank
                   FROM memories_fts fts
                   JOIN memories m ON m.id = fts.rowid
                   WHERE memories_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()

        results = [dict(row) for row in rows]

        # Update last_accessed_at for retrieved memories
        if results:
            now = time.time()
            ids = [r["id"] for r in results]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE memories SET last_accessed_at = ? WHERE id IN ({placeholders})",
                [now] + ids,
            )
            conn.commit()

        return results
    finally:
        conn.close()


def get_recent_memories(db_path: str, user_id: Optional[str] = None,
                        limit: int = 10) -> list[dict]:
    """Get most recent memories, optionally filtered by user."""
    conn = get_connection(db_path)
    try:
        if user_id:
            rows = conn.execute(
                """SELECT id, user_id, topic, fact, importance, created_at,
                          source_session, source_channel
                   FROM memories
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, user_id, topic, fact, importance, created_at,
                          source_session, source_channel
                   FROM memories
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def store_embedding(db_path: str, memory_id: int, embedding_blob: bytes) -> bool:
    """Store a precomputed embedding BLOB for a memory. Returns True on success."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (embedding_blob, memory_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_memories_with_embeddings(db_path: str, user_id: Optional[str] = None,
                                  limit: int = 500) -> list[dict]:
    """Get memories that have embeddings, optionally filtered by user."""
    conn = get_connection(db_path)
    try:
        if user_id:
            rows = conn.execute(
                """SELECT id, user_id, topic, fact, importance, created_at,
                          source_session, source_channel, embedding
                   FROM memories
                   WHERE embedding IS NOT NULL AND user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, user_id, topic, fact, importance, created_at,
                          source_session, source_channel, embedding
                   FROM memories
                   WHERE embedding IS NOT NULL
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_memories_without_embeddings(db_path: str) -> list[dict]:
    """Get all memories that do not yet have an embedding."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT id, user_id, topic, fact, importance, created_at,
                      source_session, source_channel
               FROM memories
               WHERE embedding IS NULL
               ORDER BY id""",
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_stats(db_path: str) -> dict:
    """Get database statistics."""
    conn = get_connection(db_path)
    try:
        memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        summary_count = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        unique_users = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM memories WHERE user_id IS NOT NULL"
        ).fetchone()[0]
        unique_topics = conn.execute(
            "SELECT COUNT(DISTINCT topic) FROM memories"
        ).fetchone()[0]
        return {
            "total_memories": memory_count,
            "total_summaries": summary_count,
            "unique_users": unique_users,
            "unique_topics": unique_topics,
        }
    finally:
        conn.close()
