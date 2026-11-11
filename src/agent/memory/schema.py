"""Database schema, FTS5 virtual tables, triggers, and migrations for agent memory."""

from __future__ import annotations

import re
import sqlite3


def sanitize_fts_query(raw_query: str) -> str:
    """Extracts alphanumeric words and formats prefix match for SQLite FTS5."""
    tokens = re.findall(r"\w+", raw_query)
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"*' for token in tokens)


def init_memory_db(conn: sqlite3.Connection) -> None:
    """Initializes tables, triggers, and FTS5 indices on an open SQLite connection."""
    # 1. Metadata (tracks agent run count)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)
    conn.execute("""
        INSERT OR IGNORE INTO memory_metadata (key, value)
        VALUES ('current_run_count', '1');
    """)

    # 2. General Memories Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS general_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            content TEXT NOT NULL,
            importance REAL DEFAULT 0.5,
            confidence REAL DEFAULT 1.0,
            pinned INTEGER DEFAULT 0,
            save_reason TEXT DEFAULT 'inferred',
            access_count INTEGER DEFAULT 1,
            last_used_run INTEGER DEFAULT 1,
            archived_at_run INTEGER DEFAULT NULL,
            status TEXT DEFAULT 'active',
            source_type TEXT DEFAULT 'user',
            source_ref TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 3. Course FAQs Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS course_faqs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id TEXT NOT NULL,
            question TEXT NOT NULL,
            normalized_question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_ref TEXT,
            confidence REAL DEFAULT 1.0,
            hit_count INTEGER DEFAULT 1,
            occurrence_count INTEGER DEFAULT 1,
            retrieval_count INTEGER DEFAULT 0,
            successful_use_count INTEGER DEFAULT 0,
            last_used_run INTEGER DEFAULT 1,
            archived_at_run INTEGER DEFAULT NULL,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(course_id, normalized_question)
        );
    """)

    # Schema migrations for backward compatibility
    general_columns = {row[1] for row in conn.execute("PRAGMA table_info(general_memories)")}
    if "source_type" not in general_columns:
        conn.execute("ALTER TABLE general_memories ADD COLUMN source_type TEXT DEFAULT 'user'")
    if "source_ref" not in general_columns:
        conn.execute("ALTER TABLE general_memories ADD COLUMN source_ref TEXT")

    faq_columns = {row[1] for row in conn.execute("PRAGMA table_info(course_faqs)")}
    faq_additions = {
        "occurrence_count": "INTEGER DEFAULT 1",
        "retrieval_count": "INTEGER DEFAULT 0",
        "successful_use_count": "INTEGER DEFAULT 0",
        "archived_at_run": "INTEGER DEFAULT NULL",
        "updated_at": "TIMESTAMP",
    }
    for column, definition in faq_additions.items():
        if column not in faq_columns:
            conn.execute(f"ALTER TABLE course_faqs ADD COLUMN {column} {definition}")

    conn.execute(
        "UPDATE course_faqs SET occurrence_count = hit_count "
        "WHERE occurrence_count IS NULL OR occurrence_count < hit_count"
    )
    conn.execute(
        "UPDATE course_faqs SET updated_at = created_at WHERE updated_at IS NULL"
    )

    # Rebuild virtual tables and triggers to ensure schema consistency
    for trigger in (
        "general_memories_fts_ai",
        "general_memories_fts_au",
        "general_memories_fts_ad",
        "course_faqs_fts_ai",
        "course_faqs_fts_au",
        "course_faqs_fts_ad",
    ):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP TABLE IF EXISTS general_memories_fts")
    conn.execute("DROP TABLE IF EXISTS course_faqs_fts")
    conn.execute("""
        CREATE VIRTUAL TABLE general_memories_fts USING fts5(content, key);
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE course_faqs_fts USING fts5(
            question,
            answer,
            course_id UNINDEXED
        );
    """)
    conn.execute(
        "INSERT INTO general_memories_fts(rowid, content, key) "
        "SELECT id, content, COALESCE(key, '') FROM general_memories"
    )
    conn.execute(
        "INSERT INTO course_faqs_fts(rowid, question, answer, course_id) "
        "SELECT id, question, answer, course_id FROM course_faqs"
    )
    conn.executescript("""
        CREATE TRIGGER general_memories_fts_ai AFTER INSERT ON general_memories BEGIN
            INSERT INTO general_memories_fts(rowid, content, key)
            VALUES (new.id, new.content, COALESCE(new.key, ''));
        END;
        CREATE TRIGGER general_memories_fts_au
        AFTER UPDATE OF content, key ON general_memories BEGIN
            DELETE FROM general_memories_fts WHERE rowid = old.id;
            INSERT INTO general_memories_fts(rowid, content, key)
            VALUES (new.id, new.content, COALESCE(new.key, ''));
        END;
        CREATE TRIGGER general_memories_fts_ad AFTER DELETE ON general_memories BEGIN
            DELETE FROM general_memories_fts WHERE rowid = old.id;
        END;
        CREATE TRIGGER course_faqs_fts_ai AFTER INSERT ON course_faqs BEGIN
            INSERT INTO course_faqs_fts(rowid, question, answer, course_id)
            VALUES (new.id, new.question, new.answer, new.course_id);
        END;
        CREATE TRIGGER course_faqs_fts_au
        AFTER UPDATE OF question, answer, course_id ON course_faqs BEGIN
            DELETE FROM course_faqs_fts WHERE rowid = old.id;
            INSERT INTO course_faqs_fts(rowid, question, answer, course_id)
            VALUES (new.id, new.question, new.answer, new.course_id);
        END;
        CREATE TRIGGER course_faqs_fts_ad AFTER DELETE ON course_faqs BEGIN
            DELETE FROM course_faqs_fts WHERE rowid = old.id;
        END;
    """)
    conn.commit()
