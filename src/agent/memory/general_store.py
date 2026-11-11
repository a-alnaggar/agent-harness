"""Repository for General Memories with contradiction upsert, BM25 search, and pruning."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional

from agent.tools.base import ToolError
from .privacy import validate_safe_content
from .schema import sanitize_fts_query


@dataclass
class GeneralMemoryEntry:
    id: int
    key: Optional[str]
    content: str
    importance: float
    confidence: float
    pinned: bool
    save_reason: str
    access_count: int
    last_used_run: int
    archived_at_run: Optional[int]
    status: str
    retention_score: float = 0.0


class GeneralMemoryRepository:
    """Manages persistent general memories, weighted retention scores, and lifecycle pruning."""

    def __init__(
        self,
        connection_provider: Callable[[], ContextManager[sqlite3.Connection]],
        run_provider: Callable[[], int],
    ):
        self._connection = connection_provider
        self._get_current_run = run_provider

    def compute_retention_score(
        self,
        importance: float,
        confidence: float,
        access_count: int,
        save_reason: str,
    ) -> float:
        """Calculates the weighted dynamic retention score."""
        reason_weights = {"explicit": 1.0, "system": 0.8, "inferred": 0.5}
        reason_weight = reason_weights.get(save_reason, 0.5)
        normalized_access = min(1.0, access_count / 10.0)

        score = (
            0.40 * importance
            + 0.25 * confidence
            + 0.20 * normalized_access
            + 0.15 * reason_weight
        )
        return round(score, 4)

    def save_general(
        self,
        content: str,
        key: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        pinned: bool = False,
        save_reason: str = "inferred",
        source_type: str = "user",
        source_ref: Optional[str] = None,
    ) -> str:
        """Stores or updates general memory with instant contradiction resolution on key."""
        clean_key = key.strip().lower().replace(" ", "_") if key else None
        clean_content = content.strip()
        if not clean_content:
            raise ToolError("Memory content must not be empty.")
        if save_reason not in {"explicit", "inferred", "system"}:
            raise ToolError("save_reason must be one of: explicit, inferred, system.")
        if source_type not in {"user", "gmail", "pdf", "system", "unverified"}:
            raise ToolError("source_type must be one of: user, gmail, pdf, system, unverified.")
        validate_safe_content(f"{clean_key or ''}: {clean_content} {source_ref or ''}")
        clean_importance = max(0.0, min(1.0, float(importance)))
        clean_confidence = max(0.0, min(1.0, float(confidence)))
        current_run = self._get_current_run()

        with self._connection() as conn:
            cursor = conn.cursor()

            # Automatic contradiction resolution via key upsert
            if clean_key:
                cursor.execute("SELECT id FROM general_memories WHERE key = ?", (clean_key,))
                existing = cursor.fetchone()
                if existing:
                    row_id = existing["id"]
                    cursor.execute(
                        """
                        UPDATE general_memories
                        SET content = ?, importance = ?, confidence = ?, pinned = ?,
                            save_reason = ?, access_count = access_count + 1,
                            last_used_run = ?, status = 'active', archived_at_run = NULL,
                            source_type = ?, source_ref = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (
                            clean_content,
                            clean_importance,
                            clean_confidence,
                            1 if pinned else 0,
                            save_reason,
                            current_run,
                            source_type,
                            source_ref,
                            row_id,
                        ),
                    )
                    conn.commit()
                    return f"Overwrote and updated memory key '{clean_key}' with new content."

            # Insert new entry
            cursor.execute(
                """
                INSERT INTO general_memories
                    (key, content, importance, confidence, pinned, save_reason, access_count,
                     last_used_run, status, source_type, source_ref)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, 'active', ?, ?)
                """,
                (
                    clean_key,
                    clean_content,
                    clean_importance,
                    clean_confidence,
                    1 if pinned else 0,
                    save_reason,
                    current_run,
                    source_type,
                    source_ref,
                ),
            )
            conn.commit()

        desc = f"'{clean_key}'" if clean_key else "fact"
        return f"Saved general memory {desc} (pinned={pinned}, importance={clean_importance})."

    def search_general(self, query: str, top_k: int = 5) -> list[GeneralMemoryEntry]:
        """Searches active general memories using SQLite FTS5."""
        sanitized = sanitize_fts_query(query)
        if not sanitized:
            return []

        current_run = self._get_current_run()
        results: list[GeneralMemoryEntry] = []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.* FROM general_memories m
                JOIN general_memories_fts ON general_memories_fts.rowid = m.id
                WHERE general_memories_fts MATCH ? AND m.status = 'active'
                ORDER BY bm25(general_memories_fts)
                LIMIT ?
                """,
                (sanitized, top_k),
            )

            rows = cursor.fetchall()

            for row in rows:
                score = self.compute_retention_score(
                    importance=row["importance"],
                    confidence=row["confidence"],
                    access_count=row["access_count"],
                    save_reason=row["save_reason"],
                )
                results.append(
                    GeneralMemoryEntry(
                        id=row["id"],
                        key=row["key"],
                        content=row["content"],
                        importance=row["importance"],
                        confidence=row["confidence"],
                        pinned=bool(row["pinned"]),
                        save_reason=row["save_reason"],
                        access_count=row["access_count"],
                        last_used_run=row["last_used_run"],
                        archived_at_run=row["archived_at_run"],
                        status=row["status"],
                        retention_score=score,
                    )
                )
                conn.execute(
                    "UPDATE general_memories SET access_count = access_count + 1, last_used_run = ? WHERE id = ?",
                    (current_run, row["id"]),
                )
            conn.commit()
        return results

    def delete_general(self, key_or_id: str) -> bool:
        """Deletes a memory explicitly by key or stringified id."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM general_memories WHERE key = ? OR CAST(id AS TEXT) = ?",
                (key_or_id, key_or_id),
            )
            row = cursor.fetchone()
            if not row:
                return False

            row_id = row["id"]
            conn.execute("DELETE FROM general_memories WHERE id = ?", (row_id,))
            conn.commit()
            return True

    def has_general_key(self, key: str) -> bool:
        clean_key = key.strip().lower().replace(" ", "_")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM general_memories WHERE key = ? LIMIT 1", (clean_key,)
            ).fetchone()
            return row is not None

    def get_general_value(self, key: str) -> str | None:
        clean_key = key.strip().lower().replace(" ", "_")
        with self._connection() as conn:
            row = conn.execute(
                "SELECT content FROM general_memories WHERE key = ? LIMIT 1", (clean_key,)
            ).fetchone()
            return str(row["content"]) if row is not None else None

    def prune_memory(self, min_idle_runs: int = 50) -> dict[str, int]:
        """Executes the safe lifecycle pruning for general memories and course FAQs."""
        if min_idle_runs < 1:
            raise ValueError("min_idle_runs must be at least 1")
        current_run = self._get_current_run()
        archived_count = 0
        deleted_count = 0
        faq_archived_count = 0
        faq_deleted_count = 0

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM general_memories")
            rows = cursor.fetchall()

            for row in rows:
                # Absolute immunity check
                if row["pinned"] == 1 or row["save_reason"] == "explicit":
                    continue

                score = self.compute_retention_score(
                    importance=row["importance"],
                    confidence=row["confidence"],
                    access_count=row["access_count"],
                    save_reason=row["save_reason"],
                )

                row_id = row["id"]
                status = row["status"]
                last_used = row["last_used_run"]
                archived_at = row["archived_at_run"]

                # Case 1: Active memory transition to Archived
                if status == "active":
                    if score < 0.4 and (current_run - last_used) >= min_idle_runs:
                        cursor.execute(
                            "UPDATE general_memories SET status = 'archived', archived_at_run = ? WHERE id = ?",
                            (current_run, row_id),
                        )
                        archived_count += 1

                # Case 2: Archived memory transition to Deleted
                elif status == "archived":
                    idle_archived_runs = (current_run - archived_at) if archived_at else (current_run - last_used)
                    if score < 0.2 and idle_archived_runs >= min_idle_runs:
                        cursor.execute("DELETE FROM general_memories WHERE id = ?", (row_id,))
                        deleted_count += 1

            faq_rows = cursor.execute("SELECT * FROM course_faqs").fetchall()
            for row in faq_rows:
                status = row["status"]
                occurrence_count = int(row["occurrence_count"] or 1)
                successful_uses = int(row["successful_use_count"] or 0)
                idle_runs = current_run - int(row["last_used_run"] or 1)
                low_value = occurrence_count < 3 and successful_uses == 0 and row["confidence"] < 0.8
                if status == "candidate" and low_value and idle_runs >= min_idle_runs:
                    cursor.execute(
                        "UPDATE course_faqs SET status = 'archived', archived_at_run = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (current_run, row["id"]),
                    )
                    faq_archived_count += 1
                elif status == "archived" and low_value:
                    archived_at = int(row["archived_at_run"] or row["last_used_run"] or 1)
                    if current_run - archived_at >= min_idle_runs:
                        cursor.execute("DELETE FROM course_faqs WHERE id = ?", (row["id"],))
                        faq_deleted_count += 1

            conn.commit()

        return {
            "current_run": current_run,
            "archived_count": archived_count,
            "deleted_count": deleted_count,
            "faq_archived_count": faq_archived_count,
            "faq_deleted_count": faq_deleted_count,
        }
