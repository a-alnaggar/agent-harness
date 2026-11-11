"""Repository for Course FAQs with deduplication, FTS5 search, and hit tracking."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, ContextManager, Optional

from agent.tools.base import ToolError
from .privacy import validate_safe_content
from .schema import sanitize_fts_query


def normalize_question(question: str) -> str:
    """Normalizes a question string for strict deduplication."""
    cleaned = re.sub(r"[?!.,:;\"'()\[\]{}]", " ", question.lower())
    return " ".join(cleaned.split())


@dataclass
class FaqEntry:
    id: int
    course_id: str
    question: str
    answer: str
    source_type: str
    source_ref: Optional[str]
    confidence: float
    occurrence_count: int
    retrieval_count: int
    successful_use_count: int
    last_used_run: int
    status: str

    @property
    def hit_count(self) -> int:
        """Backward-compatible alias for the old FAQ occurrence counter."""
        return self.occurrence_count


class CourseFaqRepository:
    """Manages course-linked Q&A pairs, deduplication, and full-text retrieval."""

    def __init__(
        self,
        connection_provider: Callable[[], ContextManager[sqlite3.Connection]],
        run_provider: Callable[[], int],
    ):
        self._connection = connection_provider
        self._get_current_run = run_provider

    def save_faq(
        self,
        course_id: str,
        question: str,
        answer: str,
        source_type: str = "unverified",
        source_ref: Optional[str] = None,
        confidence: float = 1.0,
    ) -> str:
        """Saves a course FAQ. Deduplicates on (course_id, normalized_question)."""
        clean_course = course_id.strip().lower()
        clean_question = question.strip()
        normalized = normalize_question(clean_question)
        clean_answer = answer.strip()
        if not clean_course or not clean_question or not clean_answer:
            raise ToolError("course_id, question, and answer must not be empty.")
        allowed_sources = {"pdf", "gmail", "user", "unverified"}
        if source_type not in allowed_sources:
            raise ToolError(f"source_type must be one of: {', '.join(sorted(allowed_sources))}.")
        clean_confidence = max(0.0, min(1.0, float(confidence)))
        current_run = self._get_current_run()

        validate_safe_content(f"{clean_question} {clean_answer} {source_ref or ''}")
        trusted_source = source_type in {"pdf", "gmail", "user"} and clean_confidence >= 0.8

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM course_faqs WHERE course_id = ? AND normalized_question = ?",
                (clean_course, normalized),
            )
            existing = cursor.fetchone()

            if existing:
                row_id = existing["id"]
                occurrence_count = int(existing["occurrence_count"] or 1) + 1
                next_status = (
                    "active"
                    if existing["status"] == "active" or trusted_source or occurrence_count >= 3
                    else "candidate"
                )
                if clean_confidence >= existing["confidence"]:
                    cursor.execute(
                        """
                        UPDATE course_faqs
                        SET question = ?, answer = ?, source_type = ?, source_ref = ?, confidence = ?,
                            hit_count = hit_count + 1, occurrence_count = ?, last_used_run = ?,
                            status = ?, archived_at_run = NULL, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (
                            clean_question,
                            clean_answer,
                            source_type,
                            source_ref,
                            clean_confidence,
                            occurrence_count,
                            current_run,
                            next_status,
                            row_id,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE course_faqs
                        SET hit_count = hit_count + 1, occurrence_count = ?, last_used_run = ?,
                            status = ?, archived_at_run = NULL, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (occurrence_count, current_run, next_status, row_id),
                    )
                conn.commit()
                return (
                    f"Updated existing FAQ for course '{clean_course}' "
                    f"(occurrences={occurrence_count}, status={next_status})."
                )

            initial_status = "active" if trusted_source else "candidate"
            cursor.execute(
                """
                INSERT INTO course_faqs
                    (course_id, question, normalized_question, answer, source_type, source_ref,
                     confidence, hit_count, occurrence_count, retrieval_count,
                     successful_use_count, last_used_run, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, 1, 0, 0, ?, ?)
                """,
                (
                    clean_course,
                    clean_question,
                    normalized,
                    clean_answer,
                    source_type,
                    source_ref,
                    clean_confidence,
                    current_run,
                    initial_status,
                ),
            )
            conn.commit()

        return (
            f"Saved new FAQ for course '{clean_course}' with status '{initial_status}': "
            f"'{clean_question}'."
        )

    def search_faq(self, course_id: str, query: str, top_k: int = 5) -> list[FaqEntry]:
        """Searches course FAQs using FTS5 scoped by course_id."""
        sanitized = sanitize_fts_query(query)
        if not sanitized:
            return []

        clean_course = course_id.strip().lower()
        current_run = self._get_current_run()
        results: list[FaqEntry] = []

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT q.* FROM course_faqs q
                JOIN course_faqs_fts ON course_faqs_fts.rowid = q.id
                WHERE q.course_id = ? AND q.status = 'active' AND course_faqs_fts MATCH ?
                ORDER BY bm25(course_faqs_fts)
                LIMIT ?
                """,
                (clean_course, sanitized, top_k),
            )

            rows = cursor.fetchall()

            for row in rows:
                results.append(
                    FaqEntry(
                        id=row["id"],
                        course_id=row["course_id"],
                        question=row["question"],
                        answer=row["answer"],
                        source_type=row["source_type"],
                        source_ref=row["source_ref"],
                        confidence=row["confidence"],
                        occurrence_count=row["occurrence_count"],
                        retrieval_count=row["retrieval_count"],
                        successful_use_count=row["successful_use_count"],
                        last_used_run=row["last_used_run"],
                        status=row["status"],
                    )
                )
                conn.execute(
                    """
                    UPDATE course_faqs
                    SET retrieval_count = retrieval_count + 1, last_used_run = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (current_run, row["id"]),
                )
            conn.commit()
        return results

    def record_faq_outcome(self, faq_id: int, successful: bool) -> bool:
        """Records whether a retrieved FAQ was actually useful to an answer."""
        if not successful:
            with self._connection() as conn:
                row = conn.execute("SELECT 1 FROM course_faqs WHERE id = ?", (faq_id,)).fetchone()
                return row is not None
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE course_faqs
                SET successful_use_count = successful_use_count + 1,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (faq_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
