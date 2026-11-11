"""SQLite FTS5-based indexing and search engine for course materials."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — declared in pyproject.toml dependencies

from .models import SearchResult


def _sanitize_fts5_query(raw_query: str) -> str:
    """Sanitizes user query for SQLite FTS5 matching, preventing syntax errors."""
    tokens = re.findall(r"\w+", raw_query)
    if not tokens:
        return ""
    # Search for tokens using prefix matching
    return " OR ".join(f'"{token}"*' for token in tokens)


class PdfKnowledgeIndexer:
    """Indexes and queries PDF documents using SQLite FTS5 with SHA-256 hash caching.

    Supports both single-database mode (via db_path) and modular per-course databases
    (via base_courses_dir, where each course has its own folder and index.db).
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        base_courses_dir: Optional[Path] = None,
    ):
        if db_path is None and base_courses_dir is None:
            raise ValueError("Either db_path or base_courses_dir must be provided")

        self.db_path = db_path
        self.base_courses_dir = base_courses_dir

        if self.db_path and self.db_path.suffix == ".db":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db(self.db_path)

    def _resolve_db_path(self, course_id: Optional[str] = None) -> Path:
        if self.db_path and self.db_path.suffix == ".db":
            return self.db_path
        if self.base_courses_dir and course_id:
            course_dir = self.base_courses_dir / course_id
            course_dir.mkdir(parents=True, exist_ok=True)
            return course_dir / "index.db"
        if self.db_path:
            return self.db_path
        if self.base_courses_dir:
            return self.base_courses_dir / "default_index.db"
        raise ValueError("Cannot resolve database path")

    def _get_connection(self, course_id: Optional[str] = None) -> sqlite3.Connection:
        target_path = self._resolve_db_path(course_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db(target_path)
        conn = sqlite3.connect(str(target_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self, db_file: Path) -> None:
        with sqlite3.connect(str(db_file)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indexed_files (
                    course_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (course_id, document_id)
                );
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS course_chunks_fts USING fts5(
                    course_id,
                    document_id,
                    document_title,
                    page_number UNINDEXED,
                    chunk_number UNINDEXED,
                    chunk_text
                );
            """)
            conn.commit()

    def get_course_db_path(self, course_id: str) -> Path:
        """Returns the path to the database file for the given course."""
        return self._resolve_db_path(course_id)

    def index_pdf(
        self,
        course_id: str,
        document_id: str,
        document_title: str,
        file_path: Path,
    ) -> bool:
        """Indexes a PDF file. If the file hash matches existing index, indexing is skipped.

        Returns True if newly indexed, False if skipped (cached).
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        file_bytes = file_path.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        with self._get_connection(course_id) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_hash FROM indexed_files WHERE course_id = ? AND document_id = ?",
                (course_id, document_id),
            )
            row = cursor.fetchone()
            if row and row["file_hash"] == file_hash:
                return False  # Already indexed and unchanged

            # Remove previous chunks if re-indexing modified file
            conn.execute(
                "DELETE FROM course_chunks_fts WHERE course_id = ? AND document_id = ?",
                (course_id, document_id),
            )

            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if not text:
                    continue

                # Split page text into coherent paragraph chunks
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                if not paragraphs:
                    paragraphs = [text]

                for chunk_idx, chunk in enumerate(paragraphs):
                    conn.execute(
                        """
                        INSERT INTO course_chunks_fts
                            (course_id, document_id, document_title, page_number, chunk_number, chunk_text)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (course_id, document_id, document_title, page_num + 1, chunk_idx, chunk),
                    )

            doc.close()

            conn.execute(
                "INSERT OR REPLACE INTO indexed_files (course_id, document_id, file_path, file_hash) VALUES (?, ?, ?, ?)",
                (course_id, document_id, str(file_path.resolve()), file_hash),
            )
            conn.commit()

        return True

    def search(
        self,
        course_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Searches course materials strictly within the specified course_id using FTS5."""
        sanitized = _sanitize_fts5_query(query)
        if not sanitized:
            return []

        top_k = max(1, min(20, int(top_k)))

        with self._get_connection(course_id) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT course_id, document_id, document_title, page_number, chunk_text, rank
                FROM course_chunks_fts
                WHERE course_id = ? AND course_chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (course_id, sanitized, top_k),
            )
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append(
                    SearchResult(
                        course_id=row["course_id"],
                        document_id=row["document_id"],
                        document_title=row["document_title"],
                        page_number=int(row["page_number"]),
                        snippet=row["chunk_text"],
                        rank=float(row["rank"]),
                    )
                )
            return results

    def get_page_text(
        self,
        course_id: str,
        document_id: str,
        page_number: int,
    ) -> Optional[str]:
        """Retrieves full text of a specific page within a course document."""
        with self._get_connection(course_id) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_text
                FROM course_chunks_fts
                WHERE course_id = ? AND document_id = ? AND page_number = ?
                ORDER BY chunk_number
                """,
                (course_id, document_id, page_number),
            )
            rows = cursor.fetchall()
            if not rows:
                return None
            return "\n\n".join(row["chunk_text"] for row in rows)
