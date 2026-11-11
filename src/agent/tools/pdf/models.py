"""Data models for course material search results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    course_id: str
    document_id: str
    document_title: str
    page_number: int
    snippet: str
    rank: float
