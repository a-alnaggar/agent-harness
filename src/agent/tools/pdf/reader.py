"""PDF reading and search tools for document analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import Tool, ToolError
from ..builtin._paths import resolve_in_workspace
from ..permissions import PermissionTier


def _extract_pdf_pages(target: Path, start_page: int, max_pages: int) -> tuple[int, list[tuple[int, str]]]:
    """Extracts text page-by-page from a PDF file using PyMuPDF (fitz) or PyPDF2 fallback."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(target))
        total_pages = len(doc)
        pages_text: list[tuple[int, str]] = []

        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, start_idx + max_pages)

        for page_num in range(start_idx, end_idx):
            page = doc[page_num]
            text = page.get_text("text").strip()
            pages_text.append((page_num + 1, text or "(No extractable text on this page)"))

        doc.close()
        return total_pages, pages_text
    except ImportError:
        pass

    # Fallback to PyPDF2 if fitz is somehow unavailable
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(target))
        total_pages = len(reader.pages)
        pages_text = []

        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, start_idx + max_pages)

        for page_num in range(start_idx, end_idx):
            page = reader.pages[page_num]
            text = (page.extract_text() or "").strip()
            pages_text.append((page_num + 1, text or "(No extractable text on this page)"))

        return total_pages, pages_text
    except Exception as exc:
        raise ToolError(f"Failed to read PDF file '{target.name}': {exc}") from exc


class ReadPdfTool(Tool):
    name = "read_pdf"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Extract text from a PDF document page-by-page. "
        "Allows reading specific page ranges using start_page and max_pages."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the PDF file, relative to the workspace root.",
            },
            "start_page": {
                "type": "integer",
                "description": "1-based page number to start reading from (default: 1).",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum number of pages to read (default: 5, max: 20).",
            },
        },
        "required": ["path"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str, start_page: int = 1, max_pages: int = 5) -> str:
        target = resolve_in_workspace(self.root, path)
        if not target.is_file():
            raise ToolError(f"PDF file '{path}' does not exist")
        if target.suffix.lower() != ".pdf":
            raise ToolError(f"File '{path}' is not a PDF (expected .pdf extension)")

        start_page = max(1, int(start_page))
        max_pages = max(1, min(20, int(max_pages)))

        try:
            total_pages, pages = _extract_pdf_pages(target, start_page, max_pages)
        except Exception as exc:
            raise ToolError(f"Error reading PDF '{path}': {exc}") from exc

        if not pages:
            return f"[Document: {path} | Total Pages: {total_pages}] (Requested start_page {start_page} exceeds total pages)"

        end_page = pages[-1][0]
        header = f"[Document: {path} | Total Pages: {total_pages} | Showing Pages: {start_page}-{end_page}]\n"

        sections = []
        for page_num, text in pages:
            sections.append(f"--- Page {page_num} ---\n{text}")

        return header + "\n\n".join(sections)


class SearchPdfTool(Tool):
    name = "search_pdf"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Search for keywords or phrases across all pages in a PDF document. "
        "Returns matching page numbers and contextual snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the PDF file, relative to the workspace root.",
            },
            "query": {
                "type": "string",
                "description": "Keyword or phrase to search for (case-insensitive).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of match snippets to return (default: 10).",
            },
        },
        "required": ["path", "query"],
    }

    def __init__(self, root: Path):
        self.root = root

    def run(self, *, path: str, query: str, max_results: int = 10) -> str:
        target = resolve_in_workspace(self.root, path)
        if not target.is_file():
            raise ToolError(f"PDF file '{path}' does not exist")
        if target.suffix.lower() != ".pdf":
            raise ToolError(f"File '{path}' is not a PDF (expected .pdf extension)")

        query_str = query.strip()
        if not query_str:
            raise ToolError("Search query cannot be empty")

        query_lower = query_str.lower()
        max_results = max(1, min(50, int(max_results)))

        try:
            total_pages, all_pages = _extract_pdf_pages(target, start_page=1, max_pages=1000)
        except Exception as exc:
            raise ToolError(f"Error searching PDF '{path}': {exc}") from exc

        matches: list[str] = []
        for page_num, text in all_pages:
            lines = text.splitlines()
            for line_idx, line in enumerate(lines):
                if query_lower in line.lower():
                    # Extract small context around the matching line
                    snippet = line.strip()
                    matches.append(f"- Page {page_num}: \"{snippet}\"")
                    if len(matches) >= max_results:
                        break
            if len(matches) >= max_results:
                break

        if not matches:
            return f"[Document: {path} | Total Pages: {total_pages}] No matches found for query: '{query_str}'"

        header = f"[Document: {path} | Matches found for '{query_str}' ({len(matches)} results)]:\n"
        return header + "\n".join(matches)


def _escape_evidence(text: str) -> str:
    """Prevents document text from breaking the evidence wrapper."""
    return text.replace("</untrusted_course_evidence>", "&lt;/untrusted_course_evidence&gt;")


class SearchCourseMaterialTool(Tool):
    name = "search_course_material"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Search indexed course materials using full-text search (SQLite FTS5). "
        "Returns relevant passages with document title, page number, and evidence text. "
        "Does not accept arbitrary filesystem paths."
    )
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Configured course identifier (e.g. 'sample-course').",
            },
            "query": {
                "type": "string",
                "description": "Keyword or topic to search for within course documents.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of passages to return (default: 5, max: 20).",
            },
        },
        "required": ["course_id", "query"],
    }

    def __init__(self, indexer: Any):
        self.indexer = indexer

    def run(self, *, course_id: str, query: str, top_k: int = 5) -> str:
        if not course_id or not course_id.strip():
            raise ToolError("'course_id' is required and cannot be blank.")
        if not query or not query.strip():
            raise ToolError("'query' is required and cannot be blank.")

        top_k_val = int(top_k)
        if top_k_val < 1 or top_k_val > 20:
            raise ToolError("'top_k' must be an integer between 1 and 20.")

        results = self.indexer.search(course_id=course_id.strip(), query=query.strip(), top_k=top_k_val)

        if not results:
            return f"No evidence found in course '{course_id}' for query: '{query}'."

        blocks = []
        for res in results:
            blocks.append(
                f"<untrusted_course_evidence>\n"
                f"Document: {res.document_title} (ID: {res.document_id})\n"
                f"Page: {res.page_number}\n"
                f"Text:\n{_escape_evidence(res.snippet)}\n"
                f"</untrusted_course_evidence>"
            )

        header = f"Found {len(results)} relevant passage(s) in course '{course_id}':\n\n"
        return header + "\n\n".join(blocks)


class ReadCoursePageTool(Tool):
    name = "read_course_page"
    permission_tier = PermissionTier.REGULAR
    description = (
        "Read the full context of a specific page from an indexed course document. "
        "Useful when a search result requires inspecting surrounding context."
    )
    parameters = {
        "type": "object",
        "properties": {
            "course_id": {
                "type": "string",
                "description": "Configured course identifier.",
            },
            "document_id": {
                "type": "string",
                "description": "Identifier of the document to inspect (e.g. 'syllabus').",
            },
            "page": {
                "type": "integer",
                "description": "1-based page number to read.",
            },
        },
        "required": ["course_id", "document_id", "page"],
    }

    def __init__(self, indexer: Any):
        self.indexer = indexer

    def run(self, *, course_id: str, document_id: str, page: int) -> str:
        if not course_id or not course_id.strip():
            raise ToolError("'course_id' is required and cannot be blank.")
        if not document_id or not document_id.strip():
            raise ToolError("'document_id' is required and cannot be blank.")

        page_num = int(page)
        if page_num < 1:
            raise ToolError("'page' must be a positive integer (1-based).")

        text = self.indexer.get_page_text(
            course_id=course_id.strip(),
            document_id=document_id.strip(),
            page_number=page_num,
        )

        if not text:
            return f"No text found for course '{course_id}', document '{document_id}', page {page_num}."

        return (
            f"<untrusted_course_evidence>\n"
            f"Document ID: {document_id} | Page: {page_num}\n"
            f"Text:\n{_escape_evidence(text)}\n"
            f"</untrusted_course_evidence>"
        )
