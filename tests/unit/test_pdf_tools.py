from pathlib import Path
import pytest
import fitz

from agent.tools.base import ToolError
from agent.tools.pdf.reader import ReadPdfTool, SearchPdfTool


@pytest.fixture
def sample_pdf_workspace(tmp_path: Path) -> Path:
    """Creates a temporary workspace with a 3-page test PDF."""
    pdf_path = tmp_path / "course_syllabus.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Sample Course 2026\nInstructor: Prof. Dr. Example\nMilestone 1: Foundations")

    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Grading and Deadlines Policy:\nLate submissions receive a 10% penalty per day.\nReports are due Wednesdays at 23:59.")

    # Page 3
    page3 = doc.new_page()
    page3.insert_text((50, 72), "Project Deliverables:\nPhase 3 involves Sub-Agents and Observability.\nDemos in July.")

    doc.save(str(pdf_path))
    doc.close()

    # Also create a non-pdf file
    (tmp_path / "notes.txt").write_text("not a pdf", encoding="utf-8")
    return tmp_path


def test_read_pdf_basic(sample_pdf_workspace: Path):
    tool = ReadPdfTool(sample_pdf_workspace)
    output = tool.run(path="course_syllabus.pdf", start_page=1, max_pages=1)

    assert "Total Pages: 3" in output
    assert "--- Page 1 ---" in output
    assert "Sample Course 2026" in output
    assert "Page 2" not in output


def test_read_pdf_page_range(sample_pdf_workspace: Path):
    tool = ReadPdfTool(sample_pdf_workspace)
    output = tool.run(path="course_syllabus.pdf", start_page=2, max_pages=2)

    assert "Total Pages: 3" in output
    assert "Showing Pages: 2-3" in output
    assert "Late submissions receive a 10% penalty" in output
    assert "Phase 3 involves Sub-Agents" in output
    assert "Page 1" not in output


def test_read_pdf_invalid_file(sample_pdf_workspace: Path):
    tool = ReadPdfTool(sample_pdf_workspace)
    with pytest.raises(ToolError, match="does not exist"):
        tool.run(path="non_existent.pdf")

    with pytest.raises(ToolError, match="not a PDF"):
        tool.run(path="notes.txt")


def test_read_pdf_sandbox_violation(sample_pdf_workspace: Path):
    tool = ReadPdfTool(sample_pdf_workspace)
    with pytest.raises(ToolError, match="outside the workspace root"):
        tool.run(path="../../secret.pdf")


def test_search_pdf_match(sample_pdf_workspace: Path):
    tool = SearchPdfTool(sample_pdf_workspace)
    output = tool.run(path="course_syllabus.pdf", query="late submissions")

    assert "Matches found for 'late submissions'" in output
    assert "Page 2" in output
    assert "10% penalty" in output


def test_search_pdf_case_insensitive(sample_pdf_workspace: Path):
    tool = SearchPdfTool(sample_pdf_workspace)
    output = tool.run(path="course_syllabus.pdf", query="OBSERVABILITY")

    assert "Page 3" in output
    assert "Phase 3 involves Sub-Agents and Observability" in output


def test_search_pdf_no_match(sample_pdf_workspace: Path):
    tool = SearchPdfTool(sample_pdf_workspace)
    output = tool.run(path="course_syllabus.pdf", query="Quantum Physics")

    assert "No matches found" in output


def test_search_pdf_empty_query(sample_pdf_workspace: Path):
    tool = SearchPdfTool(sample_pdf_workspace)
    with pytest.raises(ToolError, match="cannot be empty"):
        tool.run(path="course_syllabus.pdf", query="   ")
