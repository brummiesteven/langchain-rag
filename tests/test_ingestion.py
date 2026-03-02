"""Tests for the ingestion module — file loading, collection, and extension filtering.

These tests verify the document ingestion pipeline WITHOUT making any Azure calls.
They use temporary files on disk to test:

1. SUPPORTED_EXTENSIONS — which file types we accept
2. collect_files() — finding supported files in a path (single file or directory)
3. load_document() — loading a file into LangChain Document objects

The actual Azure AI Search indexing (vector_store.add_documents) is NOT tested
here — that would require integration tests with a live search service.
"""

import tempfile
from pathlib import Path

import pytest

from ingestion.ingest import SUPPORTED_EXTENSIONS, collect_files, load_document


# ─── Extension filtering ─────────────────────────────────────────────────────

def test_supported_extensions():
    """Verify which file extensions are accepted and rejected.

    We support PDF, TXT, and MD. Common office formats like DOCX are NOT
    supported because we don't have a loader for them.
    """
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".docx" not in SUPPORTED_EXTENSIONS


# ─── collect_files tests ──────────────────────────────────────────────────────
# collect_files() takes a Path and returns a list of supported files.
# It handles both single files and directories (recursive).

def test_collect_files_single_txt():
    """A single supported file should return itself in a list."""
    # Create a temp .txt file and verify collect_files finds it
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test")
        path = Path(f.name)
    files = collect_files(path)
    assert len(files) == 1
    assert files[0] == path
    path.unlink()  # Clean up temp file


def test_collect_files_unsupported():
    """An unsupported file type should return an empty list (not raise)."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"test")
        path = Path(f.name)
    files = collect_files(path)
    assert len(files) == 0
    path.unlink()


def test_collect_files_directory():
    """A directory should be searched recursively for supported files.

    Creates a temp directory structure:
        tmpdir/
        ├── doc.txt        ← supported
        ├── notes.md       ← supported
        ├── ignore.docx    ← NOT supported (skipped)
        └── sub/
            └── nested.txt ← supported (found recursively)

    Should find 3 files total (doc.txt, notes.md, nested.txt).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "doc.txt").write_text("hello")
        (root / "notes.md").write_text("# Notes")
        (root / "ignore.docx").write_text("skip")  # Should be skipped
        subdir = root / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        files = collect_files(root)
        extensions = {f.suffix for f in files}
        # Only .txt and .md should be found (not .docx)
        assert extensions == {".txt", ".md"}
        assert len(files) == 3  # doc.txt, notes.md, nested.txt


# ─── load_document tests ─────────────────────────────────────────────────────
# load_document() reads a file and returns a list of LangChain Document objects.
# Each Document has .page_content (the text) and .metadata (source info).

def test_load_document_txt():
    """Loading a .txt file should return its content as a Document."""
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write("Hello world content")
        path = Path(f.name)
    docs = load_document(path)
    # TextLoader returns one Document for the entire file
    assert len(docs) >= 1
    assert "Hello world content" in docs[0].page_content
    path.unlink()


def test_load_document_md():
    """Loading a .md file should return its content (treated same as .txt)."""
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write("# Heading\n\nSome markdown content")
        path = Path(f.name)
    docs = load_document(path)
    assert len(docs) >= 1
    assert "markdown content" in docs[0].page_content
    path.unlink()


def test_load_document_unsupported():
    """Loading an unsupported file type should raise ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"test")
        path = Path(f.name)
    # Should raise with a message mentioning "Unsupported"
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)
    path.unlink()
