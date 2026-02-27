"""Tests for the ingestion module — file loading, collection, and extension filtering."""

import tempfile
from pathlib import Path

import pytest

from ingestion.ingest import SUPPORTED_EXTENSIONS, collect_files, load_document


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".docx" not in SUPPORTED_EXTENSIONS


def test_collect_files_single_txt():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test")
        path = Path(f.name)
    files = collect_files(path)
    assert len(files) == 1
    assert files[0] == path
    path.unlink()


def test_collect_files_unsupported():
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"test")
        path = Path(f.name)
    files = collect_files(path)
    assert len(files) == 0
    path.unlink()


def test_collect_files_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "doc.txt").write_text("hello")
        (root / "notes.md").write_text("# Notes")
        (root / "ignore.docx").write_text("skip")
        subdir = root / "sub"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("nested")

        files = collect_files(root)
        extensions = {f.suffix for f in files}
        assert extensions == {".txt", ".md"}
        assert len(files) == 3


def test_load_document_txt():
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write("Hello world content")
        path = Path(f.name)
    docs = load_document(path)
    assert len(docs) >= 1
    assert "Hello world content" in docs[0].page_content
    path.unlink()


def test_load_document_md():
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
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"test")
        path = Path(f.name)
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)
    path.unlink()
