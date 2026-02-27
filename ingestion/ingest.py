"""CLI script to ingest documents into Azure AI Search.

Supports .pdf, .txt, and .md files. Documents are split into chunks
using RecursiveCharacterTextSplitter and indexed into the vector store.

Usage:
    python -m ingestion.ingest path/to/docs/
    python -m ingestion.ingest document.pdf
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vectorstore import get_vector_store

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# 1000-char chunks with 200-char overlap preserves paragraph/sentence
# boundaries while giving the retriever enough context per chunk.
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def load_document(path: Path):
    """Load a single file into a list of LangChain Documents.

    Raises ValueError for unsupported file types.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}")


def collect_files(path: Path) -> list[Path]:
    """Recursively collect all supported files from a path.

    If path is a single file, returns it in a list (if supported) or [].
    If path is a directory, recursively finds all supported files.
    """
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        return []
    return [
        f
        for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def ingest(path: str) -> int:
    """Load, chunk, and index documents into Azure AI Search.

    Returns 0 on success, 1 on error.
    """
    target = Path(path)
    if not target.exists():
        print(f"Error: {path} does not exist")
        return 1

    files = collect_files(target)
    if not files:
        print(f"No supported files found in {path}")
        return 1

    print(f"Found {len(files)} file(s) to ingest")

    all_docs = []
    for file in files:
        print(f"  Loading {file.name}...")
        docs = load_document(file)
        all_docs.extend(docs)

    print(f"Loaded {len(all_docs)} page(s), splitting into chunks...")
    chunks = SPLITTER.split_documents(all_docs)
    print(f"Created {len(chunks)} chunk(s)")

    print("Indexing into Azure AI Search...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print("Done!")
    return 0


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.ingest <path>")
        sys.exit(1)
    sys.exit(ingest(sys.argv[1]))


if __name__ == "__main__":
    main()
