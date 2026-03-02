"""CLI script to ingest documents into Azure AI Search.

WHAT IS INGESTION?
  Ingestion is the process of taking your raw documents (PDFs, text files, etc.)
  and making them searchable by the RAG pipeline. The process is:

  1. LOAD — Read the document file and extract its text content.
     - PDFs are loaded with PyPDFLoader (extracts text per page)
     - TXT/MD files are loaded with TextLoader (reads the whole file)
     - Each loader returns LangChain Document objects containing the text
       and metadata (source file path, page number, etc.)

  2. SPLIT — Break the text into smaller chunks.
     - We use RecursiveCharacterTextSplitter with 1000-char chunks and 200-char
       overlap. "Recursive" means it tries to split on natural boundaries first
       (paragraph breaks, then sentences, then words) rather than cutting mid-word.
     - The 200-char overlap ensures that if a key fact spans two chunks, it
       appears in both — so the retriever can still find it.
     - Why chunk at all? LLMs have limited context windows, and smaller chunks
       give more precise retrieval. A whole 50-page PDF as one chunk would be
       too large and too vague for similarity search.

  3. INDEX — Store the chunks in Azure AI Search.
     - Each chunk is converted to a vector (embedding) by the embedding model
     - The vector + original text + metadata are stored in the search index
     - Later, the RAG pipeline searches this index to find relevant chunks

USAGE:
    # Ingest all supported files in a directory (recursive)
    python -m ingestion.ingest path/to/docs/

    # Ingest a single file
    python -m ingestion.ingest document.pdf
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.vectorstore import get_vector_store

# File extensions we know how to process. Anything else is silently skipped
# (or raises ValueError if passed directly to load_document).
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# The text splitter used by both CLI ingestion and the Streamlit UI's upload.
# 1000-char chunks with 200-char overlap is a well-tested default for RAG.
# - chunk_size=1000: each chunk will be roughly 1000 characters
# - chunk_overlap=200: consecutive chunks share 200 characters at their edges
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def load_document(path: Path):
    """Load a single file into a list of LangChain Documents.

    Each Document has:
      - page_content: the text content of the document (or page for PDFs)
      - metadata: a dict with info like source file path, page number, etc.

    Raises ValueError for unsupported file types.

    Note: PyPDFLoader returns one Document per page. TextLoader returns one
    Document for the entire file.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        # PyPDFLoader extracts text from each page of the PDF.
        # str(path) is needed because PyPDFLoader expects a string, not a Path.
        return PyPDFLoader(str(path)).load()
    if ext in (".txt", ".md"):
        # TextLoader reads the entire file as a single Document.
        # encoding="utf-8" is explicit to avoid platform-specific defaults.
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}")


def collect_files(path: Path) -> list[Path]:
    """Recursively collect all supported files from a path.

    If path is a single file: returns it in a list (if supported) or [].
    If path is a directory: recursively finds all supported files using rglob.

    Examples:
        collect_files(Path("report.pdf"))  → [Path("report.pdf")]
        collect_files(Path("report.docx")) → []  (unsupported)
        collect_files(Path("docs/"))       → [Path("docs/a.pdf"), Path("docs/sub/b.txt"), ...]
    """
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        return []
    # rglob("*") recursively yields all files in all subdirectories
    return [
        f
        for f in path.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def ingest(path: str) -> int:
    """Load, chunk, and index documents into Azure AI Search.

    This is the main ingestion pipeline. It:
    1. Finds all supported files at the given path
    2. Loads them into Document objects
    3. Splits them into chunks using the text splitter
    4. Indexes all chunks into Azure AI Search (creates embeddings + stores)

    Args:
        path: A file path or directory path (string).

    Returns:
        0 on success, 1 on error (for use as a CLI exit code).
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

    # Step 1: Load all files into Document objects
    all_docs = []
    for file in files:
        print(f"  Loading {file.name}...")
        docs = load_document(file)
        all_docs.extend(docs)

    # Step 2: Split into chunks
    print(f"Loaded {len(all_docs)} page(s), splitting into chunks...")
    chunks = SPLITTER.split_documents(all_docs)
    print(f"Created {len(chunks)} chunk(s)")

    # Step 3: Index into Azure AI Search.
    # add_documents() calls the embedding model for each chunk to generate vectors,
    # then upserts the vectors + text into the search index. This is the most
    # expensive step — it makes API calls for every chunk.
    print("Indexing into Azure AI Search...")
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print("Done!")
    return 0


def main():
    """CLI entry point — called when running `python -m ingestion.ingest <path>`."""
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.ingest <path>")
        sys.exit(1)
    sys.exit(ingest(sys.argv[1]))


if __name__ == "__main__":
    main()
