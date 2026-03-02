# RAG Chat Application

A Python-based Retrieval Augmented Generation (RAG) chat application that lets you upload documents and ask questions about them. Built with LangChain, Streamlit, Azure OpenAI (via APIM), and Azure AI Search.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                      │
│  Chat interface + sidebar file uploader                     │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│  RAG Chain (rag/chain.py)                                   │
│                                                             │
│  Stage 1: Condense follow-up → standalone question          │
│  Stage 2: Retrieve relevant docs from Azure AI Search       │
│  Stage 3: Format docs as context string                     │
│  Stage 4: Generate answer via RAG prompt + LLM              │
│  Stage 5: Return answer + source documents                  │
│                                                             │
│  Wrapped in RunnableWithMessageHistory for session memory    │
└──────┬───────────────────┬──────────────────────────────────┘
       │                   │
       ▼                   ▼
┌──────────────┐   ┌──────────────────┐
│ Azure OpenAI │   │ Azure AI Search  │
│ (via APIM)   │   │ (direct)         │
│              │   │                  │
│ - Chat LLM   │   │ - Vector store   │
│ - Embeddings │   │ - Retriever      │
└──────────────┘   └──────────────────┘
```

Two separate API calls are made:

- **Azure OpenAI** — LLM and embedding requests route through **Azure API Management (APIM)**, which acts as a gateway proxy. Authentication uses **OAuth2 client credentials flow** (Azure AD / Entra ID) — a Bearer token is obtained via `ClientSecretCredential` and sent in the `Authorization` header on every request. Secrets can optionally be stored in **Azure Key Vault**.
- **Azure AI Search** — Vector store reads and writes go **directly** to the search service using its own endpoint and admin key. APIM is not involved.

---

## Project Structure

```
langchainrag/
├── .env.example              # Template for required env vars
├── .gitignore
├── requirements.txt          # Production dependencies
├── requirements-dev.txt      # Adds pytest for development
├── Dockerfile                # Production container (non-root)
├── docker-compose.yml        # One-command local container run
├── config.py                 # Centralised settings from env vars
├── app.py                    # Streamlit entry point
├── rag/
│   ├── __init__.py
│   ├── llm.py                # AzureChatOpenAI + embeddings factories
│   ├── vectorstore.py        # Azure AI Search vector store + retriever
│   ├── chain.py              # Two-stage LCEL RAG chain with chat history
│   └── prompts.py            # Prompt templates
├── ingestion/
│   ├── __init__.py
│   └── ingest.py             # CLI script to load docs into Azure AI Search
└── tests/
    ├── __init__.py
    ├── test_config.py
    ├── test_chain.py
    └── test_ingestion.py
```

---

## Setup

### 1. Install dependencies

```bash
# Production only
pip install -r requirements.txt

# With test tooling
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your Azure credentials:

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_TENANT_ID` | Yes | Azure AD / Entra ID tenant ID |
| `AZURE_CLIENT_ID` | Yes | App registration client ID |
| `AZURE_CLIENT_SECRET` | Yes | App registration client secret |
| `APIM_ENDPOINT` | Yes | APIM gateway URL (e.g. `https://your-apim.azure-api.net`) |
| `APIM_SCOPE` | Yes | OAuth2 scope for APIM (e.g. `api://your-app-id/.default`) |
| `AZURE_KEYVAULT_NAME` | No | Key Vault name (enables secret fetching from vault) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Yes | Chat model deployment name (e.g. `gpt-4o`) |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Yes | Embedding deployment name (e.g. `text-embedding-ada-002`) |
| `AZURE_OPENAI_API_VERSION` | No | Defaults to `2024-06-01` |
| `AZURE_SEARCH_ENDPOINT` | Yes | Azure AI Search service URL |
| `AZURE_SEARCH_KEY` | Yes | Azure AI Search admin key (can be overridden from Key Vault) |
| `AZURE_SEARCH_INDEX_NAME` | No | Defaults to `rag-index` |

### 3. Ingest documents

```bash
# Ingest a directory of documents
python -m ingestion.ingest path/to/docs/

# Or a single file
python -m ingestion.ingest document.pdf
```

Supported file types: `.pdf`, `.txt`, `.md`

### 4. Run the app

```bash
streamlit run app.py
```

Opens at http://localhost:8501. You can also upload documents via the sidebar.

---

## Docker

```bash
# Build and run with docker-compose
docker compose up --build

# Or manually
docker build -t rag-chat .
docker run -p 8501:8501 --env-file .env rag-chat
```

The container runs as a non-root user (`appuser`) and includes a healthcheck on `/_stcore/health`.

---

## How It Works

### Document Ingestion

The ingestion pipeline (`ingestion/ingest.py`) processes documents in three steps:

1. **Load** — `PyPDFLoader` extracts text from PDFs. `TextLoader` reads `.txt` and `.md` files with UTF-8 encoding. Both return LangChain `Document` objects containing `page_content` (the text) and `metadata` (source file path, page number for PDFs, etc.).

2. **Split** — `RecursiveCharacterTextSplitter` breaks documents into 1000-character chunks with 200-character overlap. "Recursive" means it tries to split on paragraph breaks first, then sentences, then words, preserving natural text boundaries. The overlap ensures context isn't lost at chunk edges — if a key fact spans two chunks, the overlap catches it.

3. **Index** — Chunks are passed to `AzureSearch.add_documents()`, which calls the embedding model to generate a vector for each chunk, then upserts both the vector and the original text into the Azure AI Search index.

### The RAG Chain

The core pipeline (`rag/chain.py`) uses LangChain Expression Language (LCEL) to compose a multi-stage chain. LCEL lets you connect components with the `|` pipe operator — the output of each stage flows into the next, similar to Unix pipes.

#### LCEL building blocks used

- **`RunnablePassthrough.assign(key=fn)`** — Passes the entire input dict through unchanged, but adds a new key computed by `fn`. This is how state accumulates through the pipeline: each stage adds a new field (standalone question, retrieved docs, context string, answer) without losing previous fields.

- **`RunnableLambda(fn)`** — Wraps a plain Python function as a chainable runnable. Used at the end of the pipeline to filter the accumulated dict down to just `answer` and `source_documents`.

- **`StrOutputParser()`** — Takes the LLM's `AIMessage` output and extracts the text content as a plain string.

- **`RunnableWithMessageHistory`** — Wraps the chain to automatically load/save conversation history per session. Before each invocation it injects past messages into the `chat_history` placeholder, and after each invocation it saves the new exchange.

- **`ChatPromptTemplate.from_messages()`** — Defines a structured prompt with typed message roles (system, human) and variable placeholders (`{context}`, `{question}`).

- **`MessagesPlaceholder("chat_history")`** — A dynamic slot in the prompt template that gets replaced with the actual list of past messages at runtime.

#### Pipeline stages

The chain runs in five stages. At each stage the data is a Python dict that accumulates new keys:

```
Input: {"question": "What does the report say about Q3?"}
```

**Stage 1 — Condense question**
```python
RunnablePassthrough.assign(standalone_question=condense_question)
```
If there's chat history, the follow-up question is rewritten into a standalone query using the condense prompt + LLM. For example, if the previous exchange was about a financial report and the user asks "What about Q3?", this rewrites it to "What does the financial report say about Q3 performance?". Without this step, the retriever would search for "What about Q3?" and get irrelevant results. If there's no history, the question passes through unchanged.

**Stage 2 — Retrieve documents**
```python
RunnablePassthrough.assign(
    source_documents=lambda x: retriever.invoke(x["standalone_question"])
)
```
The standalone question is embedded and used to search Azure AI Search for the top-k (default 4) most similar document chunks. Returns a list of `Document` objects.

**Stage 3 — Format context**
```python
RunnablePassthrough.assign(
    context=lambda x: format_docs(x["source_documents"]),
    question=lambda x: x["standalone_question"],
)
```
The retrieved documents are joined into a single string separated by double newlines. This string is what gets injected into the `{context}` placeholder in the RAG prompt.

**Stage 4 — Generate answer**
```python
RunnablePassthrough.assign(answer=answer_chain)
```
The RAG prompt template is filled in with the context and question, sent to the LLM, and the response is parsed to a string. The `answer_chain` is itself an LCEL chain: `RAG_PROMPT | llm | StrOutputParser()`.

**Stage 5 — Clean output**
```python
RunnableLambda(lambda x: {"answer": x["answer"], "source_documents": x["source_documents"]})
```
Strips intermediate keys (standalone_question, context, etc.) and returns only the answer and source documents.

#### Chat history management

`RunnableWithMessageHistory` wraps the chain and manages conversation memory:

```python
RunnableWithMessageHistory(
    rag_chain,
    _get_session_history,          # returns InMemoryChatMessageHistory for a session_id
    input_messages_key="question",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

- `input_messages_key` tells it which key in the input dict is the user's message (saved as a `HumanMessage`).
- `history_messages_key` tells it where to inject past messages into the chain's input (matching the `MessagesPlaceholder` name in the prompt templates).
- `output_messages_key` tells it which key in the output dict is the AI's response (saved as an `AIMessage`).

History is stored in-memory using `InMemoryChatMessageHistory`, keyed by `session_id`. Each Streamlit session gets a unique UUID. History is lost on process restart — this is intentional for a single-process app.

### LLM + Embeddings (APIM Configuration)

Both `AzureChatOpenAI` and `AzureOpenAIEmbeddings` are configured in `rag/llm.py` with:

- `azure_endpoint` — The APIM gateway URL (not the Azure OpenAI resource directly)
- `api_key` — Set to `"placeholder"` (LangChain requires a non-empty value for client validation)
- `azure_deployment` — The model deployment name
- `default_headers` — Adds `{"Authorization": "Bearer <token>"}` to every HTTP request

Authentication uses **OAuth2 client credentials flow** via `ClientSecretCredential` from the `azure-identity` library. On each call to `get_llm()` or `get_embeddings()`, a fresh Bearer token is obtained via `credential.get_token()`, which automatically caches and refreshes tokens. Since the chain is cached in `st.session_state`, the token is set at chain creation time — tokens last ~1 hour and sessions are typically short.

Sensitive secrets (like the Azure Search key) can optionally be stored in **Azure Key Vault**. If `AZURE_KEYVAULT_NAME` is set, the app uses the same `ClientSecretCredential` to fetch secrets from the vault, falling back to environment variables for local development.

### Vector Store (Azure AI Search)

`AzureSearch` in `rag/vectorstore.py` connects directly to Azure AI Search:

```python
AzureSearch(
    azure_search_endpoint=...,
    azure_search_key=...,
    index_name=...,
    embedding_function=embeddings.embed_query,  # callable, not object
)
```

An important subtlety: `AzureSearch` expects `embedding_function` to be a **callable** (a function that takes a string and returns a vector), not an `Embeddings` object. That's why we pass `embeddings.embed_query` — the bound method — rather than the embeddings instance itself.

The retriever is created with `.as_retriever(search_kwargs={"k": 4})`, returning the 4 most relevant chunks per query.

### Streamlit UI

The UI (`app.py`) has three sections:

- **Sidebar** — File uploader for ingesting new documents (supports multi-file upload). Uploaded files are written to a temp file (cleaned up after ingestion), loaded, chunked, and indexed. Also has a "Clear Chat History" button.
- **Chat display** — Renders the conversation using `st.chat_message`. Each assistant message includes an expandable "Sources" section showing which document chunks were used, with the source name and a 200-character preview snippet.
- **Chat input** — Sends the user's question through the RAG chain and displays the answer with sources.

The chain is cached in `st.session_state` to avoid rebuilding it on every Streamlit rerun. If Azure credentials are missing, the UI loads in a degraded state with disabled inputs and a configuration warning.

### Prompt Templates

Two prompt templates are defined in `rag/prompts.py`:

**CONDENSE_PROMPT** — Takes the chat history and a follow-up question, and rewrites the question to be standalone. This is critical for retrieval: without it, follow-up questions like "Tell me more about that" would produce poor search results because the retriever has no context about what "that" refers to.

**RAG_PROMPT** — Takes the retrieved context and the question, and instructs the LLM to answer based only on the provided context. If the context doesn't contain enough information, it says so rather than hallucinating.

Both templates use `MessagesPlaceholder("chat_history")` to inject conversation history between the system and human messages.

---

## Configuration

Settings are managed through a frozen `Settings` dataclass in `config.py`:

- Environment variables are loaded via `python-dotenv` from `.env`
- Required variables raise `EnvironmentError` if missing
- Optional variables (`AZURE_OPENAI_API_VERSION`, `AZURE_SEARCH_INDEX_NAME`) have sensible defaults
- Settings are cached as a singleton via `@lru_cache` — created on first access, not at import time
- `repr()` redacts API keys and subscription keys so they don't leak into logs or tracebacks

---

## Tests

```bash
pytest tests/ -v
```

The test suite covers:

- **test_config.py** — Settings loading with all vars, missing required vars, defaults, and secret redaction in repr
- **test_chain.py** — Document formatting, session history creation, clearing, and idempotent clear of nonexistent sessions
- **test_ingestion.py** — Supported extensions, file collection (single file, unsupported file, recursive directory), document loading for txt/md, and rejection of unsupported types

Tests run without Azure credentials — the config module uses lazy initialisation so importing chain/ingestion modules doesn't trigger env-var validation.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Two-stage chain (condense + RAG) | Follow-up questions like "Tell me more" need rewriting for retrieval to work |
| `InMemoryChatMessageHistory` | Simplest for single-process Streamlit; swap to Redis/Cosmos later if needed |
| OAuth2 Bearer token via `ClientSecretCredential` | Production uses Entra ID client credentials; `api_key` is a placeholder for LangChain validation |
| Key Vault for secret storage | Secrets can be fetched from vault at startup; env vars used as fallback for local dev |
| `embedding_function=embeddings.embed_query` | `AzureSearch` expects a callable, not an embeddings object |
| Chain cached in `st.session_state` | Avoids rebuilding on every Streamlit rerun |
| `RecursiveCharacterTextSplitter(1000, 200)` | Respects paragraph/sentence boundaries; well-tested default for RAG |
| Lazy settings via `get_settings()` | Allows importing modules without env vars (for tests and UI preview) |
| Non-root Docker user | Basic container hygiene — limits blast radius if compromised |
