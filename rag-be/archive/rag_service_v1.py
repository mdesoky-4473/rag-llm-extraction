"""
RAG Service — Backend API for LLM Retrieval
=============================================
Run with: python rag_service.py

Endpoints:
    POST /ask          — Ask a question, get an AI-grounded answer
    POST /reindex      — Re-pull documents from Google Drive and rebuild index
    GET  /health       — Health check
    GET  /stats        — Index statistics
"""

import os
import io
import pickle
import numpy as np
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import anthropic
import pdfplumber

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


# =============================================================================
# CONFIGURATION — Update these for your environment
# =============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "YOUR_KEY_HERE")
GDRIVE_CREDENTIALS_PATH = os.environ.get("GDRIVE_CREDENTIALS_PATH", "YOUR_JSON_CREDENTIALS_PATH_HERE")
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "YOUR_FOLDER_ID_HERE")
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
#FOLDER_PATH = os.environ.get("FOLDER_PATH", r".\documents")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.05
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.1


# =============================================================================
# PYDANTIC MODELS — Request/Response schemas
# =============================================================================

class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    strict_mode: bool = True   # True = no interpretation, False = allow it

class Source(BaseModel):
    document: str
    score: float
    text_preview: str

class QueryResponse(BaseModel):
    answer: str
    confidence: str
    top_score: float
    sources: List[Source]

class IndexStats(BaseModel):
    total_documents: int
    total_chunks: int
    vector_dimensions: int
    document_names: List[str]

class ReindexResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int


# =============================================================================
# RAG PIPELINE
# =============================================================================

class RAGPipeline:
    """Encapsulates the entire RAG pipeline as a single class."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.chunks = []
        self.chunk_vectors = None
        self.documents = []
        self.llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.gdrive_service = None
        self.is_indexed = False

    # --- Google Drive Connection ---
    def connect_gdrive(self):
        """Authenticate with Google Drive."""
        creds = None
        token_path = os.path.join(
            os.path.dirname(GDRIVE_CREDENTIALS_PATH), "token.pickle"
        )

        if os.path.exists(token_path):
            with open(token_path, "rb") as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    GDRIVE_CREDENTIALS_PATH, GDRIVE_SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(token_path, "wb") as f:
                pickle.dump(creds, f)

        self.gdrive_service = build("drive", "v3", credentials=creds)
        print("✅ Connected to Google Drive")

    # --- Document Loading ---
    def load_documents_from_gdrive(self, folder_id: str) -> List[dict]:
        """Load documents from a Google Drive folder."""
        if not self.gdrive_service:
            self.connect_gdrive()

        results = self.gdrive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name, mimeType)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="allDrives",
        ).execute()

        files = results.get("files", [])
        documents = []

        for file in files:
            name = file["name"]
            mime = file["mimeType"]
            file_id = file["id"]

            try:
                if mime == "application/vnd.google-apps.document":
                    content = self.gdrive_service.files().export(
                        fileId=file_id, mimeType="text/plain"
                    ).execute()
                    text = content.decode("utf-8")

                elif mime == "application/pdf":
                    content = self.gdrive_service.files().get_media(
                        fileId=file_id, supportsAllDrives=True
                    ).execute()
                    pdf = pdfplumber.open(io.BytesIO(content))
                    text = "\n".join(
                        p.extract_text() for p in pdf.pages if p.extract_text()
                    )
                    pdf.close()

                elif mime == "text/plain":
                    content = self.gdrive_service.files().get_media(
                        fileId=file_id, supportsAllDrives=True
                    ).execute()
                    text = content.decode("utf-8")

                else:
                    print(f"  Skipping {name} ({mime})")
                    continue

                if len(text.strip()) > 10:
                    documents.append({"id": name, "text": text, "source": name})
                    print(f"  ✅ Loaded: {name} ({len(text)} chars)")

            except Exception as e:
                print(f"  ❌ Error loading {name}: {e}")

        return documents

    def load_documents_from_folder(self, folder_path: str) -> List[dict]:
        """Load documents from a local folder (fallback)."""
        documents = []
        for filename in sorted(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, filename)
            try:
                if filename.endswith(".txt"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()
                elif filename.endswith(".pdf"):
                    with pdfplumber.open(filepath) as pdf:
                        text = "\n".join(
                            p.extract_text() for p in pdf.pages if p.extract_text()
                        )
                else:
                    continue

                if len(text.strip()) > 10:
                    documents.append({"id": filename, "text": text, "source": filename})
                    print(f"  ✅ Loaded: {filename} ({len(text)} chars)")

            except Exception as e:
                print(f"  ❌ Error loading {filename}: {e}")

        return documents

    # --- Chunking ---
    def chunk_document(self, doc: dict) -> List[dict]:
        """Split a document into overlapping chunks."""
        text = doc["text"]
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            if end < len(text):
                last_period = text[start:end].rfind(". ")
                if last_period > CHUNK_SIZE * 0.5:
                    end = start + last_period + 2

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"{doc['id']}_chunk_{idx}",
                    "text": chunk_text,
                    "source": doc["id"],
                    "chunk_index": idx,
                })
                idx += 1
            start = end - CHUNK_OVERLAP

        return chunks

    # --- Indexing ---
    def index(self, folder_id: str = None, local_path: str = None):
        """Load documents, chunk, embed, and store vectors."""
        if folder_id:
            self.documents = self.load_documents_from_gdrive(folder_id)
        elif local_path:
            self.documents = self.load_documents_from_folder(local_path)
        else:
            raise ValueError("Provide folder_id or local_path")

        self.chunks = []
        for doc in self.documents:
            self.chunks.extend(self.chunk_document(doc))

        if not self.chunks:
            raise ValueError("No chunks created — check your documents")

        chunk_texts = [c["text"] for c in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
        self.is_indexed = True

        print(f"✅ Indexed {len(self.chunks)} chunks from {len(self.documents)} documents")

    # --- Retrieval ---
    def retrieve(self, query: str, top_k: int = TOP_K) -> List[dict]:
        """Embed query and find most similar chunks."""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [
            {"chunk": self.chunks[i], "score": float(similarities[i])}
            for i in top_indices
        ]

    # --- Prompt Building ---
# --- Prompt Building ---
    def build_prompt(self, query: str, results: List[dict], strict_mode: bool = True) -> str:
        """Build LLM prompt with retrieved context and guardrails."""
        context_parts = []
        for r in results:
            context_parts.append(
                f"[Source: {r['chunk']['source']} | Relevance: {r['score']:.2f}]\n"
                f"{r['chunk']['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        if strict_mode:
            rules = """RULES:
- Answer ONLY using the provided context below.
- If the context does not contain the answer, say "I don't have that information."
- Do NOT interpret, extrapolate, or infer beyond what is explicitly stated.
- Cite which source document your answer comes from."""
        else:
            rules = """RULES:
- Answer using the provided context below.
- If the context does not contain the exact answer, provide your best interpretation and clearly state it is an interpretation.
- Cite which source document your answer comes from."""

        return f"""You are an expert knowledge assistant.

{rules}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    # --- LLM Call ---
    def call_llm(self, prompt: str) -> str:
        """Call Claude with the augmented prompt."""
        response = self.llm_client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    # --- Full Pipeline ---
    def answer(self, query: str, top_k: int = TOP_K, strict_mode: bool = True) -> dict:
        """Run the complete RAG pipeline."""
        if not self.is_indexed:
            raise RuntimeError("Index not built — call /reindex first")

        results = self.retrieve(query, top_k)
        top_score = results[0]["score"]

        # Confidence guardrail
        if top_score < CONFIDENCE_THRESHOLD:
            return {
                "answer": "I don't have information about that in the current documents.",
                "confidence": "LOW",
                "top_score": top_score,
                "sources": [],
            }

        prompt = self.build_prompt(query, results, strict_mode)
        answer = self.call_llm(prompt)

        confidence = "HIGH" if top_score > 0.3 else "MEDIUM"
        sources = [
            {
                "document": r["chunk"]["source"],
                "score": round(r["score"], 3),
                "text_preview": r["chunk"]["text"][:500],
            }
            for r in results
        ]

        return {
            "answer": answer,
            "confidence": confidence,
            "top_score": round(top_score, 3),
            "sources": sources,
        }


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="RAG Service",
    description="Retrieval-Augmented Generation API powered by Claude",
    version="1.0.0",
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: specify exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = RAGPipeline()


@app.post("/ask", response_model=QueryResponse)
def ask(query: QueryRequest):
    """Ask a question — retrieves relevant context and generates an answer."""
    if not pipeline.is_indexed:
        raise HTTPException(status_code=503, detail="Index not built. Call POST /reindex first.")

    try:
        result = pipeline.answer(query.question, query.top_k, query.strict_mode)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex", response_model=ReindexResponse)
def reindex():
    """Re-pull documents from Google Drive and rebuild the index."""
    try:
        pipeline.index(folder_id=GDRIVE_FOLDER_ID)
        return ReindexResponse(
            status="success",
            documents_loaded=len(pipeline.documents),
            chunks_created=len(pipeline.chunks),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "healthy",
        "indexed": pipeline.is_indexed,
    }


@app.get("/stats", response_model=IndexStats)
def stats():
    """Get current index statistics."""
    if not pipeline.is_indexed:
        raise HTTPException(status_code=503, detail="Index not built.")

    return IndexStats(
        total_documents=len(pipeline.documents),
        total_chunks=len(pipeline.chunks),
        vector_dimensions=pipeline.chunk_vectors.shape[1],
        document_names=[d["id"] for d in pipeline.documents],
    )


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    # Try to build index on startup
    print("🚀 Starting RAG Service...")
    try:
        pipeline.index(folder_id=GDRIVE_FOLDER_ID)
    except Exception as e:
        print(f"⚠️  Could not auto-index: {e}")
        print("   Call POST /reindex after starting to build the index.")

    uvicorn.run(app, host="0.0.0.0", port=8000)
