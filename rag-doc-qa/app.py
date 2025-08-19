import os
import io
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from openai import OpenAI

# ---------- .env yükle ----------
DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

# ---------- Config ----------
ENV_API_KEY    = (os.getenv("KEY") or "").strip()
MODEL   = (os.getenv("MODEL") or "model").strip()
CHROMA_DIR     = (os.getenv("CHROMA_DIR") or "./chroma").strip()
COLLECTION_NAME= (os.getenv("COLLECTION_NAME") or "docs").strip()

# Chroma telemetry sustur (log kirliliğini azaltır)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")

# ---------- FastAPI ----------
app = FastAPI(title="RAG Doc QA", version="1.0.0")

# ---------- Embedding (local) ----------
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# ---------- Chroma ----------
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedder
)

# ---------- Helpers ----------
def extract_text_from_pdf(file_like: io.BytesIO) -> str:
    try:
        reader = PdfReader(file_like)
        texts = []
        for p in reader.pages:
            texts.append(p.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF okunamadı: {e}")

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    chunks = []
    i, n = 0, len(text)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(text[i:j])
        i = j - overlap if j - overlap > i else j
    return [c.strip() for c in chunks if c and c.strip()]

# ---------- Schemas ----------
class IngestResponse(BaseModel):
    document_id: str
    chunks: int

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    # Manuel key için: body'de gönderirsen .env'i override eder
    api_key: str | None = None

class AskResponse(BaseModel):
    answer: str
    context: List[str]

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": collection.count()}

@app.get("/debug/env")
def debug_env():
    key_tail = ENV_API_KEY[-6:] if ENV_API_KEY else ""
    return {
        "has_env_key": bool(ENV_API_KEY),
        "starts_with_sk": ENV_API_KEY.startswith("sk-") if ENV_API_KEY else False,
        "tail": key_tail,
        "model": MODEL
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF yükleyin.")
    content = await file.read()
    text = extract_text_from_pdf(io.BytesIO(content))
    if not text.strip():
        raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı.")
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Parçalama sonrası metin bulunamadı.")

    doc_id = str(uuid.uuid4())
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{"doc_id": doc_id, "filename": file.filename, "chunk": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metas)
    return IngestResponse(document_id=doc_id, chunks=len(chunks))

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # --- API key seçimi: body'de varsa onu kullan; yoksa .env ---
    use_key = (req.api_key or ENV_API_KEY or "").strip()
    if not use_key or not use_key.startswith("sk-"):
        raise HTTPException(status_code=401, detail="OpenAI key bulunamadı veya geçersiz format.")

    oai = OpenAI(api_key=use_key)

    # --- Vektör arama ---
    try:
        results = collection.query(query_texts=[q], n_results=max(1, req.top_k))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vektör sorgusu hatası: {e}")

    docs = (results.get("documents") or [[]])[0]
    if not docs:
        return AskResponse(answer="Bağlamda yeterli bilgi yok.", context=[])

    context_text = "\n\n".join(docs)
    prompt = (
        "Aşağıdaki bağlama göre soruya yanıt ver. Sadece bağlamdaki bilgilere dayan. "
        "Emin değilsen 'Bağlamda yeterli bilgi yok' de.\n\n"
        f"# Bağlam:\n{context_text}\n\n# Soru: {q}\n\n# Cevap:"
    )

    # --- LLM çağrısı ---
    try:
        completion = oai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Yardımcı ve dürüst bir doküman asistanısın."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        msg = str(e)
        if "invalid_api_key" in msg or "401" in msg:
            raise HTTPException(status_code=401, detail="OpenAI key geçersiz (invalid_api_key).")
        raise HTTPException(status_code=500, detail=f"LLM yanıt üretemedi: {e}")

    return AskResponse(answer=answer, context=docs)
