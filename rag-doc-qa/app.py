import os
import io
import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# LLM: OpenAI
from openai import OpenAI

load_dotenv()

# --- OpenAI client ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    raise RuntimeError("OPENAI_API_KEY yok ya da hatalı. .env dosyanı kontrol et.")

oai = OpenAI(api_key=OPENAI_API_KEY)


# --- Config ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- App ---
app = FastAPI(title="RAG Doc QA", version="0.1.0")

# --- Embedding function (yerel) ---
# Not: chroma'nın hazır embedding wrapper'ını kullanalım; o da sentence-transformers'ı çağırır.
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# --- Chroma client & collection ---
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedder
)

# --- OpenAI client ---
oai = OpenAI()

# ---------- Yardımcı Fonksiyonlar ----------

def extract_text_from_pdf(file_like: io.BytesIO) -> str:
    try:
        reader = PdfReader(file_like)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF okunamadı: {e}")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
    # Basit karakter tabanlı chunking; daha sonra RecursiveSplitter'e geçebiliriz
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    # Boşları ayıkla
    return [c.strip() for c in chunks if c and c.strip()]


class IngestResponse(BaseModel):
    document_id: str
    chunks: int

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    context: List[str]

# ---------- Endpoints ----------

@app.get("/health")
def health():
    # Basit durum kontrolü
    count = collection.count()
    return {"status": "ok", "chunks_indexed": count}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF yükleyin.")

    content = await file.read()
    text = extract_text_from_pdf(io.BytesIO(content))

    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="PDF'den metin çıkarılamadı.")

    # Parçala
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Parçalama sonrası metin bulunamadı.")

    # Kimlikler ve metadata
    doc_id = str(uuid.uuid4())
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "filename": file.filename, "chunk": i} for i in range(len(chunks))]

    # Chroma'ya ekle
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    return IngestResponse(document_id=doc_id, chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    # Benzer parçaları getir
    results = collection.query(query_texts=[q], n_results=req.top_k)
    docs = results.get("documents", [[]])[0]

    # Bağlamı oluştur
    context_text = "\n\n".join(docs)
    prompt = (
        "Aşağıdaki bağlama göre soruya yanıt ver. Sadece bağlamda olan bilgilere dayan. "
        "Emin değilsen 'Bağlamda yeterli bilgi yok' de.\n\n"
        f"# Bağlam:\n{context_text}\n\n# Soru: {q}\n\n# Cevap:"
    )

    try:
        completion = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Yardımcı ve dürüst bir doküman asistanısın."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM yanıt üretemedi: {e}")

    return AskResponse(answer=answer, context=docs)