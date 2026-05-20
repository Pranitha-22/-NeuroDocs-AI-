import io
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import fitz
from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ingestion.image_extraction import extract_images_from_pdf
from app.ingestion.pdf_loader import extract_pages_from_pdf
from app.rag.chunking import chunk_pages
from app.rag.qa_pipeline import generate_rag_result
from app.rag.vector_store import create_faiss_index, load_faiss_index, save_faiss_index
from app.voice.tts import text_to_speech

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_ROOT = DATA_DIR / "extracted_images"
DOCUMENT_ROOT = DATA_DIR / "documents"
WEB_DIR = PROJECT_ROOT / "web"
MAX_UPLOAD_BYTES = int(os.getenv("NEURODOCS_MAX_UPLOAD_MB", "50")) * 1024 * 1024
RATE_LIMIT_PER_MINUTE = int(os.getenv("NEURODOCS_RATE_LIMIT_PER_MINUTE", "60"))

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
DOCUMENT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroDocs AI API", version="0.1.0")
cors_origins = os.getenv("NEURODOCS_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

DOCUMENTS = {}
RATE_BUCKETS = {}


class AskRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = 10


class ExportRequest(BaseModel):
    question: str
    answer: str
    sources: list[dict] = []


def require_auth(request: Request, authorization: str | None = Header(default=None)):
    api_key = os.getenv("NEURODOCS_API_KEY")
    if not api_key:
        return

    expected = f"Bearer {api_key}"
    query_token = request.query_params.get("token")
    if authorization != expected and query_token != api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API token")


def rate_limit(request: Request):
    if RATE_LIMIT_PER_MINUTE <= 0:
        return

    now = time.time()
    client = request.client.host if request.client else "unknown"
    window_start = now - 60
    bucket = [stamp for stamp in RATE_BUCKETS.get(client, []) if stamp >= window_start]

    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")

    bucket.append(now)
    RATE_BUCKETS[client] = bucket


api_guard = [Depends(require_auth), Depends(rate_limit)]


def document_dir(document_id: str) -> Path:
    return DOCUMENT_ROOT / document_id


def metadata_path(document_id: str) -> Path:
    return document_dir(document_id) / "metadata.json"


def chunks_path(document_id: str) -> Path:
    return document_dir(document_id) / "chunks.json"


def index_path(document_id: str) -> Path:
    return document_dir(document_id) / "index.faiss"


def persisted_pdf_path(document_id: str) -> Path:
    return document_dir(document_id) / "source.pdf"


def save_json(path: Path, payload: dict | list):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_document_record(document_id: str, metadata: dict, chunks: list, index):
    return {
        "filename": metadata["filename"],
        "pages": metadata["pages"],
        "chunks": chunks,
        "index": index,
        "image_folder": metadata["image_folder"],
        "image_count": metadata["image_count"],
        "processing_time": metadata["processing_time"],
        "created_at": metadata.get("created_at", time.time()),
        "persisted": True,
    }


def persist_document(document_id: str, metadata: dict, chunks: list, index):
    doc_dir = document_dir(document_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    save_json(metadata_path(document_id), metadata)
    save_json(chunks_path(document_id), chunks)
    save_faiss_index(index, index_path(document_id))


def load_persisted_documents():
    loaded = 0
    for doc_dir in DOCUMENT_ROOT.iterdir():
        if not doc_dir.is_dir():
            continue

        document_id = doc_dir.name
        try:
            metadata = json.loads(metadata_path(document_id).read_text(encoding="utf-8"))
            chunks = json.loads(chunks_path(document_id).read_text(encoding="utf-8"))
            index = load_faiss_index(index_path(document_id))
            DOCUMENTS[document_id] = build_document_record(document_id, metadata, chunks, index)
            loaded += 1
        except Exception:
            continue

    return loaded


@app.on_event("startup")
def startup():
    # Clear in-memory documents
    DOCUMENTS.clear()

    # Delete persisted documents
    if DOCUMENT_ROOT.exists():
        shutil.rmtree(DOCUMENT_ROOT)

    DOCUMENT_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(DOCUMENTS)}


@app.get("/", response_class=HTMLResponse)
def web_app():
    return FileResponse(
        WEB_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/app.css")
def web_css():
    return FileResponse(
        WEB_DIR / "app.css",
        media_type="text/css",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/app.js")
def web_js():
    return FileResponse(
        WEB_DIR / "app.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/documents", dependencies=api_guard)
def list_documents():
    documents = []
    for document_id, document in DOCUMENTS.items():
        documents.append({
            "document_id": document_id,
            "filename": document["filename"],
            "pages": document["pages"],
            "chunks": len(document["chunks"]),
            "image_count": document["image_count"],
            "processing_time": document["processing_time"],
            "created_at": document["created_at"],
        })
    documents.sort(key=lambda item: item["created_at"], reverse=True)
    return {"documents": documents}


@app.post("/api/documents", dependencies=api_guard)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    started = time.time()
    document_id = str(uuid.uuid4())
    image_folder = IMAGE_ROOT / document_id
    doc_dir = document_dir(document_id)
    image_folder.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        total_bytes = 0
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                tmp.close()
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
                shutil.rmtree(doc_dir, ignore_errors=True)
                shutil.rmtree(image_folder, ignore_errors=True)
                max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
                raise HTTPException(status_code=413, detail=f"PDF is too large. Maximum size is {max_mb}MB.")
            tmp.write(chunk)
        temp_pdf_path = tmp.name

    try:
        pages = extract_pages_from_pdf(temp_pdf_path)
        chunks = chunk_pages(pages)
        index = create_faiss_index(chunks)
        image_count = extract_images_from_pdf(temp_pdf_path, str(image_folder))
        shutil.copyfile(temp_pdf_path, persisted_pdf_path(document_id))
    finally:
        try:
            os.remove(temp_pdf_path)
        except OSError:
            pass

    metadata = {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "image_folder": str(image_folder),
        "image_count": image_count,
        "processing_time": round(time.time() - started, 2),
        "created_at": time.time(),
    }
    persist_document(document_id, metadata, chunks, index)
    DOCUMENTS[document_id] = build_document_record(document_id, metadata, chunks, index)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks),
        "image_count": image_count,
        "processing_time": metadata["processing_time"],
    }


@app.post("/api/ask", dependencies=api_guard)
def ask_document(req: AskRequest):
    document = DOCUMENTS.get(req.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found or expired")

    result = generate_rag_result(req.question, document["chunks"], document["index"], top_k=req.top_k)
    return {
        "document_id": req.document_id,
        "question": req.question,
        **result,
    }


@app.get("/api/analytics/{document_id}", dependencies=api_guard)
def analytics(document_id: str):
    document = DOCUMENTS.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found or expired")

    return {
        "document_id": document_id,
        "filename": document["filename"],
        "pages": document["pages"],
        "chunks": len(document["chunks"]),
        "image_count": document["image_count"],
        "processing_time": document["processing_time"],
    }


@app.get("/api/images/{document_id}", dependencies=api_guard)
def list_images(document_id: str):
    document = DOCUMENTS.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found or expired")

    image_folder = Path(document["image_folder"])
    if not image_folder.exists():
        return {"document_id": document_id, "images": []}

    images = []
    for path in sorted(image_folder.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            images.append({
                "filename": path.name,
                "url": f"/api/images/{document_id}/{path.name}",
            })

    return {"document_id": document_id, "images": images}


@app.get("/api/images/{document_id}/{filename}", dependencies=api_guard)
def get_image(document_id: str, filename: str):
    document = DOCUMENTS.get(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found or expired")

    image_folder = Path(document["image_folder"]).resolve()
    image_path = (image_folder / filename).resolve()

    if image_folder not in image_path.parents or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)

@app.post("/api/tts")
def generate_tts(request: dict):
    text = request.get("text", "")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    audio_url = text_to_speech(text)

    return {
        "audio_url": audio_url
    }

@app.post("/api/export/pdf", dependencies=api_guard)
def export_pdf(req: ExportRequest):
    pdf = fitz.open()
    page = pdf.new_page(width=595, height=842)
    cursor_y = 54

    def write_block(title, body, y):
        title_rect = fitz.Rect(54, y, 541, y + 22)
        body_rect = fitz.Rect(54, y + 28, 541, 760)
        page.insert_textbox(title_rect, title, fontsize=15, fontname="helv", color=(0.05, 0.1, 0.2))
        used = page.insert_textbox(body_rect, body, fontsize=10.5, fontname="helv", color=(0.1, 0.1, 0.1), lineheight=1.35)
        return y + 32 + max(80, abs(used))

    cursor_y = write_block("NeuroDocs AI Answer", f"Question: {req.question}\n\n{req.answer}", cursor_y)

    if req.sources:
        source_lines = []
        for source in req.sources:
            page_no = source.get("page", "unknown")
            snippet = source.get("snippet", "")
            source_lines.append(f"[{source.get('id', '?')}] Page {page_no}: {snippet}")
        write_block("Sources", "\n\n".join(source_lines), min(cursor_y + 20, 620))

    output = pdf.tobytes()
    pdf.close()

    return Response(
        content=output,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=neurodocs-answer.pdf"},
    )

