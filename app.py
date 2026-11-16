"""Локальный веб-интерфейс общего ядра; без серверного хранения диалогов."""
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import get_settings
from indexer import build_index, doc_title
from kb_loader import KB_ARCHIVE_NAMES, load_kb
from pipeline import answer
from solution import make_client, runtime_settings
from verifier import _reference_text

ROOT = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app):
    settings = runtime_settings(get_settings())
    archive = Path(os.environ.get("KB_PATH", "knowledge-base.zip"))
    if not archive.is_file():
        archive = next((Path(settings.data_dir) / n for n in KB_ARCHIVE_NAMES if (Path(settings.data_dir) / n).is_file()), archive)
    if not archive.is_file():
        raise RuntimeError("Архив базы не найден: задайте KB_PATH")
    client = make_client(settings)
    try:
        app.state.settings, app.state.client = settings, client
        app.state.index = build_index(load_kb(archive), None if settings.use_llm_only else client, settings)
        yield
    finally:
        transport = getattr(client, "_http", None)
        if transport is not None:
            transport.close()

app = FastAPI(lifespan=lifespan)

class Turn(BaseModel):
    role: str
    text: str = Field(min_length=1, max_length=10000)

class Request(BaseModel):
    turns: list[Turn] = Field(min_length=1, max_length=30)

def dialogue(turns):
    if turns[-1].role != "user" or any(t.role != ("user" if i % 2 == 0 else "assistant") for i, t in enumerate(turns)):
        raise HTTPException(400, "Ожидается чередование реплик, последняя — пользователя")
    # Метки в тексте не должны становиться самостоятельными репликами парсера.
    import re
    from dialog import ROLE_RE
    return "\n".join(("Пользователь: " if t.role == "user" else "Ассистент: ") +
                     ROLE_RE.sub("", t.text.replace("\r", "\n")) for t in turns)

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/documents")
def documents():
    return [{"id": c.doc_id, "title": _reference_text(doc_title(c), app.state.index)}
            for c in app.state.index.docs()]

@app.get("/api/documents/{doc_id}")
def document(doc_id: str, node: str = ""):
    index = app.state.index
    try:
        card = index.doc(doc_id)
        parsed = next(p for p in index.parsed if p.card.doc_id == doc_id)
        text = parsed.text
        fragment = index.node(node) if node else None
        if fragment and fragment.doc_id != doc_id:
            raise KeyError(node)
    except (KeyError, StopIteration):
        raise HTTPException(404, "Документ или фрагмент не найден")
    excerpt = fragment.text if fragment else ""
    start = -1
    if excerpt:
        if 0 <= fragment.start < fragment.end <= len(text) and text[fragment.start:fragment.end] == excerpt:
            start = fragment.start
        else:
            found = text.find(excerpt)
            if found >= 0 and text.find(excerpt, found + 1) < 0:
                start = found
    return {"title": _reference_text(doc_title(card), index), "text": text, "fragment": excerpt,
            "start": start, "end": start + len(excerpt) if start >= 0 else -1}

@app.post("/api/chat")
def chat(request: Request):
    result = answer(dialogue(request.turns), app.state.index, app.state.client, app.state.settings)
    sources = []
    for node_id in dict.fromkeys(result.node_ids):
        try:
            node = app.state.index.node(node_id)
            sources.append({"node": node_id, "document": node.doc_id,
                            "title": _reference_text(doc_title(app.state.index.doc(node.doc_id)), app.state.index),
                            "label": _reference_text(node.label, app.state.index)})
        except KeyError:
            try:
                card = app.state.index.doc(node_id)
            except KeyError:
                continue
            sources.append({"node": "", "document": node_id,
                            "title": _reference_text(doc_title(card), app.state.index), "label": "документ целиком"})
    return {"answer": result.answer, "fallback": result.fallback, "sources": sources}
