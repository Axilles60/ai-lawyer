"""Contextual Retrieval (T3.38, OFF): офлайн-контексты узлов и их применение при индексации.

Файл contextual_cache.json (отгружается в архиве):
{"model": <LLM>, "created": <ISO>, "contexts": {doc_sha256: {node_id: context}}}
Schema v2 binds every hint to the full document SHA and exact canonical node text.
Hints are stored separately from the source; only search_text renders their prefix.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

CACHE_NAME = str(Path(__file__).resolve().parent / "contextual_cache.json")
_CONTEXT_PREFIX = "[Контекст: {ctx}] "


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def context_record(text: str, context: str) -> dict:
    return {"text_sha256": text_hash(text), "context": context}


def load_contexts(path: Path | str | None = None) -> dict:
    """Read v2 only: legacy node IDs alone do not prove compatibility after rechunking."""
    path = Path(path or CACHE_NAME)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 2:
            return {}
        contexts = data.get("contexts")
        return contexts if isinstance(contexts, dict) else {}
    except Exception:
        return {}


def apply_contexts(nodes, contexts: dict, doc_sha: str) -> int:
    """Attach verified hints without changing text that verifier treats as primary evidence."""
    per_doc = contexts.get(doc_sha) or {}
    if not isinstance(per_doc, dict):
        per_doc = {}
    count = 0
    for node in nodes:
        record = per_doc.get(node.node_id)
        if not isinstance(record, dict) or record.get("text_sha256") != text_hash(node.text):
            node.contextual_text = ""
            continue
        ctx = record.get("context")
        if not isinstance(ctx, str) or not ctx.strip():
            node.contextual_text = ""
            continue
        ctx = " ".join(ctx.split())[:300]
        if node.contextual_text != ctx:
            node.contextual_text = ctx
            count += 1
    return count


def write_contexts(path: Path | str, model: str, contexts: dict) -> None:
    from datetime import datetime, timezone
    # Do not silently certify legacy strings using the current chunk's coordinates.
    if any(not isinstance(record, dict) or not isinstance(record.get("context"), str)
           or not isinstance(record.get("text_sha256"), str) or len(record["text_sha256"]) != 64
           for doc in contexts.values() for record in doc.values()):
        raise ValueError("Contextual records require source text fingerprints")
    payload = {"schema_version": 2, "model": model, "created": datetime.now(timezone.utc).isoformat(),
               "contexts": contexts}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
