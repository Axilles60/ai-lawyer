"""N7 — генерация ответа (LLM #2, 03b §4.2; T1.14).

Модель получает досье и возвращает JSON: decision, claims (text с метками [[Д…]], sources, quote), missing.
Здесь — только вызов и приведение ответа к Draft. Проверки V1/V2/V10, сборка текста со ссылками
и повторная генерация — T1.15 (verifier, pipeline).
"""

from __future__ import annotations

import dataclasses
import json
import re

from dossier import build_context, render_context
from kb_types import Claim, Context, Dialog, Draft, RouterResult
from llm_client import LLMError, Trace
from prompts import answer_messages

TAG_RE = re.compile(r"\[\[\s*(Д\d+(?:\.\d+)?)\s*\]\]")
DECISIONS = {"answer", "refuse", "clarify"}


def _tag(value) -> str:
    """«[[Д1.4]]», «Д1.4», «[Д1.4]» → «Д1.4»."""
    m = re.search(r"Д\d+(?:\.\d+)?", str(value or ""))
    return m.group(0) if m else ""


def parse_draft(data) -> Draft:
    """Ответ модели → Draft. Мягкое приведение типов; строгая проверка схемы — V1 (T1.15)."""
    if not isinstance(data, dict):
        raise ValueError("ответ генератора — не объект JSON")
    decision = str(data.get("decision", "answer")).strip().lower()
    decision = decision if decision in DECISIONS else "answer"
    claims = []
    for item in data.get("claims") or []:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        sources = item.get("sources") or []
        if isinstance(sources, str):
            sources = [sources]
        tags = [t for t in (_tag(s) for s in sources) if t]
        for t in TAG_RE.findall(text):  # метки из текста тоже считаются источниками
            if t not in tags:
                tags.append(t)
        claims.append(Claim(text=text, sources=tags, quote=str(item.get("quote") or "").strip()))
    missing = str(data.get("missing") or "").strip()
    return Draft(decision=decision, claims=claims, missing=missing)


def _chain_of_note(ctx, dialog, client, trace, deadline):
    """T3.19: сжатие досье дословными выписками, относящимися к вопросу. При ошибке или
    пустом результате — исходное досье (OFF-поведение части контракта)."""
    from prompts import CHAIN_OF_NOTE_SYSTEM
    blocks = [(tag, text) for tag, _doc, _whole, text in ctx.blocks]
    if not blocks:
        return ctx
    payload = json.dumps({"question": getattr(dialog, "last_user", ""),
                          "blocks": [{"tag": t, "text": x[:2000]} for t, x in blocks]},
                         ensure_ascii=False)
    try:
        data = client.chat_json([{"role": "system", "content": CHAIN_OF_NOTE_SYSTEM},
                                 {"role": "user", "content": payload}], trace=trace,
                                **({"deadline": deadline} if deadline is not None else {}))
    except Exception:
        return ctx
    notes = data.get("notes") if isinstance(data, dict) else None
    if not isinstance(notes, dict):
        return ctx
    kept = []
    for tag, doc_id, whole, text in ctx.blocks:
        note = str(notes.get(tag) or "").strip()
        if note:
            kept.append((tag, doc_id, whole, f"[{tag}] {note}"))
    if not kept:
        return ctx
    if trace is not None:
        trace.event("chain_of_note", blocks_in=len(ctx.blocks), blocks_out=len(kept))
    return dataclasses.replace(ctx, blocks=kept)


def generate(index, dialog: Dialog, router: RouterResult, resolution, retrieval, client, settings=None,
             trace: Trace | None = None, extra_note: str = "",
             deadline: float | None = None) -> tuple[Draft, Context]:
    """Досье → модель → Draft. Исключения LLMError/ValueError пробрасываются: решение о запасном ответе —
    в pipeline (T1.15)."""
    ctx = (build_context(index, retrieval, resolution, router, lean=True)
           if getattr(settings, "use_lean_context", False)
           else build_context(index, retrieval, resolution, router))
    q19 = getattr(settings, "use_answer_variants", False)
    content = render_context(ctx, index, router, resolution, dialog, use_answer_variants=True) if q19 else \
        render_context(ctx, index, router, resolution, dialog)
    if extra_note:
        content += f"\n\nЗАМЕЧАНИЯ К ПРЕДЫДУЩЕМУ ОТВЕТУ (исправь):\n{extra_note}"
    if getattr(settings, "use_safe_context", False):
        from safe_context import clean
        # A critique/cycle note may quote an untrusted instruction after the dossier was
        # rendered. Clean only the final visible copy, not the original Context or note.
        content = clean(content)
    options = {}
    if getattr(settings, "use_scoped_missing", False):
        options["use_scoped_missing"] = True
    if q19:
        options["use_answer_variants"] = True
    if getattr(settings, "use_conditional_example_v3", False):
        options["use_conditional_example_v3"] = True
    if getattr(settings, "use_conditional_example_v2", False):
        options["use_conditional_example_v2"] = True
    if getattr(settings, "use_conditional_example", False):
        options["use_conditional_example"] = True
    if getattr(settings, "use_historical_law_limit", False):
        options["use_historical_law_limit"] = True
    if getattr(settings, "use_actionable_gaps", False):
        options["use_actionable_gaps"] = True
    if getattr(settings, "use_applicability_guard", False):
        options["use_applicability_guard"] = True
    if getattr(settings, "use_conditional_first", False):
        options["use_conditional_first"] = True
    if getattr(settings, "use_complete_answer", False):
        options["use_complete_answer"] = True

    if getattr(settings, "use_claim_discipline", False):
        options["use_claim_discipline"] = True
    if getattr(settings, "use_norm_analysis", False):
        options["use_norm_analysis"] = True
    if getattr(settings, "use_minimal", False):
        options["use_minimal"] = True
    if getattr(settings, "use_typed_facts", False):
        options["use_typed_facts"] = True
    if getattr(settings, "use_chain_of_note", False) and client is not None:
        ctx = _chain_of_note(ctx, dialog, client, trace, deadline)
    messages = answer_messages(content, **options)
    data = client.chat_json(messages, trace=trace,
                            **({"deadline": deadline} if deadline is not None else {}))
    draft = parse_draft(data)
    if trace is not None:
        trace.event("generation", decision=draft.decision, claims=len(draft.claims), missing=bool(draft.missing),
                    context_chars=len(content))
    return draft, ctx


__all__ = ["generate", "parse_draft", "TAG_RE", "LLMError"]
