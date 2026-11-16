"""Эксперимент «всё на DeepSeek» (USE_LLM_ONLY): поиск, ранжирование и «эмбеддинги» силами генеративной модели.

Без реранкера и эмбеддера (bge-m3 и Qwen3 Reranker не вызываются, векторы индекса не строятся):
- «эмбеддер»: модель расширяет запрос — термины, синонимы, ссылки на нормы, гипотетический фрагмент ответа;
  это уходит в лексический (FTS5) канал, который не требует никакой модели;
- «ретривер»: модель получает карту базы (документы D…, пункты N…) и выбирает пункты и документы по смыслу;
- «реранкер»: модель ранжирует объединённый список кандидатов одним листовым запросом;
- контекст: найденные документы целиком в порядке релевантности, остаток бюджета добирается остальными документами
  (база небольшая — обычно читается почти вся);
- циклы: при отказе или запасном ответе весь цикл поиск → ранжирование → досье → генерация повторяется (pipeline).

Каждый шаг при сбое модели молча деградирует к лексическому поиску, поэтому режим не может быть хуже запасного пути.
Идентификаторы, которых нет в карте, отбрасываются (модель не вносит в досье выдуманные пункты).
"""

from __future__ import annotations

import dataclasses
import threading

from dialog import history_text
from indexer import doc_header
from kb_types import Dialog, RouterResult
from llm_client import LLMError, Trace
from retriever import C0_BONUS, Candidate, ContextItem, Retrieval, _rrf, search_candidates

MAP_MAX_CHARS = 110_000        # потолок размера карты базы в запросе выбора
SELECT_MAX_NODES = 30
SELECT_MAX_DOCS = 5
RERANK_KEEP = 30
CONTEXT_SCALE = (1.0, 1.0, 1.0, 0.55, 0.3)   # доля бюджета контекста по ступеням экономии 0…4

EXPAND_SYSTEM = """Ты помогаешь искать ответ на юридический вопрос в базе документов компании (договоры, регламенты, законы).
Сам на вопрос не отвечай. Верни только JSON: {"keywords": ["..."], "hypothetical_answer": "..."}.
keywords — до 15 слов и коротких выражений, которые могут встретиться в нужных пунктах: термины в разных формах и синонимы,
названия документов и сторон, номера статей и пунктов, если они названы в вопросе.
hypothetical_answer — 2–3 предложения в стиле документа: как мог бы звучать нужный пункт. Не придумывай числа и имена."""

SELECT_SYSTEM = """Ты — поисковый модуль юридического ассистента. Ниже карта базы: документы (D…) и их пункты (N…).
По вопросу выбери пункты, где может находиться ответ, и документы, которые стоит прочитать целиком. Выбирай с запасом
(до 30 пунктов и до 5 документов), включай связанные пункты: определения, исключения, сроки, ответственность.
Используй ТОЛЬКО идентификаторы из карты. Верни только JSON: {"nodes": ["N12", "N40"], "docs": ["D3"]}."""

RERANK_SYSTEM = """Ты — модуль ранжирования юридического ассистента. Ниже вопрос и фрагменты документов с идентификаторами.
Расставь фрагменты по полезности для ответа: сначала те, что содержат ответ, условия, исключения, сроки, суммы,
последствия, затем вспомогательные. Явно нерелевантные не включай. Используй ТОЛЬКО идентификаторы из списка.
Верни только JSON: {"ranking": ["N12", "N40"]} (до 30 идентификаторов, лучшие первыми)."""

_CAP_LOCK = threading.Lock()
_CAP = {"chars": 0}           # общий потолок контекста; уменьшается, если модель отклонила слишком длинный запрос
_HANDLE_LOCK = threading.Lock()


def llm_only_settings(s):
    """Настройки режима: большой контекст, больше перегенераций, длинные таймауты вызовов с большим запросом."""
    return dataclasses.replace(
        s, use_llm_only=True, context_budget_chars=220_000, whole_doc_chars=120_000, parent_max_chars=20_000,
        max_whole_docs=64, rerank_top_n=RERANK_KEEP, rerank_candidates=60, rerank_doc_chars=900, max_regenerations=3,
        http_timeout_s=max(s.http_timeout_s, 200.0), chat_timeout_s=max(s.chat_timeout_s, 170.0),
        call_budget_chat_s=max(s.call_budget_chat_s, 240.0), question_budget_s=max(s.question_budget_s, 420.0))


# ---------------- общий потолок контекста ----------------

def effective_budget(settings) -> int:
    with _CAP_LOCK:
        cap = _CAP["chars"]
    base = int(settings.context_budget_chars)
    return min(base, cap) if cap else base


def looks_like_context_overflow(error: str) -> bool:
    low = (error or "").lower()
    return "http 400" in low or "http 413" in low or "context" in low or "too long" in low or "maximum" in low


def note_generation_failure(error: str, settings) -> None:
    """Запрос отклонён из-за длины (HTTP 400/413, «context»): следующие вопросы начинаются с меньшим контекстом."""
    if looks_like_context_overflow(error):
        with _CAP_LOCK:
            now = min(settings.context_budget_chars, _CAP["chars"] or settings.context_budget_chars)
            _CAP["chars"] = max(30_000, int(now * 0.6))


def reset_cap() -> None:
    with _CAP_LOCK:
        _CAP["chars"] = 0


# ---------------- идентификаторы и карта базы ----------------

def _handles(index) -> tuple[dict, dict, dict, dict]:
    """(N→node_id, node_id→N, D→doc_id, doc_id→D) для поисковых узлов и документов."""
    with _HANDLE_LOCK:
        cached = getattr(index, "_llm_rag_handles", None)
        if not (isinstance(cached, tuple) and len(cached) == 4 and all(isinstance(x, dict) for x in cached)):
            by_handle = {f"N{i}": n.node_id for i, n in enumerate(index.searchable_nodes(), 1)}
            docs = {f"D{i}": d.doc_id for i, d in enumerate(index.docs(), 1)}
            cached = (by_handle, {v: k for k, v in by_handle.items()}, docs, {v: k for k, v in docs.items()})
            # The cache belongs to the object, never to a recyclable Python id. A restricted
            # test/index adapter may disallow attributes; recomputing is safe in that case.
            try:
                index._llm_rag_handles = cached
            except (AttributeError, TypeError):
                pass
    return cached


def _visible(index, text: str) -> str:
    if getattr(index, "use_safe_context", False):
        from safe_context import clean
        return clean(text)
    return text


def kb_map(index, max_chars: int = MAP_MAX_CHARS) -> str:
    by_handle, by_node, _, doc_handle = _handles(index)
    lines: list[str] = []
    for card in index.docs():
        lines.append(f"{doc_handle[card.doc_id]} | {_visible(index, doc_header(card))}")
        for node in index.doc_nodes(card.doc_id):
            handle = by_node.get(node.node_id)
            if handle:
                title = " ".join(_visible(index, node.heading or node.text).split())[:70]
                lines.append(f"  {handle} {_visible(index, node.label)}: {title}")
    return _visible(index, "\n".join(lines)[:max_chars])


def _ids(value, known: dict, limit: int) -> list[str]:
    """Идентификаторы из ответа модели (строки или объекты с id); неизвестные отбрасываются, порядок сохраняется."""
    out: list[str] = []
    if isinstance(value, dict):
        value = value.get("ranking") or value.get("nodes") or value.get("ids") or []
    if not isinstance(value, list):
        return out
    for item in value:
        if isinstance(item, dict):
            item = item.get("id") or item.get("node") or ""
        handle = str(item).strip().strip("[]").upper()
        if handle in known and known[handle] not in out:
            out.append(known[handle])
        if len(out) >= limit:
            break
    return out


def _call(client, messages, trace: Trace | None, deadline: float | None, safe_context: bool = False):
    if safe_context:
        from safe_context import clean
        # Clean data at the final request boundary as well as before presentation truncation.
        # System instructions are trusted constants and are not part of document cleanup.
        messages = [dict(m, content=clean(m["content"])) if m.get("role") == "user" else m for m in messages]
    kwargs = {"deadline": deadline} if deadline is not None else {}
    return client.chat_json(messages, trace=trace, max_tokens=2048, **kwargs)


def _question_block(dialog: Dialog, router: RouterResult) -> str:
    parts = []
    history = history_text(dialog)
    if history:
        parts.append(f"ИСТОРИЯ ДИАЛОГА:\n{history}")
    parts.append(f"ВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{dialog.last_user}")
    if router.query and router.query != dialog.last_user:
        parts.append(f"(Суть вопроса: {router.query})")
    return "\n\n".join(parts)


# ---------------- шаги поиска ----------------

def llm_expand(client, dialog, router, hint, trace, deadline, safe_context: bool = False) -> list[str]:
    """«Эмбеддер» силами LLM: запросы для лексического канала (ключевые слова и гипотетический фрагмент)."""
    user = _question_block(dialog, router)
    if hint:
        user += f"\n\nПредыдущая попытка не нашла: {hint}\nДобавь термины, которые помогут найти недостающее."
    data = _call(client, [{"role": "system", "content": EXPAND_SYSTEM}, {"role": "user", "content": user}],
                 trace, deadline, safe_context=safe_context)
    if not isinstance(data, dict):
        return []
    words = data.get("keywords") or []
    queries: list[str] = []
    if isinstance(words, list):
        queries.append(" ".join(str(w) for w in words if str(w).strip())[:600])
    hypo = str(data.get("hypothetical_answer") or "").strip()
    if hypo:
        queries.append(hypo[:800])
    return [q for q in queries if q]


def llm_select(index, client, dialog, router, hint, trace, deadline) -> tuple[list[str], list[str]]:
    """«Ретривер» силами LLM: выбор пунктов и документов по карте базы. Возвращает (node_id…, doc_id…)."""
    by_handle, _, docs, _ = _handles(index)
    user = f"КАРТА БАЗЫ:\n{kb_map(index)}\n\n{_question_block(dialog, router)}"
    if hint:
        user += f"\n\nПредыдущая попытка не нашла: {hint}"
    data = _call(client, [{"role": "system", "content": SELECT_SYSTEM}, {"role": "user", "content": user}],
                 trace, deadline, safe_context=getattr(index, "use_safe_context", False))
    if not isinstance(data, dict):
        return [], []
    return _ids(data.get("nodes"), by_handle, SELECT_MAX_NODES), _ids(data.get("docs"), docs, SELECT_MAX_DOCS)


def llm_rerank(index, client, dialog, router, pool: list[str], settings, trace, deadline) -> list[str]:
    """«Реранкер» силами LLM: листовое ранжирование пула кандидатов. Возвращает node_id по убыванию пользы."""
    by_handle, by_node, _, _ = _handles(index)
    # Прямые попадания по реквизитам («ст. 164, п. 4») могут быть служебными узлами без идентификатора карты:
    # им нечего показывать модели, и обращаться к by_node[nid] нельзя (иначе вопрос падал в «техническую ошибку»)
    pool = [nid for nid in pool if nid in by_node]
    if not pool:
        return []
    lines = []
    for nid in pool:
        node = index.node(nid)
        card = index.doc(node.doc_id)
        label = " ".join(_visible(index, str(card.number or card.title or card.act or "")).split())[:40]
        text = " ".join(index.search_text(nid).split())[:settings.rerank_doc_chars]
        lines.append(f"[{by_node[nid]}] ({label}; {_visible(index, node.label)}) {text}")
    user = f"{_question_block(dialog, router)}\n\nФРАГМЕНТЫ:\n" + "\n".join(lines)
    data = _call(client, [{"role": "system", "content": RERANK_SYSTEM}, {"role": "user", "content": user}],
                 trace, deadline, safe_context=getattr(index, "use_safe_context", False))
    in_pool = set(pool)
    return _ids(data, {k: v for k, v in by_handle.items() if v in in_pool}, RERANK_KEEP)


def full_context_items(index, ranked_nodes: list[str], budget: int, hit_nodes: list[str]) -> tuple[list[ContextItem], list[str]]:
    """Документы целиком в порядке релевантности; остаток бюджета добирается остальными документами.
    Возвращает (элементы контекста, не вошедшие документы)."""
    order: list[str] = []
    for nid in ranked_nodes:
        doc_id = index.node(nid).doc_id
        if doc_id not in order:
            order.append(doc_id)
    for card in index.docs():
        if card.doc_id not in order:
            order.append(card.doc_id)
    hits: dict[str, list[str]] = {}
    for nid in hit_nodes:
        hits.setdefault(index.node(nid).doc_id, []).append(nid)
    items: list[ContextItem] = []
    dropped: list[str] = []
    used = 0
    for doc_id in order:
        size = len(index.doc_text(doc_id))
        if items and used + size > budget:
            dropped.append(doc_id)
            continue
        items.append(ContextItem(doc_id, True, hits.get(doc_id, []), size))
        used += size
    return items, dropped


def retrieve_llm(index, dialog: Dialog, router: RouterResult, resolution, client, settings,
                 trace: Trace | None = None, deadline: float | None = None, economy: int = 0,
                 hint: str = "") -> Retrieval:
    """Поиск → ранжирование → бюджет контекста силами LLM. Ступени экономии (pipeline, `economy`):
    ≥2 без выбора по карте; ≥3 без LLM-ранжирования; ≥4 без расширения запроса."""
    s = settings
    out = Retrieval()
    base = search_candidates(index, dialog, router, resolution, None, s, trace, deadline)   # FTS + C0 + бусты, без векторов
    cands: dict[str, Candidate] = {c.node_id: c for c in base}
    used: list[str] = []

    if economy < 4:
        try:
            queries = llm_expand(client, dialog, router, hint, trace, deadline,
                                 safe_context=getattr(index, "use_safe_context", False))
            for q in queries + ([hint] if hint else []):
                try:
                    _rrf([nid for nid, _ in index.search_fts(q, s.fts_k)], s.rrf_w_fts, s.rrf_k, cands, "llm_exp")
                except Exception:   # синтаксис MATCH не должен ронять вопрос
                    continue
            used.append("expand")
        except (LLMError, ValueError) as exc:
            if trace is not None:
                trace.event("llm_expand_failed", error=str(exc)[:200])

    selected_nodes: list[str] = []
    if economy < 2:
        try:
            selected_nodes, _docs = llm_select(index, client, dialog, router, hint, trace, deadline)
            for rank, nid in enumerate(selected_nodes, 1):
                cand = cands.setdefault(nid, Candidate(nid))
                cand.score += 0.6 * C0_BONUS / rank
                cand.channels["llm_sel"] = rank
            used.append("select")
        except (LLMError, ValueError) as exc:
            if trace is not None:
                trace.event("llm_select_failed", error=str(exc)[:200])

    ordered = sorted(cands.values(), key=lambda c: -c.score)
    out.candidates = ordered
    pool = [c.node_id for c in ordered[: s.rerank_candidates]]
    for nid in selected_nodes:
        if nid not in pool:
            pool.append(nid)
    ranked: list[tuple[str, float]] = [(c.node_id, c.score) for c in ordered[: s.rerank_candidates]]
    if economy < 3 and pool:
        try:
            order = llm_rerank(index, client, dialog, router, pool, s, trace, deadline)
            if order:
                rest = [n for n in pool if n not in order]
                ranked = [(n, 1.0 - i / (len(order) + len(rest) + 1)) for i, n in enumerate(order + rest)]
                out.reranker_used = True
                used.append("rerank")
        except (LLMError, ValueError) as exc:
            if trace is not None:
                trace.event("llm_rerank_failed", error=str(exc)[:200])

    direct = [c.node_id for c in ordered if "direct" in c.channels]
    top = ranked[: s.rerank_top_n]
    top_ids = {n for n, _ in top}
    for nid in direct:                      # прямые попадания по реквизитам остаются в верхушке
        if nid not in top_ids:
            top.append((nid, 0.0))
    out.reranked = top
    out.best_node_id = top[0][0] if top else None
    # На высоких ступенях экономии уменьшается и контекст: большая генерация — главная статья расхода времени
    budget = int(effective_budget(s) * CONTEXT_SCALE[min(max(economy, 0), len(CONTEXT_SCALE) - 1)])
    items, dropped = full_context_items(index, [n for n, _ in ranked], budget, [n for n, _ in top])
    out.items, out.dropped = items, dropped
    if trace is not None:
        trace.event("llm_retrieval", steps=used, candidates=len(ordered), pool=len(pool), docs=len(items),
                    chars=sum(i.chars for i in items), dropped=len(dropped), economy=economy, hint=bool(hint))
    return out
