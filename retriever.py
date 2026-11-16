"""N4–N5: поиск по базе (03b §2; T1.12). Реранк и бюджет контекста — T1.13.

Каналы среза:
- C0 — прямой доступ по реквизитам (Index.lookup): «ст. 70 ТК», «п. 3.2 договора П-17/2024». Дополнение
  к поиску, а не замена: в номере может быть ошибка (S4), поэтому остальные каналы работают всегда;
- C1 — полнотекстовый (FTS5/BM25) по каждому варианту запроса;
- C2 — по векторам bge-m3 (все варианты запроса — одним батч-вызовом); без эмбеддера — пропускается (C2).
Слияние — взвешенный RRF (k = 60); C0 получает фиксированный приоритет.
Бусты метаданных: узлы целевых документов и их спутников — выше; документы организаций, которые совпали
только частично и не стали целью, — ниже (S7). Это бусты, а не фильтры.

Агрегация (assemble): найденные узлы сворачиваются к адаптивному родителю; малый документ — к документу
целиком (03a §2.2). Целиком отдаются только документы распознанных целей, документы из верха выдачи и их
спутники, не больше max_whole_docs.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from kb_types import Dialog, RouterResult
from llm_client import LLMError, Trace

C0_BONUS = 1.0           # прямое попадание по реквизитам — выше любой суммы RRF
ARTICLE_REF_RE = re.compile(
    r"(?<![а-яa-z])(?:ст\.|стать\w*)\s*\d+(?:\.\d+)*(?:\s*,?\s*(?:ч\.|част\w*|п\.|пункт\w*|пп\.|подпункт\w*)\s*"
    r"(?:\d+(?:\.\d+)*(?:-\d+)?|[а-яё]+))*", re.IGNORECASE)
POINT_REF_RE = re.compile(r"(?<![а-яa-z])(?:пп\.|подпункт\w*|п\.|пункт\w*)\s*\d+(?:\.\d+)*(?:-\d+)?", re.IGNORECASE)


@dataclass
class Candidate:
    node_id: str
    score: float = 0.0
    channels: dict[str, int] = field(default_factory=dict)  # канал → место в его выдаче (с 1)


@dataclass
class Selection:
    """Результат агрегации: что пойдёт в контекст (до бюджета, T1.13)."""
    targets: list[str] = field(default_factory=list)          # doc_id (целиком) или node_id (родитель)
    hits: dict[str, list[str]] = field(default_factory=dict)  # цель → найденные в ней узлы
    scores: dict[str, float] = field(default_factory=dict)


def query_variants(dialog: Dialog, router: RouterResult) -> list[str]:
    """Варианты запроса: очищенный самостоятельный (роутер), исходная реплика, подвопросы
    и общий вопрос «шаг назад» (T3.30) — если отличаются."""
    variants = []
    for q in (router.query, dialog.last_user, *getattr(router, "subqueries", []),
              getattr(router, "stepback", "")):
        q = (q or "").strip()
        if q and q not in variants:
            variants.append(q)
    return variants


def direct_hits(index, dialog: Dialog, router: RouterResult, resolution) -> list[str]:
    """C0: положения, названные в реплике или запросе, — в названных или целевых документах."""
    text = f"{dialog.last_user}\n{router.query}"
    refs: list[str] = []
    covered: list[tuple[int, int]] = []
    for m in ARTICLE_REF_RE.finditer(text):
        # Часть/пункт могут стоять после названия акта: «ст. 59 ТК РФ, часть первая» — берём окно до конца фразы
        tail = re.split(r"[.?!;\n]", text[m.end(): m.end() + 60], maxsplit=1)[0]
        refs.append(m.group(0) + tail)
        covered.append((m.start(), m.end() + len(tail)))
    for m in POINT_REF_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in covered):
            refs.append(m.group(0))
    refs += router.node_refs
    hits: list[str] = []
    doc_refs = list(dict.fromkeys(router.doc_refs + [text]))
    target_docs = set(resolution.target_docs) if resolution else set()
    for ref in refs:
        if not ref.strip():
            continue
        is_article = bool(re.search(r"ст\.|стать", ref, re.IGNORECASE))
        found: list[str] = []
        for doc_ref in doc_refs:
            found = index.lookup(doc_ref, ref)
            if found:
                break
        if is_article and not found:
            found = index.lookup("", ref)  # «ст. 70» без акта — по всей базе
        if not is_article and target_docs:
            found = [n for n in found if index.node(n).doc_id in target_docs] or \
                [n for d in target_docs for n in index.lookup(index.doc(d).number or "", ref)
                 if index.node(n).doc_id == d]
        elif not is_article:
            found = []  # «п. 3.2» без документа неоднозначен — прямого попадания нет
        for n in found[:3]:
            if n not in hits:
                hits.append(n)
    return hits


def meta_sets(index, dialog: Dialog, resolution, settings=None) -> tuple[set[str], set[str]]:
    """Документы для буста (цели и спутники) и для понижения (организации, совпавшие только частично и не
    ставшие целью, — «похожий, но другой контрагент», S7)."""
    boost = set(resolution.target_docs) | set(resolution.satellites) if resolution else set()
    demote: set[str] = set()
    if resolution and resolution.target_docs:
        from router import recognize  # локальный импорт: избегаем цикла модулей
        for m in recognize(dialog, index).entities:
            if m.kind == "org" and not m.full:
                demote.update(d for d in m.doc_ids if d not in boost)
        if getattr(settings, "demote_other", False) and not resolution.candidates:
            # Кандидат D (S7, мягкий вариант T): цель однозначна — прочие документы компании понижаются, не удаляются
            related = set(boost)
            for d in list(boost):
                for r in index.related_docs(d):
                    related.update((r.src_doc, r.dst_doc))
            demote.update(c.doc_id for c in index.docs() if c.kind != "law" and c.doc_id not in related)
    return boost, demote


def meta_factor(doc_id: str, boost: set[str], demote: set[str], settings) -> float:
    if doc_id in boost:
        return settings.boost_target
    if doc_id in demote:
        return settings.penalty_other
    return 1.0


def _rrf(ranked: list[str], weight: float, k: int, scores: dict[str, Candidate], channel: str) -> None:
    for rank, nid in enumerate(ranked, 1):
        cand = scores.setdefault(nid, Candidate(nid))
        cand.score += weight / (k + rank)
        if channel not in cand.channels or rank < cand.channels[channel]:
            cand.channels[channel] = rank


def _prf_terms(index, first_hits: list[str], query_tokens: set[str],
               top_m: int = 8, n_terms: int = 5) -> list[str]:
    """T3.43: original measured RM3-lite, restored from 31771a9/7f52717."""
    from collections import Counter
    from textnorm import index_tokens
    counter: Counter = Counter()
    for nid in first_hits[:top_m]:
        try:
            text = index.search_text(nid)
        except Exception:
            continue
        for tok in index_tokens(text):
            if len(tok) >= 4 and tok not in query_tokens:
                counter[tok] += 1
    return [tok for tok, _ in counter.most_common(n_terms)]


def search_candidates(index, dialog: Dialog, router: RouterResult, resolution, client, settings,
                      trace: Trace | None = None, deadline: float | None = None) -> list[Candidate]:
    """Каналы C0–C2 → взвешенный RRF → бусты метаданных. Узлы-кандидаты по убыванию счёта."""
    s = settings
    variants = query_variants(dialog, router)
    cands: dict[str, Candidate] = {}

    # C1 — полнотекстовый
    fts_first: list[list[str]] = []
    for i, q in enumerate(variants):
        hits = [nid for nid, _ in index.search_fts(q, s.fts_k)]
        fts_first.append(hits)
        _rrf(hits, s.rrf_w_fts, s.rrf_k, cands, f"fts{i}")
    if getattr(s, "use_prf", False) and fts_first:
        from textnorm import index_tokens
        q_toks = {t for v in variants for t in index_tokens(v)}
        expanded = _prf_terms(index, fts_first[0], q_toks)
        if expanded:
            prf_q = " ".join([variants[0], *expanded])
            _rrf([nid for nid, _ in index.search_fts(prf_q, s.fts_k)],
                 s.rrf_w_fts, s.rrf_k, cands, "fts_prf")
            if trace is not None:
                trace.event("prf", terms=expanded)

    # C2 — по векторам (одним батч-вызовом для всех вариантов)
    if client is not None and index.vectors is not None and variants:
        try:
            dense_variants = ([BGE_QUERY_PREFIX + v for v in variants]
                              if getattr(s, "use_bge_query_prefix", False) else variants)
            vecs = client.embed(dense_variants, trace=trace, **({"deadline": deadline} if deadline is not None else {}))
            for i, v in enumerate(vecs):
                _rrf([nid for nid, _ in index.search_dense(v, s.dense_k)], s.rrf_w_dense, s.rrf_k, cands, f"dense{i}")
        except LLMError as exc:
            if trace is not None:
                trace.event("dense_skipped", error=str(exc)[:200])

    # C4 — HyDE (T3.17, OFF): гипотетический фрагмент нормы как дополнительный вариант.
    # Исследование (SemEval-2026, deep-research-report GPT 6.1-sol): HyDE сильнее всего
    # работает в BM25-канале (+26.7% против +4% у dense) → вариант идёт в оба канала.
    if (getattr(s, "use_hyde", False) and client is not None and index.vectors is not None and variants):
        try:
            hyde_text = _hyde_text(client, variants[0], trace, deadline)
            if hyde_text:
                hyde_vec = client.embed([hyde_text], trace=trace,
                                        **({"deadline": deadline} if deadline is not None else {}))[0]
                _rrf([nid for nid, _ in index.search_dense(hyde_vec, s.dense_k)],
                     s.rrf_w_dense, s.rrf_k, cands, "hyde0")
                _rrf([nid for nid, _ in index.search_fts(hyde_text, s.fts_k)],
                     s.rrf_w_fts, s.rrf_k, cands, "hyde_fts")
                if trace is not None:
                    trace.event("hyde", chars=len(hyde_text))
        except LLMError as exc:
            if trace is not None:
                trace.event("hyde_skipped", error=str(exc)[:200])

    # C6 — Query2Doc (T3.40, OFF): псевдо-ответ на запрос в лексический канал (Wang 2023).
    if (getattr(s, "use_query2doc", False) and client is not None and variants):
        try:
            q2d_text = _query2doc_text(client, variants[0], trace, deadline)
            if q2d_text:
                _rrf([nid for nid, _ in index.search_fts(q2d_text, s.fts_k)],
                     s.rrf_w_fts, s.rrf_k, cands, "q2d0")
                if trace is not None:
                    trace.event("query2doc", chars=len(q2d_text))
        except LLMError as exc:
            if trace is not None:
                trace.event("query2doc_skipped", error=str(exc)[:200])

    # C5 — doc2query (T3.39, OFF): векторы офлайн-сгенерированных вопросов к узлам —
    # дополнительный dense-канал без платформенных вызовов (поиск — numpy локально).
    # Идентификаторы матрицы строго валидируются по фактическому индексу: при несовпадении
    # базы/нарезки канал отключается целиком (урок платформенного прогона 2fad06a: 18,571 %).
    if getattr(s, "use_d2q", False) and index.vectors is not None and variants and client is not None:
        try:
            qvec = client.embed([variants[0]], trace=trace,
                                **({"deadline": deadline} if deadline is not None else {}))[0]
            d2q = _d2q_top(qvec, s.dense_k, index, model=s.emb_model)
            if d2q:
                _rrf(d2q, s.rrf_w_dense, s.rrf_k, cands, "d2q0")
                if trace is not None:
                    trace.event("d2q", hits=len(d2q))
        except LLMError as exc:
            if trace is not None:
                trace.event("d2q_skipped", error=str(exc)[:200])

    # C0 — прямое попадание по реквизитам
    for rank, nid in enumerate(direct_hits(index, dialog, router, resolution), 1):
        cand = cands.setdefault(nid, Candidate(nid))
        cand.score += C0_BONUS / rank
        cand.channels["direct"] = rank

    # Бусты метаданных: цели и спутники выше; документы частично совпавших организаций, не ставших целью, — ниже
    boost_docs, demote_docs = meta_sets(index, dialog, resolution, settings)
    for cand in cands.values():
        cand.score *= meta_factor(index.node(cand.node_id).doc_id, boost_docs, demote_docs, s)

    ranked = sorted(cands.values(), key=lambda c: -c.score)
    if trace is not None:
        trace.event("retrieval", candidates=len(ranked), direct=sum("direct" in c.channels for c in ranked),
                    variants=len(variants))
    return ranked


def _strict_filter(index, ranked_nodes: list[str], resolution, settings) -> list[str]:
    """Кандидат T (S7): если документ-цель распознан однозначно (контрагент, номер; не Q19-неоднозначность),
    узлы других документов компании отбрасываются — похожий договор другого контрагента не попадает в ответ.
    Законы и связанные с целями документы (спутники, доп. соглашения, редакции) остаются."""
    if not getattr(settings, "strict_targets", False) or not resolution or not resolution.target_docs             or resolution.candidates:
        return ranked_nodes
    allowed = set(resolution.target_docs) | set(resolution.satellites)
    for d in list(allowed):
        for r in index.related_docs(d):
            allowed.update((r.src_doc, r.dst_doc))
    return [n for n in ranked_nodes
            if index.doc(index.node(n).doc_id).kind == "law" or index.node(n).doc_id in allowed]


def assemble(index, ranked_nodes: list[str], resolution, settings, top_n: int | None = None) -> Selection:
    """Сворачивание найденных узлов к контексту: малый документ — целиком, иначе адаптивный родитель.
    Целиком (03b §2, S7): если цель распознана (контрагент, номер) — только документы целей и их спутники;
    иначе — документы из верха выдачи и их спутники; не больше max_whole_docs. Остальные найденные узлы
    малых документов идут отдельными пунктами — похожие документы не смешиваются."""
    sel = Selection()
    ranked_nodes = _strict_filter(index, ranked_nodes, resolution, settings)
    top = ranked_nodes[:top_n] if top_n else ranked_nodes
    whole_allowed: list[str] = []
    for d in (resolution.target_docs if resolution else []):
        if index.doc(d).whole and d not in whole_allowed:
            whole_allowed.append(d)
    if not whole_allowed:
        for nid in top:
            d = index.node(nid).doc_id
            if index.doc(d).whole and d not in whole_allowed:
                whole_allowed.append(d)
    satellites: list[str] = []
    for d in list(whole_allowed):
        for r in index.related_docs(d):
            other = r.dst_doc if r.src_doc == d else r.src_doc
            if index.doc(other).whole and other not in whole_allowed and other not in satellites:
                satellites.append(other)
    if resolution:
        satellites += [d for d in resolution.satellites if index.doc(d).whole and d not in whole_allowed
                       and d not in satellites]
    whole_allowed = (whole_allowed + satellites)[: max(0, settings.max_whole_docs)] if settings.max_whole_docs \
        else []

    for rank, nid in enumerate(top):
        node = index.node(nid)
        target = node.doc_id if node.doc_id in whole_allowed else index.context_parent(nid)
        if target == node.doc_id and node.doc_id not in whole_allowed:
            target = nid  # малый документ, не допущенный целиком, — сам узел
        if target not in sel.hits:
            sel.targets.append(target)
            sel.hits[target] = []
            sel.scores[target] = 0.0
        sel.hits[target].append(nid)
        sel.scores[target] += 1.0 / (rank + 1)  # несколько попаданий в одного родителя поднимают его
    # Документы целей и спутники, не попавшие в выдачу, всё равно идут в контекст (последними)
    for d in whole_allowed:
        if d not in sel.hits:
            sel.targets.append(d)
            sel.hits[d] = []
            sel.scores[d] = 0.0
    found = [t for t in sel.targets if sel.hits[t]]
    found.sort(key=lambda t: -sel.scores[t])
    sel.targets = found + [t for t in sel.targets if not sel.hits[t]]
    return sel


# ---------------- реранк и бюджет контекста (T1.13; 03b §3) ----------------

@dataclass
class ContextItem:
    target: str            # doc_id (документ целиком) или node_id (родитель/узел)
    whole: bool
    hits: list[str]        # найденные узлы внутри (для пометки «◀ найдено» в досье)
    chars: int


@dataclass
class Retrieval:
    """Итог поиска для генерации: что войдёт в контекст и в каком порядке."""
    candidates: list[Candidate] = field(default_factory=list)
    reranked: list[tuple[str, float]] = field(default_factory=list)   # (node_id, relevance) top-N
    items: list[ContextItem] = field(default_factory=list)
    best_node_id: str | None = None
    reranker_used: bool = False
    dropped: list[str] = field(default_factory=list)                  # не вошло по бюджету


def select_top_n(index, ranked: list[tuple[str, float]], settings,
                 protect: set[str] | None = None) -> list[tuple[str, float]]:
    """Выбор top-`rerank_top_n` из упорядоченной выдачи. При включённом кандидате PC
    (03b §3) — мягкая квота на контекстного родителя: первый проход в порядке рангов берёт
    не больше `rerank_parent_cap` узлов на родителя, второй добирает пропущенные по рангу.
    Узлы документов из protect (цели разрешения и их спутники) квотой не ограничиваются.
    При выключенном флаге — прежний срез."""
    top_n = settings.rerank_top_n
    if not settings.use_rerank_parent_cap:
        return ranked[:top_n]
    cap = settings.rerank_parent_cap
    protected = protect or set()
    first: list[tuple[str, float]] = []
    skipped: list[tuple[str, float]] = []
    counts: dict[str, int] = {}
    for item in ranked:
        if index.node(item[0]).doc_id in protected:
            first.append(item)
        else:
            parent = index.context_parent(item[0])
            if counts.get(parent, 0) >= cap:
                skipped.append(item)
                continue
            counts[parent] = counts.get(parent, 0) + 1
            first.append(item)
        if len(first) >= top_n:
            break
    if len(first) < top_n:
        first.extend(skipped[: top_n - len(first)])
    return first

RERANK_INSTRUCTION = ("Find the contract clause, internal policy provision or article of law that directly answers the "
                      "bank employee's legal question, or that contains a condition, exception or sanction relevant to it")

HYDE_SYSTEM = """Напиши один гипотетический фрагмент нормы документа (2–4 предложения), который прямо
отвечал бы на вопрос сотрудника банка. Используй типовые формулировки договоров, внутренних регламентов
или статей закона; конкретные реквизиты, названия организаций и суммы не выдумывай. Только текст
фрагмента нормы, без пояснений и без оформления."""

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

QUERY2DOC_SYSTEM = """Напиши короткий фактический ответ (2–4 предложения) на вопрос сотрудника банка, каким он был бы
по типовому договору, внутреннему регламенту или статье закона. Типовые формулировки; конкретные реквизиты,
названия организаций и суммы не выдумывай. Только текст ответа, без пояснений и оформления."""


def _hyde_text(client, query: str, trace: Trace | None, deadline: float | None) -> str:
    result = client.chat([{"role": "system", "content": HYDE_SYSTEM},
                          {"role": "user", "content": query}],
                         max_tokens=600, trace=trace,
                         **({"deadline": deadline} if deadline is not None else {}))
    text = getattr(result, "content", result) or ""
    return str(text).strip()[:1500]


def _query2doc_text(client, query: str, trace: Trace | None, deadline: float | None) -> str:
    """T3.40: псевдо-ответ на запрос для лексического канала (Query2Doc, Wang 2023)."""
    result = client.chat([{"role": "system", "content": QUERY2DOC_SYSTEM},
                          {"role": "user", "content": query}],
                         max_tokens=400, trace=trace,
                         **({"deadline": deadline} if deadline is not None else {}))
    text = getattr(result, "content", result) or ""
    return str(text).strip()[:1200]


def _d2q_top(query_vec, k: int = 40, index=None,
             path: str = str(Path(__file__).resolve().parent / "d2q_vectors.npz"),
             model: str | None = None):
    """NPZ v2 only: exact top-k unique nodes, with per-row source/model validation.

    A generated question is usable only for the current source text. Legacy files
    cannot prove this. Optional/malformed artifacts degrade to None, not KeyError.
    """
    global _D2Q_CACHE
    try:
        import hashlib
        import io
        import numbers
        import numpy as np

        if not isinstance(k, numbers.Integral) or isinstance(k, bool):
            return None
        if k <= 0:
            return []
        known = getattr(index, "_nodes", {})
        expected_model = model or getattr(index, "emb_stats", {}).get("model")
        if query_vec is None or not known or not isinstance(expected_model, str) or not expected_model.strip():
            return None
        filename = str(Path(path).resolve())
        raw = Path(filename).read_bytes()
        fingerprint = hashlib.sha256(raw).hexdigest()
        if not isinstance(_D2Q_CACHE, dict):
            _D2Q_CACHE = {}
        cached = _D2Q_CACHE.get(filename)
        if cached is None or cached[0] != fingerprint:
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                version = data["version"]
                stored_model = data["model"]
                ids = data["node_ids"]
                hashes = data["node_text_hashes"]
                vectors = data["vectors"]
                if (version.shape != () or not np.issubdtype(version.dtype, np.integer)
                        or version.item() != 2 or stored_model.shape != ()
                        or stored_model.dtype.kind not in "US"
                        or ids.ndim != 1 or ids.dtype.kind not in "US"
                        or hashes.ndim != 1 or hashes.dtype.kind not in "US"
                        or vectors.ndim != 2 or not vectors.shape[1]
                        or len(ids) != len(hashes) or len(ids) != len(vectors)
                        or not np.issubdtype(vectors.dtype, np.number)
                        or np.iscomplexobj(vectors)):
                    return None
                ids = [str(nid) for nid in ids]
                hashes = [str(value) for value in hashes]
                with np.errstate(over="ignore", invalid="ignore"):
                    mat = vectors.astype(np.float32)
                    norms = np.linalg.norm(mat, axis=1)
                finite = np.isfinite(mat).all(axis=1) & np.isfinite(norms) & (norms > 0)
                mat = mat[finite] / norms[finite, None]
                ids = [nid for nid, keep in zip(ids, finite) if keep]
                hashes = [value for value, keep in zip(hashes, finite) if keep]
                cached = (fingerprint, str(stored_model.item()), ids, hashes, mat)
            # Cache identity includes file content, not just its path/mtime/size.
            if len(_D2Q_CACHE) >= 8 and filename not in _D2Q_CACHE:
                _D2Q_CACHE.pop(next(iter(_D2Q_CACHE), None), None)
            _D2Q_CACHE[filename] = cached
        _, stored_model, ids, hashes, mat = cached
        if stored_model.strip().casefold() != expected_model.strip().casefold():
            return None
        q = np.asarray(query_vec)
        if (q.ndim != 1 or q.shape[0] != mat.shape[1]
                or not np.issubdtype(q.dtype, np.number) or np.iscomplexobj(q)):
            return None
        q = q.astype(np.float32)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(q).all() or not np.isfinite(norm) or norm <= 0:
            return None
        selected = []
        for position, (nid, source_hash) in enumerate(zip(ids, hashes)):
            node = known.get(nid)
            if node is None or not isinstance(node.text, str) or not node.text.strip():
                continue
            if hashlib.sha256(node.text.encode("utf-8")).hexdigest() == source_hash:
                selected.append(position)
        if not selected:
            return None
        scores = mat[selected] @ (q / norm)
        best = {}
        for position, score in zip(selected, scores):
            if not np.isfinite(score):
                continue
            nid = ids[position]
            if nid not in best or score > best[nid]:
                best[nid] = float(score)
        if not best:
            return None
        return sorted(best, key=lambda nid: (-best[nid], nid))[:min(int(k), len(best))]
    except Exception:
        return None


_D2Q_CACHE = None


def rerank_pool_limit(settings) -> int:
    """T3.16: кандидат OFF-флагом расширяет пул реранкера 40→48 без изменения дефолта."""
    return 48 if getattr(settings, "use_rerank_pool_48", False) else settings.rerank_candidates


def rerank(index, query: str, candidates: list[Candidate], client, settings, trace: Trace | None = None,
           keep_first: list[str] | None = None, meta: tuple[set[str], set[str]] | None = None,
           deadline: float | None = None, target_docs: set[str] | None = None) -> tuple[list[tuple[str, float]], bool]:
    """Qwen3 Reranker по кандидатам-узлам (короткий текст поиска, не родитель); к его оценкам применяются те же
    поправки метаданных, что и к RRF (реранкер не знает, о каком контрагенте вопрос, — S7). Прямые попадания
    (C0) сохраняются в выдаче. При сбое реранкера — порядок RRF (C2)."""
    pool = [c.node_id for c in candidates[: rerank_pool_limit(settings)]]
    keep = [n for n in (keep_first or []) if n in pool]
    used = False
    ranked: list[tuple[str, float]]
    if client is not None and pool and query:
        docs = [index.search_text(n)[: settings.rerank_doc_chars] for n in pool]
        if getattr(settings, "rerank_instruct", False):
            query = f"Instruct: {RERANK_INSTRUCTION}\nQuery: {query}"
        try:
            order = client.rerank(query, docs, top_n=len(docs), trace=trace,
                                  **({"deadline": deadline} if deadline is not None else {}))
            ranked = [(pool[i], s) for i, s in order]
            if meta:
                ranked = [(n, s * meta_factor(index.node(n).doc_id, meta[0], meta[1], settings)) for n, s in ranked]
                ranked.sort(key=lambda x: -x[1])
            used = True
        except LLMError as exc:
            if trace is not None:
                trace.event("rerank_skipped", error=str(exc)[:200])
            ranked = [(c.node_id, c.score) for c in candidates[: rerank_pool_limit(settings)]]
    else:
        ranked = [(c.node_id, c.score) for c in candidates[: rerank_pool_limit(settings)]]
    top = select_top_n(index, ranked, settings, protect=meta[0] if meta else None)
    # D: только при успешном реранке; OFF сохраняет исходный путь побайтово по результату.
    if used and getattr(settings, "use_rerank_doc_diversity", False) and len(top) >= 2:
        docs = [index.node(n).doc_id for n, _ in top]
        last_node, _ = top[-1]
        if len(set(docs)) == 1 and last_node not in keep and not (
            docs[-1] in (target_docs or set()) and docs.count(docs[-1]) == 1
        ):
            demoted = meta[1] if meta else set()
            alternate = next(((n, score) for n, score in ranked[len(top):16]
                              if index.node(n).doc_id != docs[0]
                              and index.node(n).doc_id not in demoted), None)
            if alternate is not None:
                top[-1] = alternate
    # D2: заменить дубликат контекста другим родителем уже представленного документа.
    if used and getattr(settings, "use_rerank_parent_diversity", False) and len(top) >= 2:
        parents = [index.context_parent(n) for n, _ in top]
        docs = [index.node(n).doc_id for n, _ in top]
        demoted = meta[1] if meta else set()
        spare = next((i for i in range(len(top) - 1, -1, -1)
                      if top[i][0] not in keep and parents.count(parents[i]) > 1
                      and not (docs[i] in (target_docs or set()) and docs.count(docs[i]) == 1)), None)
        alternate = next(((n, score) for n, score in ranked[len(top):16]
                          if index.node(n).doc_id in docs and index.node(n).doc_id not in demoted
                          and index.context_parent(n) not in parents), None)
        if spare is not None and alternate is not None:
            top[spare] = alternate
    top_ids = {n for n, _ in top}
    for n in keep:  # C0 — всегда в верхушке
        if n not in top_ids:
            score = next((s for nid, s in ranked if nid == n), 0.0)
            top.append((n, score))
    return top, used


def apply_budget(index, sel: Selection, settings) -> tuple[list[ContextItem], list[str]]:
    """Бюджет символов контекста (R5): цели по порядку; документ, не влезающий целиком, представлен
    родителями найденных в нём узлов; первая цель включается всегда."""
    items: list[ContextItem] = []
    dropped: list[str] = []
    used = 0
    budget = settings.context_budget_chars
    doc_ids = {c.doc_id for c in index.docs()}
    seen: set[str] = set()

    def add(target: str, whole: bool, hits: list[str]) -> bool:
        nonlocal used
        if target in seen:
            return True
        text_len = len(index.context_text(target))
        if items and used + text_len > budget:
            return False
        items.append(ContextItem(target, whole, hits, text_len))
        seen.add(target)
        used += text_len
        return True

    for target in sel.targets:
        hits = sel.hits.get(target, [])
        whole = target in doc_ids
        if add(target, whole, hits):
            continue
        if whole and hits:
            for nid in hits:  # документ целиком не влез — его найденные пункты с родителями
                parent = index.context_parent(nid, allow_whole=False)
                if not add(parent, False, [nid]):
                    dropped.append(parent)
        else:
            dropped.append(target)
    if getattr(settings, "context_neighbors", False):
        _add_neighbors(index, items, add, getattr(settings, "neighbors_max", 0))
    return items, dropped


def _siblings(index, node_id: str) -> list[str]:
    node = index.node(node_id)
    if node.parent_id:
        return [c.node_id for c in index.children(node.parent_id)]
    return [n.node_id for n in sorted(index.doc_nodes(node.doc_id), key=lambda n: n.start) if not n.parent_id]


def _add_neighbors(index, items: list, add, limit: int = 0) -> None:
    """Кандидат N (03b §3): условия и исключения нормы часто стоят в соседних частях той же статьи или пункта.
    Для каждого найденного фрагмента (не документа целиком), в порядке значимости, добавляются первая часть
    того же уровня (общее правило) и смежные части — пока хватает бюджета. Уже покрытый текст не дублируется.
    limit > 0 — не больше limit добавленных частей (соседи самых значимых фрагментов идут первыми)."""
    added = 0
    covered = context_nodes(index, items)
    ancestors = set()
    for it in items:
        if it.whole:
            continue
        nid = it.target
        while nid:
            ancestors.add(nid)
            nid = index.node(nid).parent_id
    for it in list(items):
        if it.whole:
            continue
        sibs = _siblings(index, it.target)
        if it.target not in sibs or len(sibs) < 2:
            continue
        i = sibs.index(it.target)
        for cand in dict.fromkeys([sibs[0], *sibs[max(0, i - 1):i], *sibs[i + 1:i + 2]]):
            if cand == it.target or cand in covered or cand in ancestors:
                continue
            if limit and added >= limit:
                return
            before = len(items)
            if add(cand, False, []) and len(items) > before:
                covered |= context_nodes(index, [items[-1]])
                added += 1


def retrieve(index, dialog: Dialog, router: RouterResult, resolution, client, settings,
             trace: Trace | None = None, deadline: float | None = None) -> Retrieval:
    """Поиск целиком: каналы → RRF → реранк → агрегация → бюджет."""
    out = Retrieval()
    out.candidates = search_candidates(index, dialog, router, resolution, client, settings, trace, deadline)
    if not out.candidates:
        if resolution and resolution.target_docs:  # ничего не найдено, но документ назван — отдаём его
            sel = assemble(index, [], resolution, settings)
            out.items, out.dropped = apply_budget(index, sel, settings)
        return out
    direct = [c.node_id for c in out.candidates if "direct" in c.channels]
    query = router.query or dialog.last_user
    out.reranked, out.reranker_used = rerank(index, query, out.candidates, client, settings, trace, direct,
                                             meta=meta_sets(index, dialog, resolution, settings), deadline=deadline,
                                             target_docs=set(resolution.target_docs) if resolution else set())
    out.best_node_id = out.reranked[0][0] if out.reranked else None
    ranked_ids = [n for n, _ in out.reranked]
    if getattr(settings, "use_subqueries", False) and getattr(router, "subqueries", None) and out.reranker_used:
        # Кандидат S: каждый аспект реранжируется отдельно; его лучшие узлы добавляются квотой (S9).
        # Реранки подвопросов независимы — идут параллельно (вызовов столько же, время вопроса меньше);
        # результаты разбираются в исходном порядке подвопросов, поэтому выдача та же, что при последовательном.
        meta = meta_sets(index, dialog, resolution, settings)
        with ThreadPoolExecutor(max_workers=len(router.subqueries)) as pool:
            results = list(pool.map(lambda sq: rerank(index, sq, out.candidates, client, settings, trace, None,
                                                       meta=meta, deadline=deadline,
                                                       target_docs=set(resolution.target_docs) if resolution else set()),
                                    router.subqueries))
        for sub_top, used in results:
            if not used:
                continue
            added = 0
            for n, _ in sub_top:
                if added >= settings.subquery_quota:
                    break
                if n not in ranked_ids:
                    ranked_ids.append(n)
                    added += 1
        if trace is not None:
            trace.event("subqueries", count=len(router.subqueries), nodes=len(ranked_ids))
    sel = assemble(index, ranked_ids, resolution, settings)
    out.items, out.dropped = apply_budget(index, sel, settings)
    if trace is not None:
        trace.event("context", items=len(out.items), chars=sum(i.chars for i in out.items),
                    whole=sum(i.whole for i in out.items), dropped=len(out.dropped), reranker=out.reranker_used)
    return out


def context_nodes(index, items: list[ContextItem]) -> set[str]:
    """Все узлы, текст которых попадает в контекст (для Recall@контекст и проверки цитат)."""
    covered: set[str] = set()
    for item in items:
        if item.whole:
            covered.update(n.node_id for n in index.doc_nodes(item.target))
            continue
        stack = [item.target]
        full_subtree = index.context_text(item.target) == index.subtree_text(item.target)
        while stack:
            nid = stack.pop()
            covered.add(nid)
            if full_subtree:
                stack.extend(c.node_id for c in index.children(nid))
    return covered
