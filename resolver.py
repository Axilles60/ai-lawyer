"""N3 — разрешение (03b §1.4; Q2, Q13; T1.11): о каких документах вопрос, какая редакция действовала на дату
события, нужна ли уточняющая реплика.

Всё считается кодом (P5.7) и передаётся дальше как готовые факты:
- целевые документы — по полным совпадениям в последней реплике, затем в очищенном запросе роутера
  (он подставляет контрагента и номер из истории), затем в истории диалога;
- «своя» организация не различает документы (она сторона почти всех) — в выборе цели не участвует;
- редакции одного документа (связи replaces) — одно семейство: из него выбирается редакция, действовавшая
  на дату события (без даты — действующая, Q13), но в контекст идут все редакции с пометками статуса (S5);
- неоднозначность, которую не сняли ни реплика, ни история (несколько организаций по частичному совпадению,
  несколько документов одного вида без номера), → уточняющий вопрос с перечислением вариантов (Q2).
  Текст уточнения собирается кодом из карточек — без вызова LLM и без выдуманных реквизитов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from indexer import doc_title, fmt_date
from kb_types import Dialog, EntityMatch, RouterResult

PLURAL_DOC_RE = re.compile(r"(?<![а-яё])(договор(ы|ов|ам|ами|ах)|соглашени(я|й|ям|ях)|регламент(ы|ов|ам|ах))(?![а-яё])",
                           re.IGNORECASE)


@dataclass
class Resolution:
    target_docs: list[str] = field(default_factory=list)       # документы, о которых вопрос (по приоритету)
    satellites: list[str] = field(default_factory=list)        # связанные: доп. соглашения, редакции, письма
    preferred: dict[str, str] = field(default_factory=dict)    # семейство редакций → редакция на дату события
    notes: dict[str, str] = field(default_factory=dict)        # doc_id → статус на дату события
    event_date: str = ""
    clarify: str = ""                                          # текст уточняющего вопроса (пусто — не нужен)
    candidates: list[str] = field(default_factory=list)        # doc_id вариантов при неоднозначности
    source: str = ""                                           # откуда цель: reply | query | history | none
    primary_doc: str = ""                                      # предположение, не исключает альтернативы
    ambiguity_reason: str = ""


def edition_families(index) -> dict[str, str]:
    """doc_id → ключ семейства редакций (документы, связанные replaces, — одно семейство)."""
    parent = {c.doc_id: c.doc_id for c in index.docs()}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in index.relations:
        if r.kind == "replaces":
            parent[find(r.src_doc)] = find(r.dst_doc)
    return {d: find(d) for d in parent}


def _docs_of(match: EntityMatch, index) -> list[str]:
    """Документы сущности; для организации — только документы компании (договоры, соглашения…)."""
    return [d for d in match.doc_ids if index.doc(d).kind != "law"] if match.kind == "org" else list(match.doc_ids)


def _pick(matches: list[EntityMatch], index, own_id: str | None) -> tuple[list[str], list[EntityMatch]]:
    """Цели по полным совпадениям (кроме своей организации и актов законодательства) и частичные кандидаты."""
    full = [m for m in matches if m.full and m.entity_id != own_id and m.kind in ("org", "document")]
    # Номер документа (100) точнее вида документа (80): при наличии точных совпадений общие отбрасываем
    if any(m.score >= 100 for m in full):
        full = [m for m in full if m.score >= 100]
    docs: list[str] = []
    for m in full:
        for d in _docs_of(m, index):
            if d not in docs:
                docs.append(d)
    partial = [m for m in matches if not m.full and m.kind == "org" and m.entity_id != own_id and m.score >= 50]
    return docs, partial


def _variant(doc_id: str, index) -> str:
    card = index.doc(doc_id)
    other = [name for name, role, _ in card.parties if role and role != "издатель"
             and not (index.own_org() and name == index.own_org().canonical)]
    with_whom = f" с {other[0]}" if other else ""
    return f"{doc_title(card)[0].lower()}{doc_title(card)[1:]}{with_whom}"


def _clarify_text(doc_ids: list[str], index) -> str:
    variants = [_variant(d, index) for d in doc_ids]
    body = variants[0] if len(variants) == 1 else ", ".join(variants[:-1]) + " или " + variants[-1]
    return f"Уточните, пожалуйста, о каком документе идёт речь: {body}?"


def _families_count(doc_ids: list[str], families: dict[str, str]) -> int:
    return len({families.get(d, d) for d in doc_ids})


def _variants(res: Resolution, ids: list[str], matches, index) -> list[str]:
    """Порядок условных вариантов; техническая сортировка не устанавливает намерение."""
    scores = {d: max((m.score for m in matches if d in _docs_of(m, index)), default=0) for d in ids}
    best = max(scores.values(), default=0)
    leaders = [d for d in ids if scores[d] == best]
    res.primary_doc = leaders[0] if best and len(leaders) == 1 else ""
    res.ambiguity_reason = "unique_recognition_score" if res.primary_doc else "equal_or_no_distinguishing_evidence"
    ordered = sorted(ids, key=lambda d: (d != res.primary_doc, _variant(d, index).casefold()))
    res.candidates = ordered
    return ordered


def resolve_request(dialog: Dialog, router: RouterResult, rec, index, use_answer_variants: bool = False) -> Resolution:
    """Разрешение сущностей, редакций и неоднозначности. Чистая функция от вопроса и индекса."""
    own = index.own_org()
    own_id = own.entity_id if own else None
    families = edition_families(index)
    res = Resolution(event_date=router.event_date)

    # 1. Последняя реплика → 2. очищенный запрос роутера → 3. история диалога
    reply_docs, reply_partial = _pick(rec.entities, index, own_id)
    query_matches = index.resolve(router.query) if router.query else []
    query_docs, _ = _pick(query_matches, index, own_id)
    history_docs, _ = _pick(rec.history_entities, index, own_id)
    if use_answer_variants:
        # Переписывание запроса не создаёт доказательство намерения пользователя.
        supported = set(reply_docs) | set(history_docs)
        if _families_count(list(supported), families) > 1:
            supported = set()  # переписывание не сужает неоднозначную историю
        query_matches = [m for m in query_matches if m.kind not in ("org", "document") or
                         set(_docs_of(m, index)) <= supported]
        query_docs, _ = _pick(query_matches, index, own_id)
    if reply_docs:
        targets, res.source = reply_docs, "reply"
    elif query_docs:
        targets, res.source = query_docs, "query"
    elif history_docs:
        targets, res.source = history_docs, "history"
    else:
        targets, res.source = [], "none"

    # Неоднозначность: частичные совпадения нескольких организаций без полного совпадения нигде
    if not targets and reply_partial:
        cand: list[str] = []
        for m in reply_partial:
            for d in _docs_of(m, index):
                if d not in cand:
                    cand.append(d)
        primary = [d for d in cand if index.doc(d).kind == "contract"] or cand
        if len({m.entity_id for m in reply_partial}) > 1 and _families_count(primary, families) > 1:
            if not use_answer_variants:
                res.candidates = primary
                res.clarify = _clarify_text(primary, index)
                return res
            targets = _variants(res, primary, reply_partial, index)
            res.source = "reply"
        else:
            targets, res.source = cand, "reply"

    # Неоднозначность: несколько документов одного вида без номера и контрагента («договор поставки»)
    ambiguity_targets = ([d for d in targets if index.doc(d).kind == "contract"] or targets) if use_answer_variants else targets
    kinds = {index.doc(d).kind for d in ambiguity_targets}
    sources = ("reply", "query", "history") if use_answer_variants else ("reply", "query")
    if (not res.candidates and res.source in sources and len(kinds) == 1 and _families_count(ambiguity_targets, families) > 1
            and not PLURAL_DOC_RE.search(dialog.last_user)
            and all(not m.full or m.score < 100 for m in rec.entities if m.kind == "document")
            and (use_answer_variants or not any(m.full and m.kind == "org" and m.entity_id != own_id
                                               for m in rec.entities + query_matches))):
        if not use_answer_variants:
            res.candidates = targets
            res.clarify = _clarify_text(targets, index)
            return res
        targets = _variants(res, ambiguity_targets, rec.entities, index)

    res.target_docs = targets

    # Спутники: связанные документы целей (доп. соглашения, другие редакции, письма-дополнения)
    for d in targets:
        for r in index.related_docs(d):
            other = r.dst_doc if r.src_doc == d else r.src_doc
            if other not in targets and other not in res.satellites:
                res.satellites.append(other)

    # Редакции на дату события: в каждом семействе редакций — действовавшая на дату (без даты — действующая).
    # Считается для всех семейств: документ может найтись поиском, даже если в вопросе не назван (DEMO-5)
    by_family: dict[str, list[str]] = {}
    for c in index.docs():
        by_family.setdefault(families[c.doc_id], []).append(c.doc_id)
    for fam, members in by_family.items():
        if len(members) > 1:
            active = [m for m in members if index.status_on(m, router.event_date or None).startswith("действ")]
            if active:
                res.preferred[fam] = max(active, key=lambda m: index.doc(m).date)
            for m in members:
                res.notes[m] = index.status_on(m, router.event_date or None)
    for d in targets + res.satellites:
        res.notes[d] = index.status_on(d, router.event_date or None)
    return res


def edition_notes(doc_ids: list[str], index, event_date: str = "") -> dict[str, str]:
    """Пометки статуса для документов контекста (используется при сборке досье, T1.14)."""
    return {d: index.status_on(d, event_date or None) for d in doc_ids}


def describe(res: Resolution, index) -> str:
    """Короткое описание для трейса и отладки."""
    def names(ids: list[str]) -> str:
        return ", ".join(doc_title(index.doc(d)) for d in ids)
    parts = [f"источник цели: {res.source}"]
    if res.target_docs:
        parts.append(f"цели: {names(res.target_docs)}")
    if res.satellites:
        parts.append(f"спутники: {names(res.satellites)}")
    if res.preferred:
        parts.append("редакции на дату: " + names(list(res.preferred.values())))
    if res.event_date:
        parts.append(f"дата события: {res.event_date}")
    if res.clarify:
        parts.append(f"уточнение: {res.clarify}")
    return "; ".join(parts)


__all__ = ["Resolution", "resolve_request", "edition_families", "edition_notes", "describe", "fmt_date"]
