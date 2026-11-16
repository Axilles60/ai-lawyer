"""N2–N3a: распознавание реквизитов кодом и роутер (LLM #1, JSON) — 03b §1, §1.1; T1.10.

1. Код до LLM: сущности по каталогу (Index.resolve), номера документов, ссылки на пункты/статьи/части,
   даты в вопросе и истории. Это подсказки роутеру и запасной путь.
2. Роутер получает реплику, историю, подсказки и опись базы и возвращает RouterResult:
   область legal/offtopic, флаги manipulation и illegal (флаги, а не конечные ветки — Q15),
   очищенный самостоятельный запрос, реквизиты, дату события, утверждения пользователя.
3. Ответ модели проверяется кодом; при сбое API или невалидном JSON — RouterResult из кода
   (область legal, запрос = реплика пользователя, флаги — по regex-guard).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dialog import history_text
from kb_parser import DATE_RE, MONTHS
from kb_types import Dialog, EntityMatch, RouterResult
from llm_client import LLMError, Trace
from prompts import router_messages
from textnorm import DOC_NUMBER_RE, BARE_DOC_NUMBER_RE

HARD_GUARD_FLAGS = {"instruction_override", "role_change", "fake_system", "prompt_leak", "encoded"}
NODE_REF_RE = re.compile(
    r"(?<![а-яa-z])(?:пп\.|подпункт\w*|п\.|пункт\w*|ст\.|стать\w*|ч\.|част\w*)\s*\d+(?:\.\d+)*(?:-\d+)?",
    re.IGNORECASE)
MONTH_YEAR_RE = re.compile(rf"(?P<mon>{'|'.join(m[:-1] for m in MONTHS)})\w*\s+(?P<y>\d{{4}})", re.IGNORECASE)
YEAR_RE = re.compile(r"(?<!\d)(?:в\s+)?(?P<y>(?:19|20)\d{2})\s*(?:год\w*|г\.)", re.IGNORECASE)
ISO_DATE_RE = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$")


@dataclass
class Recognition:
    """Что нашёл код до вызова LLM (подсказки и запасной путь)."""
    entities: list[EntityMatch] = field(default_factory=list)          # по последней реплике
    history_entities: list[EntityMatch] = field(default_factory=list)  # по истории диалога
    doc_numbers: list[str] = field(default_factory=list)
    node_refs: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)                     # ISO, «ГГГГ-ММ» или «ГГГГ»


def find_dates(text: str) -> list[str]:
    """Даты в тексте: полные → ISO; «в августе 2024 года» → «2024-08»; «в 2024 году» → «2024»."""
    found: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in DATE_RE.finditer(text or ""):
        if m.group("d"):
            d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
        else:
            d, mo, y = int(m.group("d2")), MONTHS[m.group("mon").lower()], int(m.group("y2"))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            found.append(f"{y:04d}-{mo:02d}-{d:02d}")
            spans.append(m.span())
    for m in MONTH_YEAR_RE.finditer(text or ""):
        if any(s <= m.start() < e for s, e in spans):
            continue
        stem = m.group("mon").lower()
        month = next(v for k, v in MONTHS.items() if k.startswith(stem))
        found.append(f"{int(m.group('y')):04d}-{month:02d}")
        spans.append(m.span())
    for m in YEAR_RE.finditer(text or ""):
        if any(s <= m.start() < e for s, e in spans):
            continue
        found.append(m.group("y"))
    return list(dict.fromkeys(found))


def recognize(dialog: Dialog, index) -> Recognition:
    text = dialog.last_user
    hist = "\n".join(t.text for t in dialog.history)
    numbers = [m.group(1) for m in DOC_NUMBER_RE.finditer(text)] + \
              [m.group(1) for m in BARE_DOC_NUMBER_RE.finditer(text)]
    return Recognition(
        entities=index.resolve(text),
        history_entities=index.resolve(hist) if hist else [],
        doc_numbers=list(dict.fromkeys(numbers)),
        node_refs=list(dict.fromkeys(m.group(0).strip() for m in NODE_REF_RE.finditer(text))),
        dates=find_dates(text),
    )


def _hints(rec: Recognition, dialog: Dialog) -> dict:
    def names(matches: list[EntityMatch]) -> list[str]:
        return [f"{m.canonical}{'' if m.full else ' (частичное совпадение)'}" for m in matches[:6]]
    hints = {"организации_и_документы": names(rec.entities)}
    if rec.history_entities:
        hints["упомянуто_в_истории"] = names(rec.history_entities)
    if rec.doc_numbers:
        hints["номера_документов"] = rec.doc_numbers
    if rec.node_refs:
        hints["положения"] = rec.node_refs
    if rec.dates:
        hints["даты"] = rec.dates
    if dialog.guard:
        hints["признаки_манипуляции"] = dialog.guard
    return hints


def _str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:10]


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "да", "yes", "1"}
    return bool(value)


def parse_router_output(data, dialog: Dialog, rec: Recognition) -> RouterResult:
    """Проверка ответа модели кодом: типы, допустимые значения, пустой запрос."""
    if not isinstance(data, dict):
        raise ValueError("ответ роутера — не объект JSON")
    scope = str(data.get("scope", "legal")).strip().lower()
    scope = scope if scope in {"legal", "offtopic"} else "legal"
    manipulation = _bool(data.get("manipulation")) or bool(HARD_GUARD_FLAGS & set(dialog.guard))
    query = str(data.get("query") or "").strip()
    if scope == "legal" and not query and not manipulation:
        query = dialog.last_user  # юридический вопрос без запроса — ищем по самой реплике
    event_date = str(data.get("event_date") or "").strip()
    if not ISO_DATE_RE.match(event_date):
        event_date = rec.dates[0] if rec.dates else ""
    return RouterResult(
        scope=scope, manipulation=manipulation, illegal=_bool(data.get("illegal")), query=query,
        doc_refs=_str_list(data.get("doc_refs")) or rec.doc_numbers, node_refs=_str_list(data.get("node_refs"))
        or rec.node_refs, entities=_str_list(data.get("entities")), event_date=event_date,
        user_claims=_str_list(data.get("user_claims")),
        subqueries=[q for q in _str_list(data.get("subqueries")) if q.strip() and q.strip() != query][:3],
        stepback=str(data.get("stepback") or "").strip()[:400])


def fallback_route(dialog: Dialog, rec: Recognition) -> RouterResult:
    """Роутер без LLM (C2): ищем по реплике, флаги — по regex-guard."""
    return RouterResult(
        scope="legal", manipulation=bool(HARD_GUARD_FLAGS & set(dialog.guard)), illegal=False,
        query=dialog.last_user, doc_refs=rec.doc_numbers, node_refs=rec.node_refs,
        entities=[m.canonical for m in rec.entities if m.full], event_date=rec.dates[0] if rec.dates else "",
        user_claims=[])


def route(dialog: Dialog, index, client, settings=None, trace: Trace | None = None,
          deadline: float | None = None) -> tuple[RouterResult, Recognition]:
    """Распознавание кодом + роутер. Никогда не бросает исключение: при сбое — запасной путь."""
    rec = recognize(dialog, index)
    if client is None:
        return fallback_route(dialog, rec), rec
    hints = _hints(rec, dialog)
    if getattr(settings, "use_safe_context", False):
        from safe_context import clean_value
        hints = clean_value(hints)
    messages = router_messages(dialog.last_user, history_text(dialog), index.inventory(), hints,
                               subqueries=bool(getattr(settings, "use_subqueries", False)),
                               stepback=bool(getattr(settings, "use_stepback", False)))
    try:
        data = client.chat_json(messages, trace=trace, **({"deadline": deadline} if deadline is not None else {}))
        result = parse_router_output(data, dialog, rec)
        if trace is not None:
            trace.event("router", scope=result.scope, manipulation=result.manipulation, illegal=result.illegal,
                        event_date=result.event_date, claims=len(result.user_claims))
        return result, rec
    except (LLMError, ValueError) as exc:
        if trace is not None:
            trace.event("router_fallback", error=str(exc)[:200])
        return fallback_route(dialog, rec), rec
