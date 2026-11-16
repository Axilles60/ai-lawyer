"""Разбор документов базы знаний (T1.2: тип документа, шапка, связи уровня документа).

Всё определяется по содержимому общими правилами оформления российских документов (C1):
никаких имён контрагентов, номеров и имён файлов выданной базы. Имя файла — не используется.

Соглашение о датах: ISO yyyy-mm-dd. valid_from — первый день действия; valid_to — первый день,
с которого документ НЕ действует («утратил силу с 01.03.2025» → valid_to = 2025-03-01).

Дерево узлов (T1.3), сущности (T1.6) и темпоральная разметка (T1.7) добавляются следующими задачами.
"""

from __future__ import annotations

import re

from kb_tree import build_nodes
from kb_types import DocCard, Entity, ParsedDoc, RawDoc, Relation

HEADER_LINES = 8          # сколько первых строк считается шапкой
PREAMBLE_MAX_CHARS = 1500  # предел преамбулы, если нумерованный раздел не найден

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6, "июля": 7,
    "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_MONTHS_RE = "|".join(MONTHS)
DATE_RE = re.compile(
    rf"(?P<d>\d{{1,2}})\.(?P<m>\d{{1,2}})\.(?P<y>\d{{4}})"
    rf"|(?P<d2>\d{{1,2}})\s+(?P<mon>{_MONTHS_RE})\s+(?P<y2>\d{{4}})(?:\s*г(?:ода|\.)?)?",
    re.IGNORECASE,
)
# Та же дата без именованных групп — для встраивания в другие выражения
DATE_PAT = rf"(?:\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|\d{{1,2}}\s+(?:{_MONTHS_RE})\s+\d{{4}}(?:\s*г(?:ода|\.)?)?)"
NUMBER_RE = re.compile(
    r"(?:№|\bN)\s*(?P<num>[A-Za-zА-Яа-яЁё0-9](?:(?:[A-Za-zА-Яа-яЁё0-9\-/.]|\s(?=\d))*[A-Za-zА-Яа-яЁё0-9])?)")
ORG_RE = re.compile(
    r"\b(?P<form>ООО|АО|ПАО|ЗАО|ОАО|НАО|ИП|ГУП|МУП|ФГУП|АНО|НКО)\s+[«\"](?P<name>[^»\"]+)[»\"]")
INN_RE = re.compile(r"ИНН\s*:?\s*(\d{10,12})")
ROLE_RE = re.compile(r"именуем\w*\s+(?:в\s+дальнейшем\s+)?[«\"]([^»\"]+)[»\"]|\(\s*[«\"]([^»\"]+)[»\"]\s*\)",
                     re.IGNORECASE)
SECTION_LINE_RE = re.compile(r"^\s*\d+\.\s+\S")

# Заголовки видов документов: проверяются по строкам шапки сверху вниз, первая сработавшая решает.
KIND_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("amendment", re.compile(r"^\s*ДОПОЛНИТЕЛЬНОЕ\s+СОГЛАШЕНИЕ\b|^\s*Дополнительное\s+соглашение\s*(№|N)")),
    ("contract", re.compile(r"^\s*ДОГОВОР\b|^\s*КОНТРАКТ\b|^\s*Договор\s+[а-яё\s]*(№|N)")),
    ("memo", re.compile(r"^\s*СЛУЖЕБНАЯ\s+ЗАПИСКА\b|^\s*Служебная\s+записка\b|^\s*ДОКЛАДНАЯ\s+ЗАПИСКА\b")),
    ("letter", re.compile(r"^\s*(Исх\.|Исходящий)\s*(№|N)|^\s*ПИСЬМО\b")),
    ("regulation", re.compile(r"^\s*(РЕГЛАМЕНТ|ПОЛОЖЕНИЕ|ПОРЯДОК|ИНСТРУКЦИЯ|ПОЛИТИКА|ПРАВИЛА|СТАНДАРТ)\b")),
]
LAW_SOURCE_RE = re.compile(r"^\s*Источник\s*:\s*(?P<src>.+)$", re.MULTILINE)
ARTICLE_RE = re.compile(r"Статья\s+(?P<num>\d+(?:\.\d+)*)\.?")


# ---------------- вспомогательное ----------------

def parse_date(text: str) -> str:
    """Первая дата в тексте → ISO; пустая строка, если даты нет."""
    m = DATE_RE.search(text or "")
    return _date_from_match(m) if m else ""


def _date_from_match(m: re.Match) -> str:
    if m.group("d"):
        d, mo, y = int(m.group("d")), int(m.group("m")), int(m.group("y"))
    else:
        d, mo, y = int(m.group("d2")), MONTHS[m.group("mon").lower()], int(m.group("y2"))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return ""
    return f"{y:04d}-{mo:02d}-{d:02d}"


def norm_number(number: str) -> str:
    """Номер документа без разделителей и регистра: «П-17/2024» = «п 17-2024» = «П17/2024»."""
    return re.sub(r"[\s\-/.\\_]", "", number or "").upper().replace("Ё", "Е")


LEGAL_FORMS = {"ООО", "АО", "ПАО", "ЗАО", "ОАО", "НАО", "ИП", "ГУП", "МУП", "ФГУП", "АНО", "НКО", "РФ"}


def smart_case(heading: str) -> str:
    """«ДОГОВОР ПОСТАВКИ № П-1/2024» → «Договор поставки № П-1/2024»; кавычки «», номера и формы
    собственности не трогаем."""
    parts = re.split(r"(«[^»]*»)", heading.strip())
    out = []
    for part in parts:
        if part.startswith("«"):
            out.append(part)
            continue
        words = []
        for w in part.split(" "):
            keep = w in LEGAL_FORMS or re.search(r"\d", w) or not w.isupper() or len(w) < 2
            words.append(w if keep else w.lower())
        out.append(" ".join(words))
    text = "".join(out).strip()
    return text[:1].upper() + text[1:] if text else text


def _header(text: str) -> list[str]:
    return [line for line in text.splitlines()[:HEADER_LINES * 2] if line.strip()][:HEADER_LINES]


def _stems(text: str, n: int = 6) -> set[str]:
    return {w[:n] for w in re.findall(r"[а-яёa-z]{4,}", text.lower().replace("ё", "е"))}


def detect_kind(text: str) -> str:
    if LAW_SOURCE_RE.search("\n".join(text.splitlines()[:3])):
        return "law"
    for line in _header(text):
        for kind, pattern in KIND_PATTERNS:
            if pattern.search(line):
                return kind
        if re.match(r"^\s*Статья\s+\d+", line):
            return "law"
    return "other"


def _preamble(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if i > 0 and SECTION_LINE_RE.match(line):
            return "\n".join(lines[:i])
    return text[:PREAMBLE_MAX_CHARS]


def extract_parties(text: str) -> list[tuple[str, str, str]]:
    """Стороны из преамбулы: (имя с формой, роль «в дальнейшем», ИНН). Повторы одной организации схлопываются."""
    pre = _preamble(text)
    matches = list(ORG_RE.finditer(pre))
    parties: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        tail = pre[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(pre)][:400]
        inn = INN_RE.search(tail)
        role = ROLE_RE.search(tail)
        name = f"{m.group('form')} «{m.group('name').strip()}»"
        role_text = (role.group(1) or role.group(2)).strip() if role else ""
        parties.append((name, role_text, inn.group(1) if inn else ""))
    merged: dict[str, tuple[str, str, str]] = {}
    for name, role, inn in parties:
        key = name.lower()
        old = merged.get(key)
        merged[key] = (name, role or (old[1] if old else ""), inn or (old[2] if old else ""))
    return list(merged.values())


def effective_from(text: str) -> str:
    """Дата начала действия требования из текста: «С 21 апреля 2025 г. …», «Действует с 01.03.2025»,
    «вступает в силу с …»."""
    pattern = re.compile(rf"(?:^|[.!?]\s+|\n)С\s+(?P<date>{DATE_PAT})"
                         rf"|(?:[Дд]ействует|вступает\s+в\s+силу|применяется)\s+с\s+(?P<date2>{DATE_PAT})",
                         re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return ""
    return parse_date(m.group("date") or m.group("date2"))


# ---------------- карточки по видам ----------------

def _parse_company_doc(raw: RawDoc, kind: str, card: DocCard, hints: list[str]) -> None:
    header = _header(raw.text)
    text = raw.text

    # Заголовок и номер: первая строка шапки, задающая вид документа
    title_line = next((line for line in header for k, p in KIND_PATTERNS if k == kind and p.search(line)), header[0])
    num = NUMBER_RE.search(title_line)
    card.number = num.group("num") if num else ""
    act = NUMBER_RE.split(title_line)[0] if num else title_line
    act = re.sub(r"\s+(от|№|N)\s*$", "", act.strip())
    card.act = smart_case(re.sub(r"\s+(ООО|АО|ПАО|ЗАО|ОАО)\s+«[^»]*»\s*$", "", act))
    if kind == "letter" and re.match(r"^(Исх|Исходящий)", card.act):
        card.act = "Письмо"

    # Дата документа
    if kind == "regulation":
        # Дата редакции: «Редакция от …» — первое упоминание, относящееся к этому документу
        red = re.search(rf"^\s*Редакция\s+от\s+(?P<date>{DATE_PAT})", "\n".join(header),
                        re.IGNORECASE | re.MULTILINE)
        card.date = parse_date(red.group("date")) if red else parse_date(title_line)
        if not card.date:
            approved = re.search(r"Утвержд\w+[^\n]*", "\n".join(header))
            card.date = parse_date(approved.group(0)) if approved else ""
    else:
        card.date = parse_date(title_line)
        if not card.date:
            # «г. Москва — 12 февраля 2024 г.» — строка с местом и датой; ссылки «к договору … от …» пропускаем
            for line in header:
                if line is title_line or re.match(r"^\s*к\s+", line, re.IGNORECASE):
                    continue
                card.date = parse_date(line)
                if card.date:
                    break

    # Заголовок по существу: для писем и записок — строка темы «О …»
    if kind in ("letter", "memo"):
        subject = next((line.strip() for line in header if re.match(r"^\s*О(б|бо)?\s+[а-яё]", line)), "")
        card.title = subject or card.act
    else:
        card.title = smart_case(title_line)

    # Стороны: для договоров и соглашений — из преамбулы; для внутренних документов — издатель
    parties = extract_parties(text)
    if kind in ("contract", "amendment"):
        card.parties = parties
    else:
        issuer = next((m for line in header for m in [ORG_RE.search(line)] if m), None)
        if issuer:
            card.parties = [(f"{issuer.group('form')} «{issuer.group('name').strip()}»", "издатель", "")]

    # Статус и даты действия из шапки
    head_text = "\n".join(header)
    card.status = "действует"
    for sentence in re.split(r"(?<=[.!?])\s+|\n", head_text):
        lost = re.search(r"утрат\w+\s+сил\w*\s+с\s+(?P<date>.+?\d{4})", sentence, re.IGNORECASE)
        if not lost:
            continue
        # «Редакция от <другая дата> утрачивает силу» — это о другой редакции, не о текущем документе
        other = re.search(r"[Рр]едакци\w+\s+от\s+(?P<d>.+?\d{4})", sentence[:lost.start()])
        if other and parse_date(other.group("d")) not in ("", card.date):
            continue
        card.status = "утратил силу"
        card.valid_to = parse_date(lost.group("date"))
        break
    card.valid_from = effective_from(head_text) or (card.date if kind in ("contract", "amendment") else "")

    # Подсказки для связей уровня документа
    for line in header:
        if re.match(r"^\s*к\s+(договору|контракту|соглашению)\b", line, re.IGNORECASE):
            hints.append(line.strip())
        if re.search(r"утрат\w+\s+сил", line, re.IGNORECASE):
            hints.append(line.strip())
    for line in text.splitlines():
        if re.search(r"\bв\s+дополнение\s+к\b", line, re.IGNORECASE):
            hints.append(line.strip())


def _parse_law(raw: RawDoc, card: DocCard) -> None:
    text = raw.text
    src_m = LAW_SOURCE_RE.search(text)
    source = src_m.group("src").strip() if src_m else ""
    url = re.search(r"^\s*URL\s*:\s*(\S+)", text, re.MULTILINE)
    exported = re.search(r"^\s*Дата\s+выгрузки\s*:\s*(.+)$", text, re.MULTILINE)
    card.source_url = url.group(1) if url else ""
    card.export_date = parse_date(exported.group(1)) if exported else ""
    card.status = "действует"

    art = ARTICLE_RE.search(source) or ARTICLE_RE.search(text)
    card.number = art.group("num") if art else ""

    # Акт: аббревиатура «(ТК РФ)», иначе номер федерального закона «152-ФЗ»
    abbr = re.search(r"\(([А-ЯЁ][А-Яа-яЁё]{0,5}\s?РФ)\)", source)
    fz = re.search(r"Федеральн\w+\s+закон\w*\s+от\s+[\d.]+\s+(?:№|N)\s*(\d+-ФЗ)(?:\s+[\"«]([^\"»]+)[\"»])?", source)
    if abbr:
        card.act = abbr.group(1).replace(" ", " ")
    elif fz:
        card.act = fz.group(1)
    else:
        card.act = ""
    # Полное название акта: «Трудовой кодекс Российской Федерации», «Кодекс Российской Федерации об …»,
    # для федерального закона — его название в кавычках
    full = re.search(r"((?:[А-ЯЁ][а-яё]+\s+)?[Кк]одекс\s+Российской\s+Федерации(?:\s+об?\s+[^()]+?)?)"
                     r"\s*(?:\bот\b|\(|$)", source)
    if full:
        card.act_full = full.group(1).strip()
    elif fz and fz.group(2):
        card.act_full = f"Федеральный закон «{fz.group(2)}»"
    act_date = re.search(r"\bот\s+(\d{2}\.\d{2}\.\d{4})", source)
    card.date = parse_date(act_date.group(1)) if act_date else ""

    # Название статьи — из строки «Источник» между «Статья N.» и названием акта
    title = ""
    if art and art.re is ARTICLE_RE and source:
        rest = source[art.end():].strip()
        rest = re.sub(r"^[А-ЯЁ][А-Яа-яЁё]{0,5}\s?РФ\s+", "", rest)
        act_start = re.search(r"Федеральн\w+\s+закон|[А-ЯЁ][а-яё]+\s+кодекс|Кодекс\s", rest)
        title = (rest[:act_start.start()] if act_start else rest).strip(" .")
    if not title and fz and fz.group(2):
        title = fz.group(2)
    card.title = title


def parse(raw: RawDoc, settings=None) -> ParsedDoc:
    """Карточка документа, подсказки связей и дерево узлов (T1.2, T1.3).

    settings — объект с полями child_min_chars, child_max_chars, split_overlap (config.Settings);
    без него — стартовые значения 06-implementation-guide §7."""
    kind = detect_kind(raw.text)
    card = DocCard(doc_id=f"d{raw.sha256[:10]}", file=raw.name, sha256=raw.sha256, kind=kind,
                   chars=len(raw.text))
    hints: list[str] = []
    if kind == "law":
        _parse_law(raw, card)
    else:
        _parse_company_doc(raw, kind, card, hints)
    nodes, warnings = build_nodes(
        card, raw.text,
        child_min_chars=getattr(settings, "child_min_chars", 200),
        child_max_chars=getattr(settings, "child_max_chars", 1200),
        split_overlap=getattr(settings, "split_overlap", 0.12))
    if getattr(settings, "use_contextual", False):
        # T3.38: офлайн-контексты узлов (contextual_cache.json в архиве); применяются
        # только при совпадении полного SHA документа и текста конкретного узла.
        from contextual import apply_contexts, load_contexts
        apply_contexts(nodes, load_contexts(), card.sha256)
    return ParsedDoc(card=card, nodes=nodes, link_hints=hints, text=raw.text, warnings=warnings)


# ---------------- связи уровня документа ----------------

def _regulation_key(card: DocCard) -> set[str]:
    return _stems(card.act)


def link_documents(parsed: list[ParsedDoc]) -> list[Relation]:
    """Связи: доп. соглашение → договор (amends), новая редакция → старая (replaces),
    письмо/записка → регламент (supplements)."""
    cards = [p.card for p in parsed]
    relations: list[Relation] = []

    # amends: «к договору … № X» → договор с тем же номером
    contracts = [c for c in cards if c.kind == "contract"]
    for p in parsed:
        if p.card.kind != "amendment":
            continue
        for hint in p.link_hints:
            m = re.match(r"^\s*к\s+(договору|контракту|соглашению)\b", hint, re.IGNORECASE)
            num = NUMBER_RE.search(hint) if m else None
            if not num:
                continue
            target = next((c for c in contracts if c.number and norm_number(c.number) == norm_number(num.group("num"))),
                          None)
            if target:
                cond = re.search(r"[^.\n]*\b(применя\w+|распростран\w+)\s+(к|на)\b[^.\n]*", p.text)
                relations.append(Relation(p.card.doc_id, target.doc_id, "amends", p.card.valid_from or p.card.date,
                                          cond.group(0).strip() if cond else ""))

    # replaces: редакции одного внутреннего документа — более новая заменяет предыдущую
    regs = [c for c in cards if c.kind == "regulation" and c.date]
    groups: dict[frozenset, list[DocCard]] = {}
    for c in regs:
        groups.setdefault(frozenset(_regulation_key(c)), []).append(c)
    for group in groups.values():
        group.sort(key=lambda c: c.date)
        for old, new in zip(group, group[1:]):
            relations.append(Relation(new.doc_id, old.doc_id, "replaces", new.valid_from or old.valid_to or new.date))

    # supplements: «в дополнение к … <название регламента> (ред. от …)»
    for p in parsed:
        if p.card.kind not in ("letter", "memo"):
            continue
        for hint in p.link_hints:
            if not re.search(r"\bв\s+дополнение\s+к\b", hint, re.IGNORECASE):
                continue
            hint_stems = _stems(hint)
            candidates = [c for c in regs if _regulation_key(c) and _regulation_key(c) <= hint_stems]
            red = re.search(r"ред\w*\.?\s+от\s+([^)]+\d{4})", hint)
            if red:
                red_date = parse_date(red.group(1))
                dated = [c for c in candidates if c.date == red_date]
                candidates = dated or candidates
            if not candidates:
                continue
            target = max(candidates, key=lambda c: c.date)
            date = effective_from(p.text) or p.card.date
            relations.append(Relation(p.card.doc_id, target.doc_id, "supplements", date, hint))
    return relations


def parse_all(raw_docs: list[RawDoc], settings=None) -> tuple[list[ParsedDoc], list[Relation]]:
    """Разбор всей базы, связи между документами и темпоральная разметка уровня документа."""
    parsed = [parse(raw, settings) for raw in raw_docs]
    relations = link_documents(parsed)
    apply_temporal(parsed, relations)
    return parsed, relations


# ---------------- темпоральная разметка уровня документа (T1.7) ----------------

def _next_day(iso: str) -> str:
    from datetime import date, timedelta
    try:
        y, m, d = (int(x) for x in iso.split("-"))
        return (date(y, m, d) + timedelta(days=1)).isoformat()
    except (ValueError, AttributeError):
        return ""


def _sentence_with(text: str, pattern: str) -> str:
    """Предложение (без номера пункта), в котором встречается шаблон."""
    for sentence in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])|\n", text):
        if re.search(pattern, sentence, re.IGNORECASE):
            return re.sub(r"^\s*\d+(?:\.\d+)*\.\s*", "", sentence).strip()
    return ""


def apply_temporal(parsed: list[ParsedDoc], relations: list[Relation]) -> None:
    """Даты действия и условия применения документов компании (03a §4.7, уровень документа).

    - договор: начало — «вступает в силу с …» или дата заключения; конец — «действует до …»; при
      автоматической пролонгации или бессрочности конец не ставится, условие хранится текстом
      (срок «до … с автоматическим продлением» не означает, что договор истёк — 06 §10);
    - доп. соглашение и письмо: начало действия и условие применения («применяется к …», «до внесения …»);
    - редакции: старая теряет силу с даты начала новой, если это не сказано явно;
    - прочие внутренние документы: действуют с даты документа, если иное не сказано.
    Законы — только действующие редакции (Q1), даты из пометок не используются."""
    by_id = {p.card.doc_id: p for p in parsed}
    for p in parsed:
        c, text = p.card, p.text
        if c.kind == "law":
            continue
        start = re.search(rf"вступает\s+в\s+силу\s+с\s+(?:момента\s+|даты\s+)?(?P<what>подписания|{DATE_PAT})",
                          text, re.IGNORECASE)
        if start and start.group("what").lower() != "подписания":
            c.valid_from = parse_date(start.group("what"))
        if not c.valid_from:
            c.valid_from = c.date

        if c.kind == "contract":
            prolong = _sentence_with(text, r"автоматически\s+(продлева|пролонгир)|пролонгир\w+|продлевается")
            indefinite = _sentence_with(text, r"на\s+неопредел\w+\s+срок|бессрочн")
            until = re.search(rf"действует\s+(?:до|по)\s+(?P<date>{DATE_PAT})", text, re.IGNORECASE)
            if prolong:
                c.condition = prolong
            elif indefinite:
                c.condition = indefinite
            elif until:
                c.valid_to = _next_day(parse_date(until.group("date")))
                c.condition = _sentence_with(text, r"действует\s+(?:до|по)\s")
        elif c.kind in ("amendment", "letter", "memo"):
            # Только глагольные формы: «применяется к …», не причастие «применяемых к …»
            cond = (_sentence_with(text, r"\b(применя(ется|ются)|распространя(ется|ются))\s+(к|на)\b")
                    or _sentence_with(text, r"\b(применя(ется|ются)|распространя(ется|ются))\s+в\s+дополнение\b")
                    or _sentence_with(text, r"\bдо\s+внесения\b"))
            if cond:
                c.condition = cond

    # Редакции: заменённая теряет силу с начала действия новой (если не указано явно)
    for r in relations:
        if r.kind != "replaces":
            continue
        new, old = by_id[r.src_doc].card, by_id[r.dst_doc].card
        if not old.valid_to and (new.valid_from or new.date):
            old.valid_to = new.valid_from or new.date
            old.status = "утратил силу"


# ---------------- каталог сущностей (T1.6) ----------------

def _org_key(name: str) -> str:
    """Ключ организации: название в кавычках без регистра и ё; форма собственности не учитывается."""
    m = re.search(r"«([^»]+)»", name)
    core = m.group(1) if m else name
    return re.sub(r"\s+", " ", core).strip().lower().replace("ё", "е")


def _better_name(a: str, b: str) -> str:
    """Из двух написаний одной организации предпочитаем не набранное капсом."""
    def caps(s: str) -> bool:
        inner = re.search(r"«([^»]+)»", s)
        s = inner.group(1) if inner else s
        letters = [ch for ch in s if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters)
    if caps(a) and not caps(b):
        return b
    return a


def build_entities(parsed: list[ParsedDoc]) -> list[Entity]:
    """Каталог сущностей кодом из карточек (03a §4.5): организации (стороны и издатели, ИНН, роли),
    документы компании (вид, номер) и акты законодательства (сокращение, полное название).
    «Своя» организация — сторона большинства документов компании (C1: имя в код не вписывается)."""
    entities: list[Entity] = []

    # Организации
    orgs: dict[str, Entity] = {}
    for p in parsed:
        for name, role, inn in p.card.parties:
            key = _org_key(name)
            if inn and key in orgs and orgs[key].inn and orgs[key].inn != inn:
                key = f"{key}#{inn}"
            ent = orgs.get(key)
            if ent is None:
                ent = Entity(entity_id=f"org:{len(orgs) + 1}", kind="org", canonical=name, inn=inn)
                orgs[key] = ent
            ent.canonical = _better_name(ent.canonical, name)
            ent.inn = ent.inn or inn
            if p.card.doc_id not in ent.doc_ids:
                ent.doc_ids.append(p.card.doc_id)
            ent.roles[p.card.doc_id] = role
    company_docs = [p for p in parsed if p.card.kind != "law"]
    if orgs and len(company_docs) >= 2:
        top = max(orgs.values(), key=lambda e: len(e.doc_ids))
        if len(top.doc_ids) >= max(2, (len(company_docs) + 1) // 2):
            top.own = True
    for ent in orgs.values():
        inner = re.search(r"«([^»]+)»", ent.canonical)
        ent.aliases = sorted({ent.canonical, inner.group(1) if inner else ent.canonical})
        entities.append(ent)

    # Документы компании
    for p in company_docs:
        c = p.card
        # Номер — алиас, только если он различим сам по себе (буквы или разделители, не «1»)
        distinct_number = c.number if re.search(r"[A-Za-zА-Яа-яЁё]|[-/.]", c.number or "") else ""
        aliases = {a for a in (c.act, c.title, f"{c.act} № {c.number}" if c.number else "", distinct_number) if a}
        entities.append(Entity(entity_id=f"doc:{c.doc_id}", kind="document",
                               canonical=f"{c.act} № {c.number}".strip(" №") if c.number else c.act or c.title,
                               aliases=sorted(aliases), doc_ids=[c.doc_id]))

    # Акты законодательства: одна сущность на акт, doc_ids — его статьи
    acts: dict[str, Entity] = {}
    for p in parsed:
        c = p.card
        if c.kind != "law" or not c.act:
            continue
        ent = acts.get(c.act)
        if ent is None:
            ent = Entity(entity_id=f"act:{len(acts) + 1}", kind="act", canonical=c.act)
            acts[c.act] = ent
        ent.doc_ids.append(c.doc_id)
        names = {c.act, re.sub(r"\s*РФ$", "", c.act)}
        if c.act_full:
            names.add(c.act_full)
            words = c.act_full.split()
            if len(words) > 2 and words[1].lower().startswith("кодекс"):
                names.add(" ".join(words[:2]))           # «Трудовой кодекс»
            quoted = re.search(r"«([^»]+)»", c.act_full)
            if quoted:
                names.add(f"Закон «{quoted.group(1)}»")
                names.add(quoted.group(1))
        m = re.fullmatch(r"(\d+)-ФЗ", c.act)
        if m:
            names.update({f"ФЗ-{m.group(1)}", f"{m.group(1)} ФЗ"})
        ent.aliases = sorted(set(ent.aliases) | {n for n in names if n})
    entities.extend(acts.values())
    return entities
