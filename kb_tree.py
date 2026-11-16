"""Иерархическое разбиение документа на узлы (T1.3; 03a §2; KB-5, KB-6; U-7).

Публичный вход — kb_parser.parse(); этот модуль — его внутренняя часть.

Документы компании: раздел «N.» → пункт «N.M.» → подпункт «N.M.K.»; строки без номера дописываются
к предыдущему узлу; текст до первого раздела — узел «преамбула»; документ без нумерации — абзацы.

Законы (статья на файл, формат выгрузки правовой системы):
- нумерованные единицы «N.», «1.1.», «1.2-1.» — пункты (НК, ГК, БК) или части (прочие акты);
- «N)» — подпункты (НК, ГК, БК) или пункты (прочие акты);
- если нумерованных единиц нет (ТК), части считаются по абзацам: новая часть — абзац с заглавной буквы,
  абзац со строчной — продолжение перечисления текущей части (KB-6, «ст. 59, часть первая»);
- редакционные пометки («Часть 1 изменена с …», «Федеральным законом … внесены изменения», «… см. …»)
  выносятся в Node.notes и в текст нормы не входят (Q9);
- заглушки «утратил силу» / «исключён» остаются узлами (занимают номер) с флагом repealed.

Ссылка (label) у абзаца — ссылка его родителя: номер абзаца в ссылку не пишем, чтобы не выдать
неточное положение за точное (A2).

Длинные узлы (> child_max_chars) дробятся по предложениям с перекрытием (флаги split/fragment);
короткие соседние абзацы с одинаковой ссылкой объединяются до child_max_chars.
Вводная фраза (lead_in) — цепочка текстов родителей, оканчивающихся двоеточием (или если узел
продолжает фразу родителя со строчной буквы).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kb_types import DocCard, Node

POINT_ACTS = {"НК РФ", "ГК РФ", "БК РФ"}  # кодексы, где нумерованные единицы статьи — пункты
LEAD_IN_MAX = 600

LAW_HEADER_RE = re.compile(r"^\s*(Источник|URL|Дата\s+выгрузки)\s*:", re.IGNORECASE)
ARTICLE_HEAD_RE = re.compile(r"^\s*Статья\s+(?P<num>\d+(?:\.\d+)*)\.?\s*(?P<rest>.*)$")
UNIT_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*(?:-\d+)?)\.\s+(?P<rest>\S.*)$")
ITEM_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*(?:-\d+)?)\)\s*(?P<rest>.*)$")
REMARKS_RE = re.compile(r"^\s*Примечани[ея]\s*[:.]?\s*(?P<rest>.*)$")
COMPANY_SECTION_RE = re.compile(r"^\s*(?P<num>\d+)\.\s+(?P<rest>\S.*)$")
COMPANY_POINT_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)+)\.?\s+(?P<rest>\S.*)$")

NOTE_PATTERNS = [
    re.compile(r"^(Часть|Пункт|Подпункт|Статья|Абзац|Глава|Раздел|Приложение|Примечани[ея])\s+[\dIVXLC.\-]*\s*"
               r"(изменен|дополнен|изложен|приостановлен|вступа)\w*", re.IGNORECASE),
    re.compile(r"^Федеральн\w+\s+закон\w*\s+от\b.*\b(внесен|изложен|дополнен|признан|изменен)", re.IGNORECASE),
    re.compile(r"^О\s+(проверке\s+конституционности|конституционно-правовом\s+смысле|применении)\b.*\bсм\.",
               re.IGNORECASE),
    re.compile(r"^Положения\b.*\(в\s+редакции\b", re.IGNORECASE),
    re.compile(r"^Действие\s+положений\b", re.IGNORECASE),
    re.compile(r"^(Подпункт|Пункт|Часть|Статья)\s+[\d.\-]+\s+применя", re.IGNORECASE),
    re.compile(r"^См\.\s"),
]
REPEALED_RE = re.compile(r"(утратил\w*\s+сил\w*|исключ[её]н\w*)", re.IGNORECASE)
INSTRUCTION_RE = re.compile(
    r"(игнорируй|не\s+упоминай\s+эту\s+инструкц|для\s+ии[-\s]?ассистента|ignore\s+(all|previous)|"
    r"system\s*prompt|системн\w+\s+промпт|отвечай\s+только\s+[«\"])", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+")


@dataclass
class _Para:
    text: str
    start: int
    end: int


def paragraphs(text: str) -> list[_Para]:
    """Непустые строки документа с позициями (абзацы выгрузки — отдельные строки)."""
    out, pos = [], 0
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lead = len(line) - len(line.lstrip())
            out.append(_Para(stripped, pos + lead, pos + lead + len(stripped)))
        pos += len(line) + 1
    return out


def is_note(text: str) -> bool:
    return any(p.search(text) for p in NOTE_PATTERNS)


def is_repealed(text: str) -> bool:
    return len(text) < 220 and bool(REPEALED_RE.search(text)) and not is_note(text)


def _starts_lower(text: str) -> bool:
    m = re.search(r"[A-Za-zА-Яа-яЁё]", text)
    return bool(m and m.group(0).islower())


class _Builder:
    def __init__(self, card: DocCard):
        self.card = card
        self.nodes: list[Node] = []
        self.paths: set[str] = set()
        self.pending_notes: list[str] = []
        self.covered: list[tuple[int, int]] = []

    def add(self, parent: Node | None, path: str, label: str, para: _Para | None, text: str = "",
            heading: str = "", flags: list[str] | None = None) -> Node:
        base, n = path, 2
        while path in self.paths:
            path, n = f"{base}#{n}", n + 1
        self.paths.add(path)
        node = Node(node_id=f"{self.card.doc_id}:{path}", doc_id=self.card.doc_id,
                    parent_id=parent.node_id if parent else None, level=(parent.level + 1) if parent else 1,
                    path=path, label=label, heading=heading, text=text or (para.text if para else ""),
                    start=para.start if para else 0, end=para.end if para else 0, flags=list(flags or []))
        if self.pending_notes:
            node.notes.extend(self.pending_notes)
            node.flags.append("editorial_note")
            self.pending_notes = []
        if para:
            self.covered.append((para.start, para.end))
        if INSTRUCTION_RE.search(node.text):
            node.flags.append("instruction_like")
        self.nodes.append(node)
        return node

    def append(self, node: Node, para: _Para) -> None:
        """Продолжение узла (строка без номера в документе компании)."""
        node.text = f"{node.text}\n{para.text}" if node.text else para.text
        node.end = para.end
        self.covered.append((para.start, para.end))
        if INSTRUCTION_RE.search(para.text) and "instruction_like" not in node.flags:
            node.flags.append("instruction_like")

    def note(self, para: _Para) -> None:
        self.pending_notes.append(para.text)
        self.covered.append((para.start, para.end))


# ---------------- законы ----------------

def _law_nodes(b: _Builder, text: str) -> list[str]:
    warnings: list[str] = []
    paras = paragraphs(text)
    act = b.card.act
    unit_word, item_word = ("п.", "пп.") if act in POINT_ACTS else ("ч.", "п.")
    unit_key, item_key = ("п", "пп") if act in POINT_ACTS else ("ч", "п")
    numbered = any(UNIT_RE.match(p.text) for p in paras if not ARTICLE_HEAD_RE.match(p.text))

    article: Node | None = None
    unit: Node | None = None
    item: Node | None = None
    remarks: Node | None = None   # «Примечания:» к статье — свои пронумерованные примечания
    part_no = 0
    abz_count: dict[str, int] = {}

    def abzac(container: Node, para: _Para) -> Node:
        k = abz_count.get(container.node_id, 1) + 1
        abz_count[container.node_id] = k
        flags = ["repealed"] if is_repealed(para.text) else []
        return b.add(container, f"{container.path}/абз.{k}", container.label, para, flags=flags)

    def new_part(para: _Para | None, text: str = "", start: int = 0) -> Node:
        nonlocal part_no
        part_no += 1
        flags = ["repealed"] if para and is_repealed(para.text) else []
        node = b.add(article, f"{article.path}/ч.{part_no}", f"{article.label}, ч. {part_no}", para,
                      text=text, flags=flags)
        if not para:
            node.start, node.end = start, start + len(text)
        return node

    for para in paras:
        t = para.text
        if article is None:
            if LAW_HEADER_RE.match(t):
                b.covered.append((para.start, para.end))
                continue
            head = ARTICLE_HEAD_RE.match(t)
            if head and not is_note(t):
                num, rest = head.group("num"), head.group("rest").strip()
                is_norm_text = rest.endswith(".") and len(rest.split()) >= 8
                article = b.add(None, f"ст.{num}", f"ст. {num}", para, text=t,
                                heading="" if is_norm_text else rest)
                if is_norm_text and not numbered:
                    # Заголовок сразу содержит норму (часть первая): «Статья 1. Установить …»
                    offset = para.start + t.index(rest)
                    unit = new_part(None, text=rest, start=offset)
                continue
            b.note(para)  # вводные пометки до заголовка статьи
            continue

        if is_note(t):
            b.note(para)
            continue

        m_remarks = REMARKS_RE.match(t)
        if m_remarks:
            remarks = b.add(article, f"{article.path}/прим", f"{article.label}, примечание", para)
            unit, item = remarks, None
            continue
        if remarks is not None:
            m_unit = UNIT_RE.match(t)
            if m_unit:
                num = m_unit.group("num")
                unit = b.add(remarks, f"{remarks.path}/{num}", f"{article.label}, прим. {num}", para)
                item = None
            else:
                abzac(unit, para)
            continue

        if numbered:
            m_unit, m_item = UNIT_RE.match(t), ITEM_RE.match(t)
            if m_unit:
                num = m_unit.group("num")
                unit = b.add(article, f"{article.path}/{unit_key}.{num}", f"{article.label}, {unit_word} {num}", para,
                             flags=["repealed"] if is_repealed(t) else [])
                item = None
            elif m_item:
                parent = unit or article
                num = m_item.group("num")
                item = b.add(parent, f"{parent.path}/{item_key}.{num}", f"{parent.label}, {item_word} {num}", para,
                             flags=["repealed"] if is_repealed(t) else [])
            elif unit is None and item is None and len(t) < 120 and not re.search(r"[.;:]$", t):
                b.note(para)  # служебная строка между заголовком и первой единицей (ссылка правовой системы)
            else:
                abzac(item or unit or article, para)
        else:
            m_item = ITEM_RE.match(t)
            if m_item:
                parent = unit or new_part(None, text="", start=para.start)
                num = m_item.group("num")
                item = b.add(parent, f"{parent.path}/{item_key}.{num}", f"{parent.label}, {item_word} {num}", para,
                             flags=["repealed"] if is_repealed(t) else [])
            elif unit is not None and _starts_lower(t):
                abzac(item or unit, para)  # продолжение перечисления текущей части
            else:
                unit, item = new_part(para), None

    if article is None:
        warnings.append("Не найден заголовок статьи — документ разобран по абзацам")
        _plain_nodes(b, text)
    if b.pending_notes and b.nodes:
        b.nodes[-1].notes.extend(b.pending_notes)
        b.pending_notes = []
    return warnings


# ---------------- документы компании ----------------

def _plain_nodes(b: _Builder, text: str) -> None:
    for i, para in enumerate(paragraphs(text), 1):
        b.add(None, f"абз.{i}", "", para)


def _company_nodes(b: _Builder, text: str) -> list[str]:
    paras = paragraphs(text)
    first = next((i for i, p in enumerate(paras) if i > 0 and COMPANY_SECTION_RE.match(p.text)
                  and not COMPANY_POINT_RE.match(p.text)), None)
    if first is None:
        _plain_nodes(b, text)
        return []
    preamble = b.add(None, "преамбула", "", paras[0])
    for para in paras[1:first]:
        b.append(preamble, para)
    by_path: dict[str, Node] = {}
    current: Node = preamble
    for para in paras[first:]:
        t = para.text
        m_point = COMPANY_POINT_RE.match(t)
        m_section = COMPANY_SECTION_RE.match(t) if not m_point else None
        if m_section:
            num, rest = m_section.group("num"), m_section.group("rest").strip()
            heading = rest if len(rest) <= 120 and not rest.endswith(".") else ""
            current = b.add(None, num, f"раздел {num}", para, heading=heading)
            by_path[num] = current
        elif m_point:
            num = m_point.group("num")
            parent_path = num.rsplit(".", 1)[0]
            parent = by_path.get(parent_path)
            current = b.add(parent, num, f"п. {num}", para)
            by_path[num] = current
        else:
            b.append(current, para)
    return []


# ---------------- постобработка ----------------

def _split_text(text: str, max_chars: int, overlap: float) -> list[str]:
    """Дробление по предложениям (затем по словам) с перекрытием."""
    units: list[str] = []
    for sent in SENTENCE_SPLIT_RE.split(text):
        if len(sent) <= max_chars:
            units.append(sent)
            continue
        words, cur = sent.split(" "), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                units.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}" if cur else w
        if cur:
            units.append(cur)
    chunks, cur = [], []
    for u in units:
        if cur and len(" ".join(cur)) + 1 + len(u) > max_chars:
            chunks.append(" ".join(cur))
            keep, size = [], 0
            for prev in reversed(cur):
                if size + len(prev) > overlap * max_chars:
                    break
                keep.insert(0, prev)
                size += len(prev) + 1
            while keep and len(" ".join(keep)) + 1 + len(u) > max_chars:
                keep.pop(0)
            cur = keep
        cur.append(u)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_long(b: _Builder, max_chars: int, overlap: float) -> None:
    for node in list(b.nodes):
        if len(node.text) <= max_chars:
            continue
        node.flags.append("split")
        pos = node.start
        for i, chunk in enumerate(_split_text(node.text, max_chars, overlap), 1):
            frag = Node(node_id=f"{node.node_id}/ф{i}", doc_id=node.doc_id, parent_id=node.node_id,
                        level=node.level + 1, path=f"{node.path}/ф{i}", label=node.label, text=chunk,
                        flags=["fragment"])
            found = node.text.find(chunk[:40])
            frag.start = node.start + found if found >= 0 else pos
            frag.end = frag.start + len(chunk)
            pos = frag.end
            b.nodes.append(frag)


def _merge_small(b: _Builder, min_chars: int, max_chars: int) -> None:
    """Объединяет подряд идущие короткие листья одного родителя с одинаковой ссылкой."""
    has_children = {n.parent_id for n in b.nodes if n.parent_id}
    no_merge = {"fragment", "instruction_like", "repealed"}

    def mergeable(n: Node) -> bool:
        return n.node_id not in has_children and not (set(n.flags) & no_merge) and len(n.text) < min_chars

    result: list[Node] = []
    i, nodes = 0, b.nodes
    while i < len(nodes):
        node = nodes[i]
        group = [node]
        if mergeable(node):
            j = i + 1
            while j < len(nodes):
                nxt = nodes[j]
                size = sum(len(x.text) + 1 for x in group) + len(nxt.text)
                if (nxt.parent_id == node.parent_id and nxt.label == node.label and mergeable(nxt)
                        and size <= max_chars):
                    group.append(nxt)
                    j += 1
                else:
                    break
        if len(group) > 1:
            last = group[-1].path.rsplit("/", 1)[-1]
            merged = Node(node_id=f"{node.node_id}-{last}", doc_id=node.doc_id, parent_id=node.parent_id,
                          level=node.level, path=f"{node.path}-{last}", label=node.label,
                          text="\n".join(g.text for g in group), start=group[0].start, end=group[-1].end,
                          flags=sorted({f for g in group for f in g.flags} | {"merged"}),
                          notes=[x for g in group for x in g.notes])
            result.append(merged)
            i += len(group)
        else:
            result.append(node)
            i += 1
    b.nodes = result


def _chain(*parts: str) -> str:
    chain = " ".join(p.strip() for p in parts if p and p.strip())
    return chain if len(chain) <= LEAD_IN_MAX else "…" + chain[-LEAD_IN_MAX:]


def _assign_lead_in(nodes: list[Node]) -> None:
    """Вводная фраза: текст родителя, оканчивающийся двоеточием, или — для строчного абзаца — ближайший
    предыдущий соседний абзац с двоеточием, который открыл перечисление (по цепочке вверх)."""
    by_id = {n.node_id: n for n in nodes}
    open_list: dict[str, str] = {}  # parent_id -> текст соседа, открывшего перечисление «…:»
    for node in nodes:  # родители идут раньше детей
        parent = by_id.get(node.parent_id) if node.parent_id else None
        if parent is None:
            continue
        if "fragment" in node.flags:
            node.lead_in = parent.lead_in
            continue
        own = parent.text.split("\n")[0].strip() if parent.text and "split" not in parent.flags else ""
        base = _chain(parent.lead_in, own) if own.endswith(":") else parent.lead_in
        lower = _starts_lower(node.text)
        if lower and open_list.get(parent.node_id):
            node.lead_in = _chain(base, open_list[parent.node_id])
        elif lower and own:
            node.lead_in = _chain(parent.lead_in, own)
        else:
            node.lead_in = base
        last_line = node.text.rstrip().split("\n")[-1]
        if last_line.endswith(":"):
            open_list[parent.node_id] = node.text.strip()
        elif not lower:
            open_list.pop(parent.node_id, None)


def _validate(b: _Builder, text: str, kind: str) -> list[str]:
    warnings: list[str] = []
    # Потери текста: каждый непробельный символ должен попасть в узел, пометку или шапку
    covered = bytearray(len(text))
    for start, end in b.covered:
        covered[start:end] = b"\x01" * (end - start)
    lost = sum(1 for i, ch in enumerate(text) if not ch.isspace() and not covered[i])
    if lost:
        warnings.append(f"Потеряно символов текста: {lost}")
    for node in b.nodes:
        if not node.text.strip() and "fragment" not in node.flags and not any(
                n.parent_id == node.node_id for n in b.nodes):
            warnings.append(f"Пустой узел {node.path}")
    # Непрерывность нумерации
    groups: dict[tuple[str | None, str], list[str]] = {}
    for node in b.nodes:
        if "fragment" in node.flags or "merged" in node.flags:
            continue
        last = node.path.rsplit("/", 1)[-1].split("#")[0]
        m = re.match(r"^(?:п|пп|ч)\.(\d+)$", last) if kind == "law" else re.match(r"^(?:\d+\.)*(\d+)$", last)
        if m:
            key = "ч" if last.startswith("ч.") else last.split(".")[0] if kind == "law" else "n"
            groups.setdefault((node.parent_id, key), []).append(m.group(1))
    for (parent, _), nums in groups.items():
        ints = [int(x) for x in nums]
        expected = list(range(ints[0], ints[0] + len(ints)))
        if ints != expected:
            warnings.append(f"Нарушена нумерация в {parent or 'корне'}: {ints}")
    return warnings


def build_nodes(card: DocCard, text: str, child_min_chars: int = 200, child_max_chars: int = 1200,
                split_overlap: float = 0.12) -> tuple[list[Node], list[str]]:
    b = _Builder(card)
    warnings = _law_nodes(b, text) if card.kind == "law" else _company_nodes(b, text)
    if b.pending_notes and b.nodes:
        b.nodes[-1].notes.extend(b.pending_notes)
    warnings += _validate(b, text, card.kind)
    _merge_small(b, child_min_chars, child_max_chars)
    _split_long(b, child_max_chars, split_overlap)
    _assign_lead_in(b.nodes)
    from chunk_provenance import attach_provenance
    attach_provenance(card, text, b.nodes)
    return b.nodes, warnings
