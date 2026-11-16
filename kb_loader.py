"""Поиск входных файлов и чтение базы знаний (T1.1; ST-2, ST-3, Q4a).

- Архив базы под любым из трёх имён: knowledge-base.zip (регламент), knowledge_base.zip (сайт), kb.zip (стартер).
- Форматы TXT / MD / DOCX; кодировка текста определяется (BOM UTF-16/UTF-8, UTF-8, CP1251).
- Имена файлов в архиве без флага UTF-8 перекодируются (CP437 → UTF-8 / CP866 / CP1251).
- DOCX читается стандартной библиотекой: абзацы и таблицы в порядке документа, автонумерация Word
  восстанавливается из numbering.xml (иначе пропали бы номера пунктов «3.2.»).
- Хэш каждого документа — по нормализованному тексту (для сверки с предрасчитанным индексом, Q4a).
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from kb_types import RawDoc

KB_ARCHIVE_NAMES = ("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
QUESTIONS_NAME = "questions.csv"
SUPPORTED_SUFFIXES = {".txt", ".md", ".docx"}

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
_UTF8_FLAG = 0x800


def _warn(msg: str) -> None:
    print(f"[kb_loader] {msg}", file=sys.stderr, flush=True)


# ---------------- входные файлы ----------------

def _search_roots(data_dir: Path) -> list[Path]:
    roots = [data_dir, Path(__file__).resolve().parent / data_dir, Path.cwd() / data_dir, Path.cwd()]
    seen, out = set(), []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def find_inputs(data_dir: Path | str) -> tuple[Path, Path]:
    """Пути к questions.csv и архиву базы. Ищем в DATA_DIR, рядом с кодом и в рабочем каталоге."""
    data_dir = Path(data_dir)
    roots = _search_roots(data_dir)
    questions = next((r / QUESTIONS_NAME for r in roots if (r / QUESTIONS_NAME).is_file()), None)
    kb = next((r / n for r in roots for n in KB_ARCHIVE_NAMES if (r / n).is_file()), None)
    if questions is None:
        raise FileNotFoundError(f"Не найден {QUESTIONS_NAME} в {data_dir}")
    if kb is None:
        raise FileNotFoundError(f"Не найден архив базы ({', '.join(KB_ARCHIVE_NAMES)}) в {data_dir}")
    return questions, kb


def load_questions(path: Path | str) -> list[tuple[str, str]]:
    """Список (id, текст вопроса). Текст сохраняется как есть, включая историю диалога."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {(name or "").strip() for name in (reader.fieldnames or [])}
        if not {"id", "question"} <= fields:
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = []
        for row in reader:
            row = {(k or "").strip(): v for k, v in row.items()}
            qid = str(row.get("id") or "").strip()
            if not qid:
                continue
            rows.append((qid, str(row.get("question") or "").strip()))
    if not rows:
        raise ValueError("В questions.csv нет вопросов")
    return rows


# ---------------- текст ----------------

def normalize_text(text: str) -> str:
    """Каноническая форма текста документа: \\n, без BOM/нулевой ширины, неразрывный пробел → пробел."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace(" ", " ")
    text = _ZERO_WIDTH.sub("", text)
    return text.strip()


def decode_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1251")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def _decode_member_name(info: zipfile.ZipInfo) -> str:
    """Имена без флага UTF-8 zipfile декодирует как CP437 — восстанавливаем исходные байты."""
    name = info.filename
    if info.flag_bits & _UTF8_FLAG:
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "cp866", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


# ---------------- DOCX ----------------

def _to_roman(n: int) -> str:
    vals = [(1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def _to_letters(n: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = letters[r] + out
    return out


def _fmt_number(value: int, fmt: str) -> str:
    if fmt in ("lowerLetter",):
        return _to_letters(value)
    if fmt in ("upperLetter",):
        return _to_letters(value).upper()
    if fmt == "lowerRoman":
        return _to_roman(value)
    if fmt == "upperRoman":
        return _to_roman(value).upper()
    if fmt in ("russianLower", "russianUpper"):
        alphabet = "абвгдежзиклмнопрстуфхцчшщэюя"
        ch = alphabet[(value - 1) % len(alphabet)]
        return ch.upper() if fmt == "russianUpper" else ch
    return str(value)  # decimal и всё прочее


class _Numbering:
    """Минимальная поддержка автонумерации Word: lvlText («%1.%2.»), numFmt, start."""

    def __init__(self, xml: bytes | None):
        self.levels: dict[str, dict[int, dict]] = {}   # numId -> ilvl -> {fmt, text, start}
        self.counters: dict[str, list[int]] = {}
        if not xml:
            return
        root = ET.fromstring(xml)
        abstract: dict[str, dict[int, dict]] = {}
        for an in root.iter(_W + "abstractNum"):
            aid = an.get(_W + "abstractNumId", "")
            lvls = {}
            for lvl in an.iter(_W + "lvl"):
                ilvl = int(lvl.get(_W + "ilvl", "0"))
                fmt = lvl.find(_W + "numFmt")
                txt = lvl.find(_W + "lvlText")
                start = lvl.find(_W + "start")
                lvls[ilvl] = {
                    "fmt": fmt.get(_W + "val", "decimal") if fmt is not None else "decimal",
                    "text": txt.get(_W + "val", "") if txt is not None else "",
                    "start": int(start.get(_W + "val", "1")) if start is not None else 1,
                }
            abstract[aid] = lvls
        for num in root.iter(_W + "num"):
            nid = num.get(_W + "numId", "")
            ref = num.find(_W + "abstractNumId")
            if ref is not None:
                self.levels[nid] = abstract.get(ref.get(_W + "val", ""), {})

    def label(self, num_id: str, ilvl: int) -> str:
        lvls = self.levels.get(num_id)
        if not lvls or ilvl not in lvls:
            return ""
        counters = self.counters.setdefault(num_id, [0] * 9)
        for i in range(9):
            if counters[i] == 0 and i < ilvl:
                counters[i] = lvls.get(i, {}).get("start", 1)
        counters[ilvl] = counters[ilvl] + 1 if counters[ilvl] else lvls[ilvl]["start"]
        for i in range(ilvl + 1, 9):
            counters[i] = 0
        if lvls[ilvl]["fmt"] in ("bullet", "none"):
            return "-" if lvls[ilvl]["fmt"] == "bullet" else ""
        text = lvls[ilvl]["text"]
        for i in range(ilvl + 1):
            fmt = lvls.get(i, {}).get("fmt", "decimal")
            text = text.replace(f"%{i + 1}", _fmt_number(counters[i], fmt))
        return text


def _paragraph_text(p: ET.Element, numbering: _Numbering) -> str:
    parts = []
    for el in p.iter():
        if el.tag == _W + "t" and el.text:
            parts.append(el.text)
        elif el.tag == _W + "tab":
            parts.append("\t")
        elif el.tag in (_W + "br", _W + "cr"):
            parts.append("\n")
    text = "".join(parts)
    num_pr = p.find(f"{_W}pPr/{_W}numPr")
    if num_pr is not None:
        nid_el, lvl_el = num_pr.find(_W + "numId"), num_pr.find(_W + "ilvl")
        nid = nid_el.get(_W + "val", "") if nid_el is not None else ""
        ilvl = int(lvl_el.get(_W + "val", "0")) if lvl_el is not None else 0
        prefix = numbering.label(nid, ilvl)
        if prefix and text.strip():
            text = f"{prefix} {text.lstrip()}"
    return text


def read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        body_xml = doc.read("word/document.xml")
        try:
            numbering_xml = doc.read("word/numbering.xml")
        except KeyError:
            numbering_xml = None
    numbering = _Numbering(numbering_xml)
    body = ET.fromstring(body_xml).find(_W + "body")
    if body is None:
        return ""
    lines: list[str] = []
    for el in body:
        if el.tag == _W + "p":
            lines.append(_paragraph_text(el, numbering))
        elif el.tag == _W + "tbl":
            for row in el.iter(_W + "tr"):
                cells = []
                for cell in row.findall(_W + "tc"):
                    cell_text = " ".join(_paragraph_text(p, numbering).strip() for p in cell.iter(_W + "p"))
                    cells.append(cell_text.strip())
                lines.append(" | ".join(cells))
    return "\n".join(lines)


# ---------------- архив ----------------

def _read_member(raw: bytes, suffix: str) -> str:
    if suffix == ".docx":
        return read_docx(raw)
    return decode_bytes(raw)


def make_raw_doc(name: str, raw: bytes) -> RawDoc | None:
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None
    text = normalize_text(_read_member(raw, suffix))
    if not text:
        return None
    return RawDoc(name=name, suffix=suffix, text=text,
                  sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def load_kb(path: Path | str) -> list[RawDoc]:
    """Все поддерживаемые документы архива, отсортированные по имени. Нечитаемые — пропуск с предупреждением."""
    docs: list[RawDoc] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = _decode_member_name(info)
            base = Path(name).name
            if name.startswith("__MACOSX/") or base.startswith((".", "~$")):
                continue
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                _warn(f"Пропущен файл неподдерживаемого формата: {name}")
                continue
            try:
                doc = make_raw_doc(name, archive.read(info))
            except (zipfile.BadZipFile, KeyError, ET.ParseError, UnicodeDecodeError, ValueError) as exc:
                _warn(f"Пропущен нечитаемый документ {name}: {type(exc).__name__}")
                continue
            if doc is None:
                _warn(f"Пропущен пустой документ: {name}")
                continue
            docs.append(doc)
    if not docs:
        raise ValueError("В архиве базы нет читаемых документов")
    docs.sort(key=lambda d: d.name)
    return docs
